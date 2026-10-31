"""E2 primitives: exact batched frozen peer codes and a matched A2h context head.

This module has no launcher. Real training remains deferred behind the existing
GPU queue and requires a separately frozen cache/learning plan.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import masked_history as mh
import masked_sequence as ms


def peer_code_batch(d, source_rows, config):
    source_rows = np.asarray(source_rows, dtype=np.int64)
    if source_rows.ndim != 1 or not len(source_rows):
        raise ValueError('nonempty source-row vector required')
    if np.any((source_rows < 0) | (source_rows >= len(d['train_items']))):
        raise ValueError('source row outside training context')
    users = np.searchsorted(d['train_offsets'], source_rows, side='right')-1
    rows, index, _ = mh.gather(d['train_offsets'], users)
    items = d['train_items'][source_rows].astype(np.int64)-1
    days = d['train_days'][source_rows].astype(np.int64)
    # Duplicate global users get independent batch rows and different masks.
    keep = d['train_items'][index].astype(np.int64)-1 != items[rows]
    if not np.all(np.bincount(rows[~keep], minlength=len(users)) == 1):
        raise ValueError('each source pair must occur exactly once')
    target_rows = np.arange(len(users), dtype=np.int64)
    batch = mh.prepare(d, users, config, keep=keep, queries=(target_rows, items, days))
    sequence, width = ms.sequence_inputs(d, users, rows, index, keep,
                                        target_rows, items, days, config)
    batch.update(sequence, seq_width=width)
    return batch


@torch.inference_mode()
def frozen_peer_vectors(model, batch):
    if model.training:
        raise ValueError('frozen producer must be in evaluation mode')
    code = model.encode(batch)
    vector = code
    if model.config.context == 'session':
        zero = torch.zeros(1, model.config.rank, dtype=code.dtype, device=code.device)
        parts = [code]+model.channel(batch, 'day', model.session_implicit,
                                     model.session_explicit, batch['target_items'], zero)
        if model.config.week_days > 1:
            parts += model.channel(batch, 'week', model.week_implicit,
                                   model.week_explicit, batch['target_items'], zero)
        vector = code+model.session_net(torch.cat(parts+[batch['target_features']], 1))
    items = batch['target_items']
    logits = (model.output[items]*vector[:, None, :]).sum(2)+model.bias[items]
    if model.config.time_bins > 0:
        logits += model.time_bias[items*model.config.time_bins+batch['target_bins']]
    stars = torch.arange(1, 6, dtype=code.dtype, device=code.device)
    shift = batch['features'][batch['target_rows'], :5]@stars-3.65
    logits += shift[:, None]*(stars-3)[None, :]/1.2
    direct = model.decode(code, batch['features'], batch['target_rows'], items, batch)
    error = float((logits-direct).abs().max())
    if error > 1e-6 or not torch.isfinite(vector).all():
        raise ValueError('peer vectors do not reproduce the frozen decoder')
    return vector, error


def attach_peers(batch, batch_users, target_days, index, lookup, *, seed, cap=64):
    local_rows = batch['target_rows'].detach().cpu().numpy()
    users = np.asarray(batch_users)[local_rows]
    items = batch['target_items'].detach().cpu().numpy()+1
    days = np.asarray(target_days, dtype=np.int64)
    if days.shape != items.shape:
        raise ValueError('target dates differ from rating targets')
    if np.any((days < 0) | (days > np.iinfo(np.uint16).max)):
        raise ValueError('target date outside dataset encoding')
    queries = np.empty(len(items), dtype=[('user', '<u4'), ('item', '<u2'), ('day', '<u2')])
    queries['user'], queries['item'], queries['day'] = users, items, days
    peers = index.queries(queries, seed=seed, cap=cap)
    mask = peers['mask']
    if np.any((peers['users'] == users[:, None]) & mask):
        raise ValueError('query user appears in peer context')
    rows = peers['rows'][mask]
    values = np.asarray(lookup(rows), dtype=np.float32)
    if values.ndim != 2 or len(values) != len(rows) or not np.isfinite(values).all():
        raise ValueError('missing or invalid frozen peer codes')
    codes = np.zeros((*mask.shape, values.shape[1]), dtype=np.float32)
    codes[mask] = values
    device = batch['target_items'].device
    extra = dict(peer_codes=codes, peer_ratings=peers['ratings'].astype(np.int64),
                 peer_gaps=peers['gaps'].astype(np.float32), peer_mask=mask)
    return dict(batch, **{name: torch.from_numpy(value).to(device) for name, value in extra.items()})


class JointContextModel(ms.SequenceModel):
    def __init__(self, items, config, weeks=400, *, use_peer_values=True):
        super().__init__(items, config, weeks)
        if config.rank % config.heads:
            raise ValueError('peer rank must be divisible by attention heads')
        self.use_peer_values = use_peer_values
        self.peer_rating = nn.Embedding(6, 8, padding_idx=0)
        self.peer_input = nn.Linear(config.rank+8+3, config.rank)
        self.peer_norm = nn.LayerNorm(config.rank)
        self.peer_query = nn.Linear(config.rank, config.rank)
        self.peer_key_value = nn.Linear(config.rank, 2*config.rank)
        self.peer_output = nn.Linear(config.rank, config.rank, bias=False)
        # Warm-start predictions are exactly the base model in both arms.
        nn.init.zeros_(self.peer_output.weight)

    def load_rating_state(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        expected = {name for name in self.state_dict() if name.startswith('peer_')}
        if set(missing) != expected or unexpected:
            raise ValueError('warm state must contain exactly the rating backbone')

    def peer_delta(self, code, batch):
        values = batch['peer_codes'].detach()  # Frozen cache is never optimized.
        ratings, gaps, mask = batch['peer_ratings'], batch['peer_gaps'], batch['peer_mask']
        n, k, rank = values.shape
        if (n != len(code) or rank != self.config.rank or k < 1
                or ratings.shape != mask.shape or gaps.shape != mask.shape
                or mask.shape != (n, k)):
            raise ValueError('peer tensor shape differs from rating targets')
        if not torch.isfinite(values).all() or not torch.isfinite(gaps).all():
            raise ValueError('nonfinite peer features')
        if torch.any((ratings[mask] < 1) | (ratings[mask] > 5)):
            raise ValueError('valid peers require observed stars 1..5')
        values = torch.where(mask[:, :, None], values, 0)
        ratings = torch.where(mask, ratings, 0)
        gaps = torch.where(mask, gaps, 0)
        time = torch.stack((torch.log1p(gaps.abs())/7, gaps.clamp(-365,365)/365,
                            (gaps == 0).to(gaps.dtype)), -1)
        features = torch.cat((values, self.peer_rating(ratings), time), -1)
        # Both arms execute the same shapes and dropout calls. Control retains
        # peer counts but zeros every rating, date and frozen-code value.
        features = features*float(self.use_peer_values)
        tokens = torch.tanh(self.peer_input(features))
        heads = self.config.heads
        query = self.peer_query(self.peer_norm(code)).view(n,1,heads,rank//heads).transpose(1,2)
        key, value = self.peer_key_value(tokens).view(n,k,2,heads,rank//heads).permute(2,0,3,1,4)
        present = mask.any(1)
        safe_mask = mask.clone()
        safe_mask[:, 0] |= ~present  # Avoid an all-masked softmax for cold movies.
        attended = F.scaled_dot_product_attention(query, key, value,
            attn_mask=safe_mask[:,None,None,:], dropout_p=self.config.dropout if self.training else 0.)
        delta = self.peer_output(attended.transpose(1,2).reshape(n,rank))
        return delta*present[:,None]

    def forward_batch(self, batch):
        code = self.encode(batch)
        code = code+self.peer_delta(code,batch)
        return self.decode(code,batch['features'],batch['target_rows'],batch['target_items'],batch)
