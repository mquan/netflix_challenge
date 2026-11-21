"""Training-state prototype, not an admitted production executable.

New Torch RNG realization of the scoped factored/bipolar/zero-frequency-bias
model. Callers must verify input_digest against full source/data admission and
provide ownership/resource guards. No CLI, file I/O, labels from query arrays,
qualifying audit, automatic retry or partial-epoch restart is provided.
"""
from dataclasses import asdict,dataclass
import math
import numpy as np
import upstream_factored_rbm_torch as kernel
from upstream_factored_rbm_prediction import predict

SHARED=('p','q','d','bv','bh')
LOCAL=('bu','but')


@dataclass(frozen=True)
class Config:
    hidden:int=400
    factors:int=50
    epochs:int=15
    batch_users:int=500
    seed:int=128
    lr:float=.02
    momentum:float=.9
    decay:float=.001
    lr_bu:float=.001
    lr_but:float=.0005
    tile_rows:int=16384

    def validate(self):
        if any(type(v) is not int or v<1 for v in (self.hidden,self.factors,self.epochs,self.batch_users,self.tile_rows)):
            raise ValueError('Positive integer full-budget dimensions required')
        if type(self.seed) is not int or not 0<=self.seed<2**64:
            raise ValueError('Explicit unsigned64 realization seed required')
        if (any(not math.isfinite(v) or v<0 for v in (self.lr,self.momentum,self.decay,self.lr_bu,self.lr_but))
                or not 0<=self.momentum<1 or self.lr<=0):
            raise ValueError('Finite native-scope optimizer parameters required')


def cd_steps(epoch):
    if type(epoch) is not int or epoch<1:raise ValueError('One-based epoch required')
    return min(1+(epoch-1)//5,5)


def user_order(users,seed,epoch):
    # Distinct from Rust StdRng, explicitly deterministic for this realization.
    mixed=(seed^(epoch*0x9E3779B97F4A7C15))&((1<<64)-1)
    return np.random.default_rng(mixed).permutation(users)


class State:
    def __init__(self,reader,input_digest,config=None,device='cpu'):
        import torch
        self.config=config or Config();self.config.validate()
        self._bind(reader,input_digest,device)
        self.generator=torch.Generator(device='cpu').manual_seed(self.config.seed)
        ni,nh,nf=reader.n_items,self.config.hidden,self.config.factors
        def normal(shape,sigma):
            return (torch.randn(shape,generator=self.generator,dtype=torch.float32,device='cpu')*sigma).to(self.device)
        # Preserve native factored initialization order; Torch RNG differs.
        d=normal((ni,nh),.01);q=normal((nf,nh),.01/math.sqrt(nf));p=normal((ni,5,nf),.01)
        counts=reader.visible_counts
        totals=counts.sum(axis=1,dtype=np.int64)
        global_counts=counts.sum(axis=0,dtype=np.int64)
        values=np.where((totals>0)[:,None],counts,global_counts[None,:]).astype(np.float32)
        total=np.where(totals>0,totals,global_counts.sum(dtype=np.int64)).astype(np.float32)
        eps=np.float32(1e-6)
        bv=np.log((values+eps)/(total[:,None]+np.float32(5)*eps)).astype(np.float32)
        self.shared=dict(p=p,q=q,d=d,bv=torch.tensor(bv,device=self.device),
                         bh=torch.zeros(nh,dtype=torch.float32,device=self.device))
        self.velocity={k:torch.zeros_like(v) for k,v in self.shared.items()}
        self.bias=dict(bu=torch.zeros((reader.n_users,5),dtype=torch.float32,device='cpu'),
                       but=torch.zeros((len(reader.day_values),5),dtype=torch.float32,device='cpu'))
        self.bias_velocity={k:torch.zeros_like(v) for k,v in self.bias.items()}
        self.completed_epoch=0;self.active_epoch=None;self.completed_batches=0

    def _bind(self,reader,input_digest,device):
        import torch
        if (not isinstance(input_digest,str) or len(input_digest)!=64
                or any(c not in '0123456789abcdef' for c in input_digest)):
            raise ValueError('Full external input-admission SHA256 required')
        self.reader=reader;self.input_digest=input_digest;self.device=torch.device(device)

    def _frame(self,batch):
        import torch
        indices=dict(bu=torch.tensor(batch['global_users'],dtype=torch.int64,device='cpu'),
                     but=torch.tensor(batch['global_days'],dtype=torch.int64,device='cpu'))
        model=dict(self.shared);velocity=dict(self.velocity)
        for key in LOCAL:
            model[key]=self.bias[key].index_select(0,indices[key]).to(self.device)
            velocity[key]=self.bias_velocity[key].index_select(0,indices[key]).to(self.device)
        support={k:torch.tensor(v,dtype=torch.int64,device=self.device) for k,v in batch['support'].items()}
        return indices,model,velocity,support

    def fit_epoch(self,epoch,guard,on_batch=None):
        import torch
        if self.active_epoch is not None or epoch!=self.completed_epoch+1 or epoch>self.config.epochs:
            raise ValueError('Only next full epoch; no partial-epoch retry')
        if not callable(guard):raise ValueError('External ownership/resource guard required')
        guard();self.active_epoch=epoch;self.completed_batches=0
        order=user_order(self.reader.nonempty_users,self.config.seed,epoch)
        for lo in range(0,len(order),self.config.batch_users):
            guard();ids=order[lo:lo+self.config.batch_users]
            batch=self.reader.batch(ids,training=True)
            indices,model,velocity,support=self._frame(batch)
            steps=cd_steps(epoch)
            hu=torch.rand((steps,len(ids),self.config.hidden),generator=self.generator,dtype=torch.float32,device='cpu').to(self.device)
            vu=torch.rand((steps,len(support['items'])),generator=self.generator,dtype=torch.float32,device='cpu').to(self.device)
            gradients=kernel.batch_gradients(model,support,hu,vu,self.config.tile_rows)
            values,speeds=kernel.update(model,velocity,gradients,lr=self.config.lr,mom=self.config.momentum,
                wd=self.config.decay,lr_bu=self.config.lr_bu,lr_but=self.config.lr_but)
            self.shared={k:values[k] for k in SHARED};self.velocity={k:speeds[k] for k in SHARED}
            for key in LOCAL:
                self.bias[key].index_copy_(0,indices[key],values[key].cpu())
                self.bias_velocity[key].index_copy_(0,indices[key],speeds[key].cpu())
            if self.device.type=='mps':torch.mps.synchronize()
            self.completed_batches+=1
            guard()
            if on_batch is not None:on_batch(epoch,self.completed_batches,ids.copy())
        guard();self.completed_epoch=epoch;self.active_epoch=None

    def predict_queries(self,guard):
        import torch
        if self.active_epoch is not None:raise ValueError('Prediction requires complete epoch boundary')
        if not callable(guard):raise ValueError('External ownership/resource guard required')
        guard();count=len(self.reader.qr['user_idxs'])
        output=np.empty(count,dtype=np.float32);seen=np.zeros(count,dtype=bool)
        users=np.flatnonzero(self.reader.qr['user_cnts'])
        for lo in range(0,len(users),self.config.batch_users):
            guard();batch=self.reader.batch(users[lo:lo+self.config.batch_users])
            _,model,_,support=self._frame(batch)
            query={k:torch.tensor(v,dtype=torch.int64,device=self.device) for k,v in batch['query'].items()}
            values=predict(model,support,query,self.config.tile_rows)['prediction'].cpu().numpy()
            rows=batch['query_rows']
            if (len(values)!=len(rows) or np.any(seen[rows]) or not np.isfinite(values).all()
                    or np.any(values<1-1e-5) or np.any(values>5+1e-5)):
                raise ValueError('Complete finite unique native query predictions required')
            output[rows]=values;seen[rows]=True;guard()
        if not seen.all():raise ValueError('Every original query row must be predicted')
        guard();return output

    def snapshot(self):
        if self.active_epoch is not None:raise ValueError('No partial-epoch checkpoint/restart')
        return dict(format='own_factored_bipolar_torch_v1',config=asdict(self.config),input_digest=self.input_digest,
            dimensions=[self.reader.n_users,self.reader.n_items,len(self.reader.day_values)],
            completed_epoch=self.completed_epoch,active_epoch=None,completed_batches=self.completed_batches,
            shared={k:v.cpu().clone() for k,v in self.shared.items()},velocity={k:v.cpu().clone() for k,v in self.velocity.items()},
            bias={k:v.clone() for k,v in self.bias.items()},bias_velocity={k:v.clone() for k,v in self.bias_velocity.items()},
            generator_state=self.generator.get_state().clone(),frequency_bias='zero_LR_BIF0')

    @classmethod
    def from_snapshot(cls,record,reader,expected_input_digest,device='cpu'):
        import torch
        if (record['format']!='own_factored_bipolar_torch_v1' or record['frequency_bias']!='zero_LR_BIF0'
                or record['active_epoch'] is not None or record['input_digest']!=expected_input_digest
                or record['dimensions']!=[reader.n_users,reader.n_items,len(reader.day_values)]):
            raise ValueError('Exact source-bound complete-epoch own snapshot required')
        obj=cls.__new__(cls);obj.config=Config(**record['config']);obj.config.validate()
        obj._bind(reader,expected_input_digest,device)
        ni,nh,nf=reader.n_items,obj.config.hidden,obj.config.factors
        shapes=dict(p=(ni,5,nf),q=(nf,nh),d=(ni,nh),bv=(ni,5),bh=(nh,),
                    bu=(reader.n_users,5),but=(len(reader.day_values),5))
        for group,keys in [('shared',SHARED),('velocity',SHARED),('bias',LOCAL),('bias_velocity',LOCAL)]:
            if set(record[group])!=set(keys):raise ValueError('Every actual model/momentum tensor required')
            for key in keys:
                v=record[group][key]
                if (v.dtype!=torch.float32 or v.device.type!='cpu' or tuple(v.shape)!=shapes[key]
                        or not bool(torch.isfinite(v).all())):
                    raise ValueError('Exact finite CPU snapshot shape/dtype required')
        if (type(record['completed_epoch']) is not int or not 0<=record['completed_epoch']<=obj.config.epochs
                or type(record['completed_batches']) is not int or record['completed_batches']<0):
            raise ValueError('Valid full-epoch counters required')
        batches=(len(reader.nonempty_users)+obj.config.batch_users-1)//obj.config.batch_users
        if record['completed_batches']!=(batches if record['completed_epoch'] else 0):
            raise ValueError('Every nonempty-user batch must complete before snapshot admission')
        obj.shared={k:v.clone().to(obj.device) for k,v in record['shared'].items()}
        obj.velocity={k:v.clone().to(obj.device) for k,v in record['velocity'].items()}
        obj.bias={k:v.clone() for k,v in record['bias'].items()}
        obj.bias_velocity={k:v.clone() for k,v in record['bias_velocity'].items()}
        obj.generator=torch.Generator(device='cpu');obj.generator.set_state(record['generator_state'])
        obj.completed_epoch=record['completed_epoch'];obj.completed_batches=record['completed_batches'];obj.active_epoch=None
        return obj
