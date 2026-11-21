"""Predeclared H128/rank32 factories; no data preparation, training or CLI.

The two arms change only input/context widths and their architecture identity.
Every original optimizer, seed, epoch, patience and batch parameter is retained.
"""
import math
import upstream_complete_trajectory_mps as original
import upstream_rbm400_latent_operators as ops

CONFIGS={arm:dict(original.CONFIGS['h128'],architecture='own-rbm400-'+arm+'-h128-v1')
         for arm in ('scalar','latent')}


def model_factory(width,cfg):
    import torch
    arms=[name for name,value in CONFIGS.items() if cfg==value]
    if len(arms)!=1:raise ValueError('Exact original H128 recipe and declared arm required')
    latent=arms[0]=='latent'
    matches=[retain for retain in (False,True) if ops.dimensions(retain,latent)['nn_inputs']==width]
    if len(matches)!=1:raise ValueError('Mechanically admitted full arm width required')
    layout=ops.layout(original.signal_indices(ops.BASE_NEURAL),original.CONTEXT,
                      retain_scalar=matches[0],latent=latent)
    signals,context=layout['signals'],layout['context']
    class ContextResidual(torch.nn.Module):
        def __init__(self):
            super().__init__();h,r,d=cfg['nn_hidden'],cfg['context_rank'],cfg['nn_dropout']
            self.register_buffer('signals',torch.tensor(signals));self.register_buffer('context',torch.tensor(context))
            self.base=torch.nn.Sequential(torch.nn.Linear(width,h),torch.nn.GELU(),torch.nn.Dropout(d),torch.nn.Linear(h,h),torch.nn.GELU(),torch.nn.Dropout(d),torch.nn.Linear(h,1))
            self.project=torch.nn.Linear(len(signals),r,bias=False)
            self.gate=torch.nn.Sequential(torch.nn.Linear(len(context),h),torch.nn.GELU(),torch.nn.Dropout(d),torch.nn.Linear(h,r))
            for layer in [self.base[-1],self.gate[-1]]:torch.nn.init.zeros_(layer.weight);torch.nn.init.zeros_(layer.bias)
        def forward(self,values):
            factors=self.project(values.index_select(1,self.signals));weights=self.gate(values.index_select(1,self.context))
            return self.base(values)+(factors*weights).sum(1,keepdim=True)/math.sqrt(cfg['context_rank'])
    return ContextResidual()
