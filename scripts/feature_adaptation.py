"""Label-free replacements for a training-availability feature under shift."""
import numpy as np


def relative_unrated(features,bank):
    names=bank['extra_meta_names']
    extra_start=features.shape[1]-len(names)
    if extra_start-(len(bank['names'])+1)!=11:
        raise ValueError('unexpected generic metadata schema')
    index=extra_start+names.index('log_unrated_n')
    log_known=np.asarray(features[:,len(bank['names'])+1],dtype=np.float64)
    log_unknown=np.asarray(features[:,index],dtype=np.float64)
    replacement=np.log1p(np.expm1(log_unknown)/(np.expm1(log_known)+5))
    if not np.isfinite(replacement).all():raise ValueError('nonfinite relative unknown count')
    return index,replacement
