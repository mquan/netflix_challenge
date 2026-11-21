"""Array-only complete-trajectory extension; no alias or arm selection here.

The caller must admit the scalar pair on BOTH queries before deciding whether
it is retained. Existing design/context bits and the final intercept survive.
Actual source/model/data admission and neural fitting are separate operations.
"""
import numpy as np

BASE_RAW=486
BASE_DESIGN=562
BASE_NONLINEAR=712
BASE_NEURAL=713
BASE_CONTEXT=225
EXTRA_CONTEXT=150


def flags(retain_scalar,latent):
    if type(retain_scalar) is not bool or type(latent) is not bool:
        raise ValueError('Explicit paired-alias admission and predeclared arm required')


def dimensions(retain_scalar,latent):
    flags(retain_scalar,latent)
    return dict(raw_predictions=BASE_RAW+int(retain_scalar),
                design_columns=BASE_DESIGN+int(retain_scalar),
                nonlinear_columns=BASE_NONLINEAR+int(retain_scalar)+EXTRA_CONTEXT*int(latent),
                nn_inputs=BASE_NEURAL+int(retain_scalar)+EXTRA_CONTEXT*int(latent),
                context_columns=BASE_CONTEXT+EXTRA_CONTEXT*int(latent))


def extend(z,f,prediction,context=None,*,retain_scalar,latent):
    dims=dimensions(retain_scalar,latent)
    z=np.asarray(z);f=np.asarray(f);prediction=np.asarray(prediction)
    if (z.ndim!=2 or z.shape[1]!=BASE_DESIGN or z.dtype!=np.float64
            or f.shape!=(len(z),BASE_NONLINEAR) or f.dtype!=np.dtype('<f4')
            or prediction.shape!=(len(z),) or prediction.dtype!=np.dtype('<f4')
            or not np.isfinite(z).all() or not np.isfinite(f).all()
            or not np.isfinite(prediction).all() or not np.array_equal(z[:,-1],np.ones(len(z)))):
        raise ValueError('Complete finite original486-scalar operators and intercept required')
    if latent:
        context=np.asarray(context)
        if context.shape!=(len(z),EXTRA_CONTEXT) or context.dtype!=np.dtype('<f4') or not np.isfinite(context).all():
            raise ValueError('Every finite float32150-coordinate context row required')
    elif context is not None:raise ValueError('Scalar-only arm cannot silently discard supplied latent coordinates')
    scalar=np.clip(prediction,0,6).astype('<f4')
    design=np.column_stack((z[:,:-1],scalar,z[:,-1:])) if retain_scalar else z.copy()
    nonlinear=np.column_stack((f,scalar-np.float32(3.6))) if retain_scalar else f.copy()
    if latent:nonlinear=np.column_stack((nonlinear,context))
    if design.shape!=(len(z),dims['design_columns']) or nonlinear.shape!=(len(z),dims['nonlinear_columns']):
        raise ValueError('Mechanically declared full arm widths required')
    return design,nonlinear


def layout(original_signals,original_context,*,retain_scalar,latent):
    dims=dimensions(retain_scalar,latent)
    signals=list(original_signals);context=list(original_context)
    if (len(context)!=BASE_CONTEXT or not signals or signals[-1]!=BASE_NEURAL-1
            or any(type(v) is not int for v in signals+context)
            or sorted(signals+context)!=list(range(BASE_NEURAL))):
        raise ValueError('Every original signal/context coordinate and final ridge input required')
    extended_signals=signals[:-1]+([BASE_NONLINEAR] if retain_scalar else [])+[dims['nn_inputs']-1]
    extended_context=context+list(range(BASE_NONLINEAR+int(retain_scalar),dims['nn_inputs']-1)) if latent else context
    if sorted(extended_signals+extended_context)!=list(range(dims['nn_inputs'])):
        raise ValueError('Exact complete neural coordinate partition required')
    return dict(signals=extended_signals,context=extended_context,**dims)
