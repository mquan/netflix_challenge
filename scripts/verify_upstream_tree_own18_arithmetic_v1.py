"""Independent compensated products, all262143 supports and18-weight KKT checks."""
import itertools
import math
import numpy as np


def products(columns,y,guard=lambda:None):
    n=len(columns);gram=np.empty((n,n));rhs=np.empty(n)
    for i in range(n):
        guard()
        rhs[i]=math.fsum(float(a)*float(b) for a,b in zip(columns[i],y))
        for j in range(i+1):
            guard();gram[i,j]=gram[j,i]=math.fsum(float(a)*float(b) for a,b in zip(columns[i],columns[j]))
    return gram,rhs


def support_solution(gram,rhs,guard=lambda:None):
    if gram.shape!=(18,18) or rhs.shape!=(18,):raise ValueError('All18 full families required')
    options=[];examined=0
    for count in range(1,19):
        for support in itertools.combinations(range(18),count):
            examined+=1
            if examined%256==0:guard()
            ids=list(support);system=np.zeros((count+1,count+1))
            system[:count,:count]=gram[np.ix_(ids,ids)];system[:count,-1]=1;system[-1,:count]=1
            try:solution=np.linalg.solve(system,np.r_[rhs[ids],1.])
            except np.linalg.LinAlgError:continue
            weights=np.zeros(18);weights[ids]=solution[:-1]
            if weights.min()<-1e-9:continue
            inactive=[j for j in range(18) if j not in ids];gradient=gram@weights-rhs
            if inactive and min(gradient[inactive]+solution[-1])<-1e-9:continue
            options.append((float(weights@gram@weights-2*rhs@weights),weights))
    if examined!=262143 or not options:raise ValueError('Full independent simplex supports missing')
    return min(options,key=lambda row:row[0])[1],examined


def verify_simplex_weights(gram,rhs,actual,independent):
    actual=np.asarray(actual,dtype=np.float64)
    if actual.shape!=(18,) or not np.isfinite(actual).all() or actual.min()<0 or abs(actual.sum()-1)>1e-12:
        raise ValueError('All18 actual finite nonnegative sum-one coefficients required')
    gradient=gram@actual-rhs;dual=-float(actual@gradient);reduced=gradient+dual
    value=float(actual@gram@actual-2*rhs@actual)
    optimum=float(independent@gram@independent-2*rhs@independent)
    gap=abs(value-optimum);complementarity=float(np.max(abs(actual*reduced)))
    if np.min(reduced)<-1e-9 or complementarity>1e-10 or gap>1e-12:
        raise ValueError('Actual coefficients must attain independently enumerated optimum and full KKT conditions')
    return dict(objective=value,independent_optimum=optimum,absolute_gap=gap,
                minimum_reduced_gradient=float(np.min(reduced)),complementarity_error=complementarity,
                nonunique_coefficients_allowed=True)


def stored(path,arrays,weights,intercept=0.,guard=lambda:None):
    if len(arrays)!=18 or len(weights)!=18:raise ValueError('Every original eighteen coefficient and column required')
    actual=np.fromfile(path,dtype='<f4')
    if actual.shape!=arrays[0].shape or not np.isfinite(actual).all():raise ValueError('Complete finite stored candidate required')
    for lo in range(0,len(actual),4096):
        guard()
        hi=min(lo+4096,len(actual));expected=np.full(hi-lo,float(intercept),dtype=np.float64)
        for column,weight in zip(arrays,weights):expected=expected+float(weight)*column[lo:hi].astype(np.float64)
        expected=np.asarray(np.clip(expected,1.05,4.95),dtype='<f4')
        if not np.array_equal(actual[lo:hi].view('u4'),expected.view('u4')):raise ValueError('Every actual stored arithmetic bit must replay')
    return actual
