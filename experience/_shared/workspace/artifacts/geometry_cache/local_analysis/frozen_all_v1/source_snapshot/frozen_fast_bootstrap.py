"""Exact group-weighted full-ROC EER by searching its unique-score crossings.

This changes computation cost only. No score quantization or ROC trimming is
used. Bootstrap group draws remain identical to the registered fixed contract.
"""
import numpy as np
from frozen_common import weighted_eers,eer

def fast_eers(y,s,codes,counts):
    order=np.argsort(-s,kind='stable');ss=s[order];yy=y[order];cc=codes[order]
    ends=np.r_[-1,np.flatnonzero(ss[:-1]!=ss[1:]),len(s)-1]
    groups=counts.shape[1]
    positive=[np.flatnonzero((cc==g)&(yy==1)) for g in range(groups)]
    negative=[np.flatnonzero((cc==g)&(yy==0)) for g in range(groups)]
    tb=counts@np.array([len(v) for v in positive]);ts=counts@np.array([len(v) for v in negative]);good=(tb>0)&(ts>0)
    def rates(k):
        positions=ends[k];pb=np.zeros(len(k));ps=np.zeros(len(k))
        for g,(b,a) in enumerate(zip(positive,negative)):
            if len(b):pb+=counts[:,g]*np.searchsorted(b,positions,side='right')
            if len(a):ps+=counts[:,g]*np.searchsorted(a,positions,side='right')
        return 1-pb/np.maximum(tb,1),ps/np.maximum(ts,1)
    lower=np.zeros(len(counts),int);upper=np.full(len(counts),len(ends)-1,int)
    while np.any(upper-lower>1):
        mid=(lower+upper)//2;fn,fp=rates(mid);above=fp>=fn;upper=np.where(above,mid,upper);lower=np.where(above,lower,mid)
    fn0,fp0=rates(lower);fn1,fp1=rates(upper);d0=fp0-fn0;d1=fp1-fn1;frac=np.divide(-d0,d1-d0,out=np.zeros(len(counts)),where=d1!=d0)
    result=100*(fp0+frac*(fp1-fp0));result[~good]=np.nan
    return result

def verify():
    maximum=0.
    for seed in range(12):
        rng=np.random.default_rng(seed);n=100+seed*13;y=rng.integers(0,2,n);s=rng.normal(size=n) if seed%2 else rng.integers(-2,3,n).astype(float);codes=rng.integers(0,9,n)
        counts=rng.multinomial(9,np.full(9,1/9),size=100)
        a=weighted_eers(y,s,codes,counts);b=fast_eers(y,s,codes,counts)
        err=float(np.nanmax(np.abs(a-b)));assert err<1e-10;maximum=max(maximum,err)
    return dict(status='pass',synthetic_random_or_tied_cases=12,draws_per_case=100,max_difference=maximum,full_unique_ROC=True)

if __name__=='__main__':print(verify())
