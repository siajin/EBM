"""Find exact crop-only threshold for 100% common duration distribution.

For uniform source masses, use increasing quantile transport. Monotone interval
endpoints imply that a full feasible coupling exists iff every quantile pair
has intersecting reachable intervals. Thus c_min = max_q(1-min(Qb,Qs)/max(Qb,Qs)).
All quantile breakpoints and duration ratios are handled with integer arithmetic.
"""
from fractions import Fraction
from math import gcd
import json
import numpy as np
import pandas as pd
from analyze import OUT, load, match


def main():
    datasets, sources = load()
    rows=[]
    details={}
    for key,title,df in datasets:
        a=np.sort(np.rint(df.loc[df.y==1,'duration'].to_numpy()*16000).astype('int64'))
        b=np.sort(np.rint(df.loc[df.y==0,'duration'].to_numpy()*16000).astype('int64'))
        n,m=len(a),len(b)
        g=gcd(n,m);ca=m//g;cb=n//g;total=n*ca
        i=j=0;ra=ca;rb=cb;done=0;num=den=1;witness=None
        while i<n and j<m:
            aa,bb=int(a[i]),int(b[j])
            hi,lo=max(aa,bb),min(aa,bb)
            z=min(ra,rb)
            if hi*den>num*lo:
                num,den=hi,lo
                witness={'bonafide_duration_s':aa/16000,'spoof_duration_s':bb/16000,
                    'bonafide_sorted_index':i,'spoof_sorted_index':j,
                    'quantile_left':done/total,'quantile_right':(done+z)/total,
                    'mass_pct':100*z/total,
                    'cropped_class':'bonafide' if aa>bb else 'spoof'}
            done+=z;ra-=z;rb-=z
            if ra==0:i+=1;ra=ca
            if rb==0:j+=1;rb=cb
        assert done==total
        required=Fraction(num-den,num)
        at=match(a,b,num,den)
        assert at['matched_units']==at['total_units']
        # A slightly smaller cap must fail: maximality independently checked by
        # the ordered matching implementation rather than the quantile formula.
        lower=required-Fraction(1,1_000_000)
        lower_ratio=1/(1-lower)
        below=match(a,b,lower_ratio.numerator,lower_ratio.denominator)
        assert below['matched_units']<below['total_units']
        practical_tenths=(required.numerator*1000+required.denominator-1)//required.denominator
        rounded=Fraction(practical_tenths,1000)
        r=1/(1-rounded)
        check=match(a,b,r.numerator,r.denominator)
        assert check['matched_units']==check['total_units']
        row={'dataset':key,'minimum_crop_pct':float(required*100),
             'rounded_up_crop_pct':float(rounded*100),
             'exact_crop_fraction':str(required),
             'overlap_at_threshold_pct':100*at['max_overlap'],
             'overlap_0_0001pp_below_pct':100*below['max_overlap'],
             **witness}
        rows.append(row)
        details[key]={'threshold':row,'at_threshold_full_matching':True,
                      'below_threshold_not_full_matching':True}
        print(json.dumps(row),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'minimum_crop_100pct.csv',index=False)
    (OUT/'minimum_crop_validation.json').write_text(json.dumps({
        'sources':sources,'datasets':details,
        'policy':'Both classes; speed fixed at 1; every utterance keeps uniform within-class weight; per-file randomized crops allowed.',
        'criterion':'Exact common continuous-duration probability distribution, not bin-dependent histogram equality.',
        'formula':'max over exact quantile intervals of 1 - min(Q_b,Q_s)/max(Q_b,Q_s)',
        'sufficiency':'Monotone quantile coupling, each output duration min(Q_b,Q_s).',
        'necessity':'Reachable quantile intervals must intersect; bottleneck witness fails below threshold.',
        'rounding':'Practical cap is rounded upward to next 0.1 percentage point and rechecked.'},indent=2),encoding='utf-8')


if __name__=='__main__':main()
