"""Smoothed name-token/category statistics over multiple training vessels.

The fitted object stores no vessel/callsign identity map. A token contributes
features only if at least three distinct training vessels share it.
"""
from collections import Counter,defaultdict

import numpy as np
import pandas as pd

from src.common import normalize_name

TOKEN_PRIOR_NAMES=[f"{side}_{s}" for side in ("first_token","last_token") for s in
                   ("support_log","population_logprior","flag_loglift","type_loglift","joint_loglift")]


def tokens(name):
    words=[w for w in normalize_name(name).split() if any(c.isalpha() for c in w)]
    return (words[0],words[-1]) if words else ("","")


def category(value):
    return value if isinstance(value,str) and value.strip() else None


def mode(values):
    counts=Counter(v for v in values if v is not None)
    return sorted(counts,key=lambda v:(-counts[v],v))[0] if counts else None


class NameTokenPrior:
    def fit(self,train,catalog):
        names=catalog.set_index("vessel_id").vessel_name
        records=[]
        for vid,group in train.groupby("target_vessel_id",sort=True):
            records.append((tokens(names.loc[vid]),mode(map(category,group.flag_name)),mode(map(category,group.vessel_type))))
        self.population=len(records)
        self.catalog_size=len(catalog)
        self.catalog_counts=[Counter(),Counter()]
        for pair in map(tokens,catalog.vessel_name):
            for side,t in enumerate(pair):self.catalog_counts[side][t]+=1
        self.global_counts=[Counter(),Counter(),Counter()]
        counts=[defaultdict(lambda:[0,Counter(),Counter(),Counter()]) for _ in range(2)]
        for pair,flag,kind in records:
            values=(flag,kind,(flag,kind) if flag and kind else None)
            for i,v in enumerate(values):
                if v is not None:self.global_counts[i][v]+=1
            for side,t in enumerate(pair):
                entry=counts[side][t];entry[0]+=1
                for i,v in enumerate(values):
                    if v is not None:entry[i+1][v]+=1
        self.counts=[{t:v for t,v in side.items() if v[0]>=3 and t} for side in counts]
        return self

    def features(self,row,candidate_names):
        flag,kind=category(row.get("flag_name")),category(row.get("vessel_type"))
        values=(flag,kind,(flag,kind) if flag and kind else None)
        output=np.zeros((len(candidate_names),10),dtype=np.float32)
        totals=[sum(c.values()) for c in self.global_counts]
        for j,name in enumerate(candidate_names):
            for side,t in enumerate(tokens(name)):
                entry=self.counts[side].get(t)
                if entry is None:continue
                n=entry[0]
                result=[np.log1p(n),np.log((n+5)/(self.population*self.catalog_counts[side][t]/self.catalog_size+5))]
                for i,v in enumerate(values):
                    marginal=self.global_counts[i].get(v,0)/max(totals[i],1)
                    lift=0.0
                    if v is not None and marginal>0:
                        conditional=(entry[i+1].get(v,0)+5*marginal)/(sum(entry[i+1].values())+5)
                        lift=np.log(conditional/marginal)
                    result.append(lift)
                output[j,side*5:side*5+5]=result
        assert np.isfinite(output).all()
        return output
