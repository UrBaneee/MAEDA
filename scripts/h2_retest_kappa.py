#!/usr/bin/env python3
"""
H2 judge gate (预注册 §7.2.1, 附录 HH): human test-retest Cohen's κ (pass 1 vs
pass 2) and judge-vs-human κ on the same 60 calibration reports, with
bootstrap 95% CIs (10000 resamples, seed 20261111). Reads the local label
snapshots and the frozen judge output; makes no API call. Writes the
disagreement list to data/h2_attribution/retest_disagreements.json.

    python scripts/h2_retest_kappa.py
"""
import json,random,collections
R='data/h2_attribution/'
m1=json.load(open(R+'labeling_pass1.mapping.json'))['mapping']
m2=json.load(open(R+'labeling_pass2.mapping.json'))['mapping']
def load(d,m):
    out={}
    for b,j in m.items():
        out[j]=json.load(open(f'{R}{d}/{b}.json'))
    return out
p1=load('labels_pass1_snapshot',m1); p2=load('labels_pass2_snapshot',m2)
jd={json.loads(l)['job_id']:json.loads(l) for l in open(R+'judge_formal.jsonl')}
jobs=sorted(m1.values())
assert set(jobs)==set(m2.values())==set(jd)
L=['correct','wrong','none']
print("pass2 missing label:",[j for j in jobs if p2[j].get('label') not in L])
def kappa(a,b):
    n=len(a); po=sum(x==y for x,y in zip(a,b))/n
    ca,cb=collections.Counter(a),collections.Counter(b)
    pe=sum(ca[l]*cb[l] for l in L)/n/n
    return (po-pe)/(1-pe) if pe<1 else 1.0
a=[p1[j]['label'] for j in jobs]; b=[p2[j]['label'] for j in jobs]; c=[jd[j]['label'] for j in jobs]
def boot(x,y,B=10000,seed=20261111):
    r=random.Random(seed); n=len(x); ks=[]
    for _ in range(B):
        idx=[r.randrange(n) for _ in range(n)]
        ks.append(kappa([x[i] for i in idx],[y[i] for i in idx]))
    ks.sort(); return ks[int(.025*B)],ks[int(.975*B)]
def diffboot(B=10000,seed=20261111):
    r=random.Random(seed); n=len(a); ds=[]
    for _ in range(B):
        idx=[r.randrange(n) for _ in range(n)]
        ds.append(kappa([c[i] for i in idx],[a[i] for i in idx])-kappa([a[i] for i in idx],[b[i] for i in idx]))
    ds.sort(); return ds[int(.025*B)],ds[int(.975*B)],sum(d>=0 for d in ds)/B
def table(x,y,nx,ny):
    print(f"{nx} \\ {ny}",L)
    for l in L: print(l,[sum(1 for i,k in zip(x,y) if i==l and k==m) for m in L])
for nm,(x,y) in {"human p1 vs p2 (ceiling)":(a,b),"judge vs p1 (gate)":(c,a),"judge vs p2 (descr.)":(c,b)}.items():
    print(nm, "agree",sum(i==k for i,k in zip(x,y)),"/60 kappa %.3f"%kappa(x,y),"CI %.3f-%.3f"%boot(x,y))
table(a,b,"p1","p2")
print("p2 dist",collections.Counter(b),"uncertain",sum(bool(p2[j].get('uncertain')) for j in jobs))
print("judge-ceiling diff CI %.3f %.3f, P(diff>=0) %.3f"%diffboot())
# per arm
for arm in "BR":
    js=[j for j in jobs if f"|{arm}|" in j]
    print(arm,"p1=p2",sum(p1[j]['label']==p2[j]['label'] for j in js),"/",len(js))
dis=[j for j in jobs if p1[j]['label']!=p2[j]['label']]
json.dump([{"job":j,"p1":p1[j],"p2":p2[j],"judge":jd[j]['label'],"b1":[k for k,v in m1.items() if v==j][0],"b2":[k for k,v in m2.items() if v==j][0]} for j in dis],open(R+'retest_disagreements.json','w'),ensure_ascii=False,indent=1)
print("disagreements:",len(dis))
