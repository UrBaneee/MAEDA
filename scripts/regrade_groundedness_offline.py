"""离线重评 GPT-5.6 Sol groundedness:不发任何 API 调用,只重算 scoring functional。

用途:验证「QWK 不过」有多少来自 claim 准入(哪些句子该算 claim)、
多少来自 binning 刻度、多少才真正来自 claim 等权。

    python3 regrade_offline.py            # 在 MAEDA/data/annotations 下运行

已知限制:claim 类型靠正则近似(从 reasoning 文本反解),恢复 398/418 条,
所以只看各变体之间的增量,不要引用绝对 QWK。正式版应在 prompt 里加一个
claim_type enum 字段重跑一次,而不是靠正则。
"""
import json, re, itertools, numpy as np
from sklearn.metrics import cohen_kappa_score

SOL = 'answer_key_openai_gpt_5_6_sol_groundedness_xhigh_16k_nonpredictive20_complete.json'
VIS = re.compile(r'\b(chart|plot|graph|visual|histogram|heatmap|figure|was generated|were generated|was created|illustrat|displa)', re.I)
META = re.compile(r'\b(data.quality check|data quality|analysis was|were analyzed|was analyzed|methodolog|the report|the analysis|summar|approach|ensure)', re.I)
INTERP = re.compile(r'\b(indicat|suggest|opportunit|potential|may |might |could |likely|significant market|key contributor|strateg|recommend|growing|reliab)', re.I)


def load(p):
    d = json.load(open(p))
    return d.get('labels', d.get('scores', d))


def band(x):
    return min(int(x * 5), 4)


def qwk(h, j):
    return cohen_kappa_score([band(x) for x in h], [band(x) for x in j],
                             weights='quadratic', labels=[0, 1, 2, 3, 4])


def parse(reasoning):
    out = []
    for p in reasoning.split('|'):
        m = re.match(r'^([✓✗])\s*(.*?)\s*\(', p.strip())
        if m:
            out.append((m.group(1) == '✓', m.group(2)))
    return out


def main():
    sol = json.load(open(SOL))['scores']
    refs = {'strict': load('human_reference_nonpredictive20_strict_lower.json'),
            'lenient': load('human_reference_nonpredictive20_lenient_upper.json')}
    ids = sorted(sol)
    claims = {c: parse(sol[c]['groundedness']['reasoning']) for c in ids}
    raw = [sol[c]['groundedness']['score'] for c in ids]

    variants = {
        'A0 现状(全部 claim 等权)': lambda cl: cl,
        'A1 去掉 图表/meta': lambda cl: [x for x in cl if not (VIS.search(x[1]) or META.search(x[1]))],
        'A2 去掉 图表/meta/解释性': lambda cl: [x for x in cl if not (VIS.search(x[1]) or META.search(x[1]) or INTERP.search(x[1]))],
    }
    print(f"{'变体':34}{'strict':>9}{'lenient':>10}{'均claim数':>11}")
    for name, f in variants.items():
        sc, kept = [], []
        for c in ids:
            cl = f(claims[c])
            kept.append(len(cl))
            sc.append(sum(s for s, _ in cl) / len(cl) if cl else 1.0)
        print(f"{name:34}{qwk([refs['strict'][c]['groundedness'] for c in ids], sc):9.3f}"
              f"{qwk([refs['lenient'][c]['groundedness'] for c in ids], sc):10.3f}{np.mean(kept):11.1f}")

    print("\n纯刻度重映射(不改 functional,零 API):")
    for rn, ref in refs.items():
        h = np.array([ref[c]['groundedness'] for c in ids])
        hb = np.array([band(x) for x in h])
        order = np.argsort(raw, kind='stable')
        matched = np.empty(len(ids), dtype=int)
        i = 0
        for b in range(5):
            for _ in range(int((hb == b).sum())):
                matched[order[i]] = b
                i += 1
        print(f"  {rn:8} 现状 {qwk(h, raw):.3f} -> 分布匹配后 "
              f"{cohen_kappa_score(hb, matched, weights='quadratic'):.3f}  (乐观估计,用到了同一批 20 条的人工分布)")

    print("\nn=20 的 bootstrap 95% CI:")
    rng = np.random.default_rng(0)
    for rn, ref in refs.items():
        h = [ref[c]['groundedness'] for c in ids]
        out = []
        for _ in range(4000):
            k = rng.integers(0, len(h), len(h))
            hh, jj = [band(h[x]) for x in k], [band(raw[x]) for x in k]
            if len(set(hh)) > 1 and len(set(jj)) > 1:
                out.append(cohen_kappa_score(hh, jj, weights='quadratic', labels=[0, 1, 2, 3, 4]))
        lo, hi = np.percentile(out, [2.5, 97.5])
        print(f"  {rn:8} [{lo:.3f}, {hi:.3f}]  P(重采样过 0.60 门)={np.mean(np.array(out) >= 0.60):.0%}")


if __name__ == '__main__':
    main()
