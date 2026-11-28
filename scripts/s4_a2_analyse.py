"""A2: easy-axis precision (h || z, Delta > 1).  From data/stage4/a2_ezz.jsonl:
 (1) e_par(Delta) = Ezz_from_E (= de/dDelta) per structure, e0 = E + h m (zero-field energy in the Lieb sector);
 (2) step slopes along consecutive crystals: dh/dDelta = Delta(e_par)/Delta(m);
 (3) fit of e0 (or e_par) per structure to a - b sqrt(x), x = Delta - 1;
 (4) lower convex hull of e0(m) at each Delta, step fields, and bootstrap (4000 resamples) presence probability of each vertex.
usage: python scripts/s4_a2_analyse.py [data/stage4/a2_ezz.jsonl]"""
import os, sys, json, collections, numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
f_in = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'data/stage4/a2_ezz.jsonl')
ORDER = ['square', '1/11', '1/9', '1/7', '1/6', '1/5', '1/4', '1/3', '1/2', '3/5', '2/3', 'dice']
PATH = ['square', '1/11', '1/5', '1/3', '3/5', 'dice']


def comb(rs, key, f=lambda r, k: r[k][0]):
    v = np.array([f(r, key) for r in rs]); e = np.array([r[key][1] for r in rs]); n = len(v)
    return float(v.mean()), float(max(np.sqrt((e ** 2).sum()) / n, v.std(ddof=1) / np.sqrt(n) if n > 1 else 0))


def hull(ms, es):
    """indices of the lower convex hull vertices of points (m, e), m sorted"""
    idx = np.argsort(ms); H = []
    for i in idx:
        while len(H) >= 2:
            a, b = H[-2], H[-1]
            if (es[b] - es[a]) * (ms[i] - ms[a]) >= (es[i] - es[a]) * (ms[b] - ms[a]): H.pop()
            else: break
        H.append(i)
    return H


def main():
    G = collections.defaultdict(list)
    for l in open(f_in):
        r = json.loads(l); G[(r['Delta'], r['name'])].append(r)
    Ds = sorted({k[0] for k in G}); out = {}
    T = {}
    for (D, n), rs in G.items():
        mL = rs[0]['m_lieb']
        e0 = comb(rs, 'E', lambda r, k: r['E'][0] + r['h'] * r['m'][0])
        ez = comb(rs, 'Ezz_from_E'); E = comb(rs, 'E')
        mrat = min(r['m'][0] / mL for r in rs) if mL > 0 else float('nan')
        sd = [abs(a['E'][0] - b['E'][0]) / np.hypot(a['E'][1], b['E'][1]) for i, a in enumerate(rs) for b in rs[i + 1:]]
        T[(D, n)] = dict(m=mL, e0=e0, ez=ez, E=E, N=rs[0]['N'], h=rs[0]['h'], nseed=len(rs), mrat=mrat, seed_dev=max(sd) if sd else 0)
    print('(1) e_par = Ezz_from_E (1e-5), e0 = E + h m, min m/m_L over seeds, max seed E deviation (sigma)')
    for D in Ds:
        print(f'\n Delta = {D}')
        for n in ORDER:
            t = T.get((D, n))
            if not t: continue
            print(f"  {n:6s} N={t['N']:5d} h={t['h']:.3f} seeds={t['nseed']}  e_par={t['ez'][0]:.5f}({t['ez'][1]*1e5:3.0f})  e0={t['e0'][0]:.5f}({t['e0'][1]*1e5:3.0f})"
                  f"  m/mL_min={t['mrat']:.5f}  sdev={t['seed_dev']:.1f}")
    print('\n(2) step slopes dh/dDelta = d(e_par)/dm along the Table-I path')
    for D in Ds:
        row = []
        for a, b in zip(PATH[:-1], PATH[1:]):
            ta, tb = T.get((D, a)), T.get((D, b))
            if not ta or not tb: continue
            dm = tb['m'] - ta['m']; s = (tb['ez'][0] - ta['ez'][0]) / dm; se = np.hypot(tb['ez'][1], ta['ez'][1]) / dm
            row.append((a, b, s, se))
        print(f'  Delta = {D}: ' + '  '.join(f'{a}->{b}: {s:+.3f}({se*1e3:.0f})' for a, b, s, se in row))
        out[f'slopes_{D}'] = row
    print('\n(3) fits e0(Delta) = a + c x - b sqrt(x)  and  e_par(Delta) = a - b sqrt(x) per structure  (x = Delta - 1)')
    for n in ORDER:
        pts = [(D - 1, T[(D, n)]) for D in Ds if (D, n) in T]
        if len(pts) < 3: continue
        x = np.array([p[0] for p in pts]); ez = np.array([p[1]['ez'][0] for p in pts]); w = 1 / np.array([p[1]['ez'][1] for p in pts])
        A = np.vstack([np.ones_like(x), -np.sqrt(x)]).T
        c, *_ = np.linalg.lstsq(A * w[:, None], ez * w, rcond=None); res = (ez - A @ c) * w
        print(f'  {n:6s} e_par: a={c[0]:.5f} b={c[1]:.4f}  chi2/dof={np.sum(res**2)/max(len(x)-2,1):.2f}')
        out[f'fit_{n}'] = dict(a=c[0], b=c[1], chi2=float(np.sum(res ** 2)))
    print('\n(4) lower convex hull of e0(m), bootstrap presence (4000 resamples, Gaussian with the combined errors)')
    rng = np.random.default_rng(1)
    for D in Ds:
        names = [n for n in ORDER if (D, n) in T]
        ms = np.array([T[(D, n)]['m'] for n in names]); es = np.array([T[(D, n)]['e0'][0] for n in names]); ee = np.array([T[(D, n)]['e0'][1] for n in names])
        H = hull(ms, es)
        steps = [(names[a], names[b], (es[b] - es[a]) / (ms[b] - ms[a])) for a, b in zip(H[:-1], H[1:])]
        cnt = np.zeros(len(names))
        for _ in range(4000):
            for i in hull(ms, es + rng.normal(0, ee)): cnt[i] += 1
        pres = cnt / 4000
        print(f'  Delta = {D}: vertices ' + ' -> '.join(names[i] for i in H))
        print('     step fields: ' + '  '.join(f'{a}->{b}: {h:.4f}' for a, b, h in steps))
        print('     presence   : ' + '  '.join(f'{n}:{p:.2f}' for n, p in zip(names, pres)))
        out[f'hull_{D}'] = dict(vertices=[names[i] for i in H], steps=steps, presence=dict(zip(names, pres.tolist())))
    json.dump(out, open(os.path.join(ROOT, 'data/stage4/a2_summary.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
