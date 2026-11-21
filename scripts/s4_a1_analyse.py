"""A1: easy-plane (Delta < 1), h || z.  m(h) and E(h) per structure and size, compared with the stage-2 formula (P-c),
h_- (first h with m >= 0.99 m_Lieb), lowest-energy structure and transition fields (P-d).
usage (from the repository root):  python scripts/s4_a1_analyse.py [data/stage4/a1_mh.jsonl] [figures/]
Seeds of the same (name, k, Delta, beta, h) are combined: mean of seed means, error = std/sqrt(n) (n >= 3) or the
quadrature-combined block error / n (n = 2), whichever is larger."""
import os, sys, json, collections, numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
f_in = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'data/stage4/a1_mh.jsonl')
figdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'figures')
KT = json.load(open(os.path.join(ROOT, 'data/stage3/K_table.json')))['table']
NAMES = ['square', '1/11', '1/5', '1/3', '1/2', '3/5', 'dice']
CHIS = (0.065, 0.125)


def combine(rs, key):
    v = np.array([r[key][0] for r in rs]); e = np.array([r[key][1] for r in rs]); n = len(v)
    err_blk = np.sqrt((e ** 2).sum()) / n
    err_seed = v.std(ddof=1) / np.sqrt(n) if n >= 2 else 0.0
    return float(v.mean()), float(max(err_blk, err_seed)), float(abs(v[0] - v[-1]) / np.hypot(e[0], e[-1])) if n >= 2 else 0.0


def predict(name, D, h, chi):
    """stage-2 formula: c = n.h = min(1, m h/(2|x|K + chi h^2)); m = m_L c + chi h (1 - c^2)"""
    x = abs(D - 1)
    if name == 'square': return chi * h
    mL = KT[name]['m']; K = KT[name]['K_raw']
    c = min(1.0, mL * h / (2 * x * K + chi * h * h))
    return mL * c + chi * h * (1 - c * c)


def h_minus_pred(name, D, chi):
    if name == 'square': return float('nan')
    x = abs(D - 1); mL = KT[name]['m']; K = KT[name]['K_raw']; disc = mL * mL - 8 * chi * x * K
    return (mL - np.sqrt(disc)) / (2 * chi) if disc >= 0 else float('nan')


def load(f):
    G = collections.defaultdict(list)
    for l in open(f):
        r = json.loads(l); G[(r['name'], r['k'], r['Delta'], r['beta'], round(r['h'], 4))].append(r)
    T = {}
    for key, rs in G.items():
        d = dict(N=rs[0]['N'], m_lieb=rs[0]['m_lieb'], nseed=len(rs))
        for q in ('E', 'm', 'Exy', 'Ezz_from_E'):
            mu, er, ds = combine(rs, q); d[q] = (mu, er); d['seed_dev_' + q] = ds
        d['eq_fail'] = sum(abs(r['E_halves'][0][0] - r['E_halves'][1][0]) > 2 * np.hypot(r['E_halves'][0][1], r['E_halves'][1][1]) for r in rs if 'E_halves' in r)
        d['tau_m'] = max(r.get('tau_m', 0) for r in rs); d['block'] = min(r.get('block', 0) for r in rs)
        T[key] = d
    return T


def main():
    T = load(f_in); out = {}
    Ds = sorted({k[2] for k in T}); betas = sorted({k[3] for k in T})
    for D in Ds:
        for beta in betas:
            sub = {k: v for k, v in T.items() if k[2] == D and k[3] == beta}
            if not sub: continue
            print(f'\n===== Delta = {D}, beta = {beta} =====')
            sizes = collections.defaultdict(set)
            for k in sub: sizes[k[0]].add(k[1])
            for name in NAMES:
                for kk in sorted(sizes.get(name, [])):
                    rows = sorted((k[4], v) for k, v in sub.items() if k[0] == name and k[1] == kk)
                    if not rows: continue
                    mL = rows[0][1]['m_lieb']; N = rows[0][1]['N']
                    hm = next((h for h, v in rows if mL > 0 and v['m'][0] >= 0.99 * mL), float('nan'))
                    hp = [h_minus_pred(name, D, c) for c in CHIS]
                    ratio = [hm / p if p == p and hm == hm else float('nan') for p in hp]
                    K_eff = hm * mL / (2 * abs(D - 1)) if hm == hm else float('nan')
                    print(f'-- {name} k={kk} N={N} m_L={mL:.4f}:  h_- meas={hm:.2f}  pred(chi=.065,.125)={hp[0]:.3f},{hp[1]:.3f}'
                          f'  ratio={ratio[0]:.2f},{ratio[1]:.2f}  h_- m/(2|x|)={K_eff:.4f} vs K={KT.get(name, {}).get("K_raw", float("nan")):.4f}')
                    print('     h     m/m_L (or m)         pred .065  pred .125        E                 tau_m  blk  eqfail sdev_m')
                    for h, v in rows:
                        s = mL if mL > 0 else 1.0
                        print(f'   {h:.2f}  {v["m"][0]/s:8.4f}({v["m"][1]/s*1e4:4.0f})  {predict(name, D, h, CHIS[0])/s:8.4f}  {predict(name, D, h, CHIS[1])/s:8.4f}'
                              f'   {v["E"][0]:.5f}({v["E"][1]*1e5:3.0f})  {v["tau_m"]:6.0f} {v["block"]:5d}  {v["eq_fail"]}   {v["seed_dev_m"]:.1f}')
                    out[f'{D}_{beta}_{name}_{kk}'] = dict(N=N, m_L=mL, h_minus=hm, h_minus_pred=hp, rows=[(h, v['m'], v['E']) for h, v in rows])
            # lowest-energy structure per h, per size class (small / large)
            print('\n  lowest energy (small size | large size):')
            hs = sorted({k[4] for k in sub})
            for cls in (0, 1):
                best = []
                for h in hs:
                    cand = []
                    for name in NAMES:
                        ks = sorted(sizes.get(name, []))
                        if len(ks) <= cls: continue
                        v = sub.get((name, ks[cls], D, beta, h))
                        if v: cand.append((v['E'][0], v['E'][1], name))
                    if cand:
                        cand.sort(); e0, e0e, n0 = cand[0]
                        gap = (cand[1][0] - e0) / np.hypot(cand[1][1], e0e) if len(cand) > 1 else float('inf')
                        best.append((h, n0, cand[1][2] if len(cand) > 1 else '', gap))
                print(f'   size {"small" if cls == 0 else "large"}: ' + '  '.join(f'{h:.2f}:{n}' + ('?' if g < 2 else '') for h, n, n2, g in best))
                seq = []
                for h, n, n2, g in best:
                    if not seq or seq[-1][1] != n: seq.append((h, n))
                print('     sequence (first h where lowest): ' + ' -> '.join(f'{n}@{h:.2f}' for h, n in seq))
                out[f'{D}_{beta}_lowest_{cls}'] = best
    json.dump(out, open(os.path.join(ROOT, 'data/stage4/a1_summary.json'), 'w'), indent=1, default=float)
    try:
        plot(T, Ds, betas)
    except Exception as e:
        print('plot failed:', e)


def plot(T, Ds, betas):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    os.makedirs(figdir, exist_ok=True)
    for D in Ds:
        beta = min(b for b in betas if any(k[2] == D and k[3] == b for k in T))
        fig, axs = plt.subplots(2, 4, figsize=(15, 7)); axs = axs.ravel()
        for ax, name in zip(axs, NAMES):
            ks = sorted({k[1] for k in T if k[0] == name and k[2] == D and k[3] == beta})
            for kk, mk in zip(ks, 'os^'):
                rows = sorted((k[4], v) for k, v in T.items() if k[0] == name and k[1] == kk and k[2] == D and k[3] == beta)
                h = np.array([r[0] for r in rows]); m = np.array([r[1]['m'][0] for r in rows]); e = np.array([r[1]['m'][1] for r in rows])
                ax.errorbar(h, m, e, fmt=mk, ms=3, label=f'N={rows[0][1]["N"]}')
            hh = np.linspace(0, 0.42, 200)
            for chi, ls in zip(CHIS, ('-', '--')):
                ax.plot(hh, [predict(name, D, x, chi) for x in hh], 'k' + ls, lw=0.8, label=f'P-c, chi={chi}')
            if name != 'square': ax.axhline(KT[name]['m'], color='gray', lw=0.5)
            ax.set_title(name); ax.set_xlabel('h/J'); ax.set_ylabel('m')
        axs[0].legend(fontsize=7)
        # energies relative to square
        ax = axs[7]
        for name in NAMES:
            ks = sorted({k[1] for k in T if k[0] == name and k[2] == D and k[3] == beta})
            if not ks: continue
            kk = ks[-1]
            rows = sorted((k[4], v) for k, v in T.items() if k[0] == name and k[1] == kk and k[2] == D and k[3] == beta)
            ksq = sorted({k[1] for k in T if k[0] == 'square' and k[2] == D and k[3] == beta})[-1]
            h = []; de = []
            for x, v in rows:
                s = T.get(('square', ksq, D, beta, x))
                if s: h.append(x); de.append(v['E'][0] - s['E'][0])
            ax.plot(h, de, '-o', ms=2, label=name)
        ax.set_title('E - E_square (large size)'); ax.set_xlabel('h/J'); ax.legend(fontsize=6)
        fig.suptitle(f'Delta = {D}, beta = {beta}, h || z'); fig.tight_layout()
        fig.savefig(os.path.join(figdir, f'a1_mh_D{D:.2f}.png'), dpi=120); plt.close(fig)


if __name__ == '__main__':
    main()
