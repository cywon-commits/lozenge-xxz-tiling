"""Effective director model of stage 2 evaluated with the QMC-derived K (s3_K.py) and the classical susceptibility ratios (s3_chi_classical.py).
g_T(n) = e_T + x e_perp,T - x K_T n_z^2 - m h (n.h^) - (chi_T/2) h^2 [1 - (n.h^)^2],  x = Delta - 1, all 13 Table-I structures.
Prints the sequence of tilings and entry fields for the four field/anisotropy combinations."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import json, sys, numpy as np
KT = json.load(open('K_table.json'))['table']; CH = json.load(open('chi_classical.json'))
names = ['square', '1/11', '1/9', '1/7', '1/6', '1/5', '1/4', '1/3', '1/2', '3/5', '2/3', 'dice']
th = np.linspace(0, np.pi, 1441)
def run(x, alpha, lab, out):
    T = []
    for n in names:
        t = KT[n]; K = t[lab]; chi = 0.065 * (1 + CH[n]['rel']); eperp = t['e0'] / 3 + K / 3
        T.append((n, t['m'], t['e0'], K, chi, eperp))
    seq = []; last = None
    for h in np.arange(0.001, 0.6, 0.001):
        best = None
        for n, m, e, K, chi, ep in T:
            nz = np.cos(th); nh = np.cos(th - alpha)
            v = e + x * ep - x * K * nz ** 2 - m * h * nh - chi / 2 * h * h * (1 - nh ** 2); i = v.argmin(); val, c = v[i], nh[i]
            v2 = e + x * ep - chi / 2 * h * h                      # director along y (perpendicular to both z and h)
            if v2 < val: val, c = v2, 0.0
            if best is None or val < best[0]: best = (val, n, c, m)
        if best[1] != last:
            seq.append((best[1], round(float(h), 3), round(float(best[2]), 2))); last = best[1]
    out.append(dict(x=x, alpha=alpha, K=lab, seq=seq))
    return ' '.join(f'{n}@{h:.3f}' + ('' if (c > 0.995 or n == 'square') else f'(c={c:.2f})') for n, h, c in seq[1:])
out = []
for title, sgn, alpha in [('Delta>1, h || z (easy)', +1, 0.0), ('Delta<1, h in plane (easy)', -1, np.pi / 2),
                          ('Delta<1, h || z (hard)', -1, 0.0), ('Delta>1, h perp z (hard)', +1, np.pi / 2)]:
    print('\n' + title)
    for ax in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08):
        for lab in ('K_est',) if 'easy' in title else ('K_raw', 'K_est', 'K_top'):
            print(f'  |x|={ax:5.3f} {lab}: ' + run(sgn * ax, alpha, lab, out))
json.dump(out, open('model_paths.json', 'w'), indent=1)
