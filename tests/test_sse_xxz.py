"""Unit test: XXZ SSE against exact finite-temperature averages (full diagonalisation, N = 12 lozenge-tiling clusters).
Run from the repository root:  python tests/test_sse_xxz.py     (about 2 minutes)"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from common import *
import sse_xxz
def exact(N, b, beta, D, h):
    w_all = []; obs = []
    for nup in range(N + 1):                                  # S^z is conserved: block-diagonalise
        st = basis(N, nup); H, zz = ham(N, b, st, D); Sz = nup - N / 2
        w, v = np.linalg.eigh(H.toarray()); ezz = (v ** 2 * zz[:, None]).sum(0)
        w_all.append(w - h * Sz); obs.append(np.stack([w - h * Sz, w - D * ezz, ezz, np.full(len(w), Sz)], 1))
    w = np.concatenate(w_all); O = np.concatenate(obs); p = np.exp(-beta * (w - w.min())); p /= p.sum()
    return dict(zip(('E', 'Exy', 'Ezz', 'm'), (p @ O) / N))
ok = True; worst = 0.0
print('cluster            Delta   h   beta  ' + '  '.join(f'{k:>24s}' for k in ('E', 'Exy', 'Ezz', 'm')))
for cell, d, hist in [((6,2,2),4,{3:8,6:4}), ((6,2,2),0,{4:12}), ((4,3,1),2,{3:4,4:5,5:2,6:1})]:
    C, cl = classes(cell); N = C.N; t, b, a = [v for k, v in cl.items() if k[0] == d and dict(k[1]) == hist][0]
    for D, h, beta in [(1.0, 0.3, 8.0), (1.05, 0.2, 8.0), (1.5, 0.0, 6.0), (0.97, 0.2, 8.0), (0.5, 0.3, 8.0)]:
        ex = exact(N, b, beta, D, h); r = sse_xxz.run(b, N, beta, D, h, ntherm=1000, nmeas=40000, seed_=7, nbins=40); row = []
        for k in ('E', 'Exy', 'Ezz', 'm'):
            dev = (r[k][0] - ex[k]) / max(r[k][1], 1e-9); worst = max(worst, abs(dev)); ok &= abs(dev) < 4
            row.append(f'{ex[k]:8.5f} {r[k][0]:8.5f} [{dev:+4.1f}]')
        print(f'{str(cell):9s} d={d} {D:5.2f} {h:4.1f} {beta:4.0f}   ' + '  '.join(row), flush=True)
print(f'worst deviation {worst:.1f} sigma ->', 'ALL OK' if ok else 'FAILED')
