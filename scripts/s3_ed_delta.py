"""ED: step fields between ferrimagnetic tilings of the same cluster as a function of Delta >= 1 (h || z, state S^z = J).
h_{1->2}(Delta) = [E_2 - E_1]/(J_2 - J_1).  Checks: slope at Delta = 1 (Hellmann-Feynman), approach to the Ising form c1/Delta,
and the size of the higher Ising orders (h*Delta = c1 + c2/Delta^2 + ...)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import json
from common import *
DS = [1.0, 1.02, 1.05, 1.1, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
def ising2(N, b):                      # Eq. (4) of paper I, S = 1/2, per unit 1/Delta
    z = np.bincount(b.ravel(), minlength=N); return -0.5 * np.sum(1.0 / (z[b[:, 0]] + z[b[:, 1]] - 2))
out = []
for cell, seq in [((6,3,3), [(0,{4:18}),(2,{3:4,4:10,5:4}),(4,{3:9,4:3,5:3,6:3}),(6,{3:12,6:6})]),
                  ((10,2,1), [(0,{4:20}),(2,{3:4,4:12,5:4}),(4,{3:8,4:4,5:8})]),
                  ]:   # N = 24 (8,3,2) omitted: the S^z = 0 sector takes about an hour here
    C, cl = classes(cell); N = C.N; T = []
    for d, hist in seq:
        t, b, a = [v for k, v in cl.items() if k[0] == d and dict(k[1]) == hist][0]
        E = []; zz1 = None
        for D in DS:
            w, v, zz = ground(N, b, (N + d) // 2, D); E.append(w[0])
            if D == 1.0: zz1 = float(v[:, 0] ** 2 @ zz)
        T.append(dict(d=d, hist=hist, E=np.array(E), zz1=zz1, i2=ising2(N, b)))
    print(f'\ncell {cell} N={N}')
    for A, B in zip(T[:-1], T[1:]):
        dJ = (B['d'] - A['d']) / 2; h = (B['E'] - A['E']) / dJ; hf = (B['zz1'] - A['zz1']) / dJ; c1 = (B['i2'] - A['i2']) / dJ
        fd = (h[1] - h[0]) / (DS[1] - DS[0])
        print(f"  NB-NA {A['d']}->{B['d']}  {A['hist']} -> {B['hist']}")
        print('     Delta    :', ' '.join(f'{D:7.2f}' for D in DS))
        print('     h        :', ' '.join(f'{x:7.4f}' for x in h))
        print('     h*Delta  :', ' '.join(f'{x:7.4f}' for x in h * np.array(DS)), f'   Ising 2nd order c1 = {c1:.4f}')
        print(f'     slope at Delta=1: Hellmann-Feynman {hf:+.4f}, finite difference {fd:+.4f};  1/Delta law would give {-h[0]:+.4f};  -(c1+3R) = {-(c1+3*(h[0]-c1)):+.4f}')
        out.append(dict(cell=cell, a=A['d'], b=B['d'], D=DS, h=h.tolist(), c1=c1, hf=hf))
json.dump(out, open('ed_delta.json', 'w'), indent=1, default=float)
