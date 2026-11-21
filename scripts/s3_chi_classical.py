"""Classical transverse susceptibility at fixed director: chi_cl = (1/JN) 1^T (D+A)^+ 1
and the identity chi_cl = 1/8 - m^2 + q^T (D+A)^+ q / (16 N),  q = z - 4."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import json
from structs import *
out = {}
print('crystal   N     m       chi_cl    identity   (chi_cl-1/8)/(1/8)   log formula (m/4)[ln(1/4m)/pi + c]: c')
for nm in ORDER:
    N, b, z, m, eta = load(nm, 300)
    M = np.diag(z.astype(float))
    np.add.at(M, (b[:, 0], b[:, 1]), 1.0); np.add.at(M, (b[:, 1], b[:, 0]), 1.0)
    w, V = np.linalg.eigh(M); inv = np.where(w > 1e-9, 1 / np.where(w > 1e-9, w, 1), 0)
    one = np.ones(N); q = z - 4.0
    a = V.T @ one; chi = float(a ** 2 @ inv) / N
    c_ = V.T @ q; ident = 1 / 8 - m * m + float(c_ ** 2 @ inv) / 16 / N
    d = chi - 0.125
    cc = (d + m * m) * 4 / m - np.log(1 / (4 * m)) / np.pi if m > 0 else float('nan')
    print(f'{nm:7s} {N:5d} {m:.5f}  {chi:.6f}  {ident:.6f}   {d/0.125:+.4f}              {cc:.3f}')
    out[nm] = dict(N=N, m=m, chi_cl=chi, rel=d / 0.125)
json.dump(out, open('chi_classical.json', 'w'), indent=1)
