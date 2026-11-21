"""XXZ linear spin-wave energies of the Table-I crystals on both branches.
Neel || z (Delta >= 1): lambda^2 = eig[(Delta D - A)(Delta D + A)],  e = -3 Delta/2 + sum(lambda)/(4N)   (S = 1/2)
Neel perp z (Delta <= 1): lambda^2 = eig[(D - A)(D + Delta A)],      e = -3/2 + sum(lambda)/(4N)
Output: deviations from the square-dice line, hull vertices and step fields (field along the easy direction)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import json, sys
from structs import *
def e_lswt(N, b, z, Delta, branch):
    A = np.zeros((N, N)); np.add.at(A, (b[:, 0], b[:, 1]), 1.0); np.add.at(A, (b[:, 1], b[:, 0]), 1.0); Dg = np.diag(z.astype(float))
    M1, M2 = (Delta * Dg - A, Delta * Dg + A) if branch == 'par' else (Dg - A, Dg + Delta * A)
    w, V = np.linalg.eigh(M1); R = (V * np.sqrt(np.clip(w, 0, None))) @ V.T
    lam = np.sqrt(np.clip(np.linalg.eigvalsh(R @ M2 @ R), 0, None))
    return (-1.5 * Delta if branch == 'par' else -1.5) + lam.sum() / (4 * N)
def e_sq(Delta, branch, n=600):
    k = (np.arange(n) + 0.5) * 2 * np.pi / n; g = 0.5 * (np.cos(k)[:, None] + np.cos(k)[None, :])
    return (-1.5 * Delta + np.sqrt(Delta ** 2 - g ** 2).mean()) if branch == 'par' else (-1.5 + np.sqrt((1 - g) * (1 + Delta * g)).mean())
def hull(ms, es):
    o = np.argsort(ms); pts = [(ms[i], es[i], i) for i in o]; h = []
    for p in pts:
        while len(h) >= 2 and (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0]) >= (p[1] - h[-2][1]) * (h[-1][0] - h[-2][0]): h.pop()
        h.append(p)
    return h
G = {nm: load(nm, 250) for nm in ORDER if nm != 'square'}
grid = [('par', D) for D in (1.0, 1.05, 1.1, 1.25, 1.5, 2.0, 4.0)] + [('perp', D) for D in (0.95, 0.9, 0.75, 0.5, 0.25, 0.0)]
out = []; names = [n for n in ORDER if n != 'square']
print('deviation from the square-dice line, 1e-3 J (LSWT, S=1/2)\nbranch Delta  e_dice-e_sq | ' + ' '.join(f'{n:>6s}' for n in names[:-1]) + ' | hull vertices and step fields')
for br, D in grid:
    es = {nm: e_lswt(G[nm][0], G[nm][1], G[nm][2], D, br) for nm in names}; e0 = e_sq(D, br)
    ms = np.array([0.0] + [G[nm][3] for nm in names]); ee = np.array([e0] + [es[nm] for nm in names])
    dd = es['dice'] - e0; dev = {nm: (es[nm] - e0 - 6 * G[nm][3] * dd) * 1e3 for nm in names[:-1]}
    h = hull(ms, ee); lab = ['square'] + names; vs = [lab[p[2]] for p in h]; hs = [(h[i + 1][1] - h[i][1]) / (h[i + 1][0] - h[i][0]) for i in range(len(h) - 1)]
    print(f'{br:5s} {D:5.2f}   {dd:8.5f}  | ' + ' '.join(f'{dev[n]:6.2f}' for n in names[:-1]) + ' | ' + ' '.join(f'{v}' + (f' -{x:.3f}->' if i < len(hs) else '') for i, (v, x) in enumerate(zip(vs, hs + [0]))), flush=True)
    out.append(dict(branch=br, Delta=D, e_sq=e0, e={k: float(v) for k, v in es.items()}, dev=dev, vertices=vs, steps=hs))
json.dump(out, open('lswt_xxz.json', 'w'), indent=1, default=float)
