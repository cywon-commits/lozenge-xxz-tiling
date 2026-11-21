"""ED with the field along x (no conserved spin component): slope of the quasi-plateau, dm_x/dh ~ x^2 (x = Delta - 1).
Full Hilbert space, dice clusters N = 12 (J = 2) and N = 18 (J = 3)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import json
from common import *
def full_ops(N, b):
    dim = 1 << N; st = np.arange(dim, dtype=np.uint32); zz = np.zeros(dim); r = []; c = []
    for i, j in b:
        bi = (st >> np.uint32(i)) & 1; bj = (st >> np.uint32(j)) & 1; zz += np.where(bi == bj, 0.25, -0.25)
        f = np.nonzero(bi != bj)[0]; r.append(f); c.append(st[f] ^ np.uint32((1 << int(i)) | (1 << int(j))))
    r = np.concatenate(r); c = np.concatenate(c); XY = sp.csr_matrix((np.full(len(r), 0.5), (r, c)), shape=(dim, dim))
    rr = np.repeat(st, N); cc = (st[:, None] ^ (np.uint32(1) << np.arange(N, dtype=np.uint32))[None, :]).ravel()
    Sx = sp.csr_matrix((np.full(len(rr), 0.5), (rr, cc)), shape=(dim, dim))
    return XY, zz, Sx
out = []
for cell, d in [((6,2,2),4),((6,3,3),6)]:
    C, cl = classes(cell); N = C.N; t, b, a = [v for k, v in cl.items() if k[0] == d and set(dict(k[1])) == {3, 6}][0]
    XY, zz, Sx = full_ops(N, b); J = d / 2; m0 = J / N; hs = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
    print(f'dice N={N}, J={J}, m = {m0:.4f}')
    for x in (-0.10, -0.05, 0.05, 0.10):
        mx = []
        for h in hs:
            H = XY + sp.diags((1 + x) * zz) - h * Sx
            w, v = eigsh(H, k=1, which='SA', tol=1e-11); mx.append(float(v[:, 0] @ (Sx @ v[:, 0])) / N)
        mx = np.array(mx); sl = np.gradient(mx, hs)
        print(f'  x={x:+.2f}: (m_x - m)/x^2 =', np.round((mx - m0) / x ** 2, 4), '  (dm/dh)/x^2 =', np.round(sl / x ** 2, 4))
        out.append(dict(N=N, x=x, h=hs.tolist(), mx=mx.tolist()))
json.dump(out, open('ed_hx.json', 'w'), indent=1)
