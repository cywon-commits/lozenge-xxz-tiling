"""Shared tools for the XXZ stage-3 checks: tilings on small supercells (from the paper-I repo), exact diagonalisation.
H = J sum_<ij> [SxSx + SySy + Delta SzSz] - h.S,  S = 1/2, J = 1."""
import os, sys, json, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh
REPO = os.environ.get('LOZENGE_REPO', os.path.expanduser('~/cywon-commits/lozenge-quantum-tiling'))   # clone of the paper-I repository
sys.path.insert(0, REPO + '/code')
from periodic import Cell, DIRS, enumerate_tilings, analyse
from lswt import signs

def graph(C, tiling):
    rem = set(tiling); b = []
    for n in range(C.N):
        i, j = C.site(n)
        for d, (di, dj) in enumerate(DIRS):
            if (n, d) not in rem: b.append((n, C.idx(i + di, j + dj)))
    return np.array(b)

def simple(b):
    s = {tuple(sorted(x)) for x in b.tolist()}
    return len(s) == len(b) and all(i != j for i, j in s)

def classes(cell, limit=400000):
    """structurally distinct bipartite tilings of a supercell: key = (N_B-N_A, z histogram, sorted neighbour-z multiset)"""
    C = Cell(*cell); out = {}
    for t in enumerate_tilings(C, limit):
        a = analyse(C, t)
        if a is None: continue
        b = graph(C, t)
        if not simple(b): continue
        z = a['z']; nb = [[] for _ in range(C.N)]
        for i, j in b: nb[i].append(z[j]); nb[j].append(z[i])
        key = (round(a['m'] * 2 * C.N), tuple(sorted(a['hist'].items())), tuple(sorted((z[i], tuple(sorted(nb[i]))) for i in range(C.N))))
        out.setdefault(key, (t, b, a))
    return C, out

_pc = np.array([bin(i).count('1') for i in range(1 << 12)], np.uint8)
def basis(N, nup):
    s = np.arange(1 << N, dtype=np.uint32)
    c = _pc[s & 4095] + _pc[(s >> 12) & 4095] + (_pc[(s >> 24) & 4095] if N > 24 else 0)
    return s[c == nup]

def ham(N, b, st, Delta):
    """XXZ in a fixed S^z sector; returns (H, diag of sum SzSz)"""
    dim = len(st); zz = np.zeros(dim); rows = []; cols = []
    for i, j in b:
        bi = (st >> np.uint32(i)) & 1; bj = (st >> np.uint32(j)) & 1
        zz += np.where(bi == bj, 0.25, -0.25)
        f = np.nonzero(bi != bj)[0]
        t = st[f] ^ np.uint32((1 << int(i)) | (1 << int(j)))
        rows.append(f); cols.append(np.searchsorted(st, t))
    r = np.concatenate(rows); c = np.concatenate(cols)
    H = sp.csr_matrix((np.full(len(r), 0.5), (r, c)), shape=(dim, dim)) + sp.diags(Delta * zz)
    return H, zz

def ground(N, b, nup, Delta=1.0, k=1):
    st = basis(N, nup); H, zz = ham(N, b, st, Delta)
    if len(st) < 400:
        w, v = np.linalg.eigh(H.toarray()); return w[:k], v[:, :k], zz
    w, v = eigsh(H, k=k, which='SA', tol=1e-10); o = np.argsort(w)
    return w[o], v[:, o], zz

def PJ(J):
    """<P2(n_z)> of a symmetric-top (ferrimagnet, K = J) state |J, M = J>"""
    return J * (2 * J - 1) / ((J + 1) * (2 * J + 3))
