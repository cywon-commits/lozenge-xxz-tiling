"""Graphs of the Table-I crystals of paper I, read from the repository data."""
import json, pickle, numpy as np
from common import *
from cellgraph import cell_graph
D = REPO + '/data/'
CELLS = {'1/11': 'qdec_cells/x0p0909_cell.json', '1/9': 'qdec_cells/x0p1111_cell.json', '1/7': 'qdec_cells/x0p1429_cell.json',
         '1/5': 'qdec_cells/x0p2_cell.json', '1/4': 'qdec_cells/x0p25_cell.json',
         '3/5': 'qdec_cells/m35_cell.json', '3/47': 'qdec_cells/new_3o47.json'}
def load(name, Nmin=300):
    """returns N, bonds, z, m"""
    if name in CELLS:
        r = json.load(open(D + CELLS[name])); p, q, s = r['cell']; k = 1
        while p * q * k * k < Nmin: k += 1
        N, b, z = cell_graph(r['cell'], r['tiling'], k)
    else:
        key = {'1/6': 'M6', '1/3': 'M3', '1/2': 'M2', '2/3': 'M23', 'dice': 'dice', 'square': 'square'}[name]
        rem = pickle.load(open(D + 'paper/structs_L36.pkl', 'rb'))[key]; L = 36; N = L * L
        rem = {tuple(sorted((int(a), int(c)))) for a, c in rem}; b = []
        for i in range(L):
            for j in range(L):
                n = i * L + j
                for di, dj in ((1, 0), (0, 1), (-1, 1)):
                    m_ = ((i + di) % L) * L + (j + dj) % L
                    if tuple(sorted((n, m_))) not in rem: b.append((n, m_))
        b = np.array(b); z = np.bincount(b.ravel(), minlength=N)
    assert len(b) == 2 * N
    eta = signs(N, b); m = abs(eta.sum()) / 2 / N
    return N, b, z, m, eta
ORDER = ['square', '3/47', '1/11', '1/9', '1/7', '1/6', '1/5', '1/4', '1/3', '1/2', '3/5', '2/3', 'dice']


def cell_from_L36(key, cell, L=36):
    """reduce an L = 36 torus structure of paper I to its supercell tiling [(site index in cell, direction)]"""
    rem = pickle.load(open(D + 'paper/structs_L36.pkl', 'rb'))[key]; C = Cell(*cell); out = set()
    for a, c in rem:
        a, c = int(a), int(c)
        for n, o in ((a, c), (c, a)):
            i, j = divmod(n, L)
            for d, (di, dj) in enumerate(DIRS):
                if ((i + di) % L) * L + (j + dj) % L == o: out.add((C.idx(i, j), d))
    assert len(out) == C.N, (len(out), C.N)
    return sorted(out)

SMALL = {'1/11': ('json', 'qdec_cells/x0p0909_cell.json'), '1/9': ('json', 'qdec_cells/x0p1111_cell.json'), '1/7': ('json', 'qdec_cells/x0p1429_cell.json'),
         '1/5': ('json', 'qdec_cells/x0p2_cell.json'), '1/4': ('json', 'qdec_cells/x0p25_cell.json'), '3/5': ('json', 'qdec_cells/m35_cell.json'),
         '1/6': ('L36', 'M6', (9, 4, 4)), '1/3': ('L36', 'M3', (9, 4, 4)), '1/2': ('L36', 'M2', (6, 4, 4)), '2/3': ('L36', 'M23', (9, 4, 7)),
         'dice': ('L36', 'dice', (3, 3, 0)), 'square': ('L36', 'square', (3, 3, 0))}
def load_cell(name, k):
    """k x k copies of the supercell of a Table-I structure: returns N, bonds, z, m, eta"""
    s = SMALL[name]
    if s[0] == 'json':
        r = json.load(open(D + s[1])); cell, til = r['cell'], r['tiling']
    else:
        cell, til = s[2], cell_from_L36(s[1], s[2])
    N, b, z = cell_graph(cell, til, k); assert len(b) == 2 * N
    eta = signs(N, b); assert eta is not None
    return N, b, z, abs(eta.sum()) / 2 / N, eta
