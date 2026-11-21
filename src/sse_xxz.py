"""Stochastic series expansion for the spin-1/2 XXZ antiferromagnet in a longitudinal field on an arbitrary bipartite graph,
H = J sum_<ij> [ SxSx + SySy + Delta SzSz ] - h sum_i Sz,   J = 1.
Directed loops with heat-bath exit probabilities; generalises sse_h.py of the paper-I repository (Delta = 1) by the
diagonal weight only:  W_diag = C - (Delta/4) s_i s_j + (h_i s_i + h_j s_j)/2,  C = Delta/4 + (h_i + h_j)/2 + eps,
off-diagonal weight 1/2.  Sign-free on bipartite graphs for any Delta.
Measured per site: E, E_xy = -<n_offdiag>/(beta N)  (= J<SxSx+SySy>),  E_zz = J <sum SzSz>/N (equal-time estimator), m = <Sz>."""
import numpy as np
from numba import njit
EPS = 0.25

@njit(cache=True)
def wdiag(si, sj, C, hi, hj, D):
    return C - 0.25 * D * si * sj + 0.5 * (hi * si + hj * sj)

@njit(cache=True)
def vweight(s0, s1, s2, s3, C, hi, hj, D):
    if s0 == s2 and s1 == s3:
        return wdiag(s0, s1, C, hi, hj, D)
    if s2 == -s0 and s3 == -s1 and s0 != s1:
        return 0.5
    return 0.0

@njit(cache=True)
def seed(s):
    np.random.seed(s)

@njit(cache=True)
def sweep(spin, ops, bsites, beta, nb, n_ops, Cb, hb, nloops, D):
    N = spin.shape[0]; M = ops.shape[0]
    for p in range(M):                                   # diagonal update
        op = ops[p]
        if op == -1:
            b = np.random.randint(nb); i, j = bsites[b, 0], bsites[b, 1]
            w = wdiag(spin[i], spin[j], Cb[b], hb[b, 0], hb[b, 1], D)
            if np.random.random() * (M - n_ops) < beta * nb * w:
                ops[p] = 2 * b; n_ops += 1
        elif op % 2 == 0:
            b = op // 2; i, j = bsites[b, 0], bsites[b, 1]
            w = wdiag(spin[i], spin[j], Cb[b], hb[b, 0], hb[b, 1], D)
            if np.random.random() * beta * nb * w < (M - n_ops + 1):
                ops[p] = -1; n_ops -= 1
        else:
            b = op // 2; spin[bsites[b, 0]] *= -1; spin[bsites[b, 1]] *= -1
    link = np.full(4 * M, -1, np.int64); leg = np.zeros(4 * M, np.int64)   # linked vertex list
    first = np.full(N, -1, np.int64); last = np.full(N, -1, np.int64); cur = spin.copy()
    for p in range(M):
        op = ops[p]
        if op == -1: continue
        b = op // 2; i, j = bsites[b, 0], bsites[b, 1]
        leg[4 * p] = cur[i]; leg[4 * p + 1] = cur[j]
        if op % 2 == 1:
            cur[i] *= -1; cur[j] *= -1
        leg[4 * p + 2] = cur[i]; leg[4 * p + 3] = cur[j]
        for k in range(2):
            s = bsites[b, k]; vl = 4 * p + k; vu = 4 * p + 2 + k
            if last[s] != -1:
                link[last[s]] = vl; link[vl] = last[s]
            else:
                first[s] = vl
            last[s] = vu
    for s in range(N):
        if first[s] != -1:
            link[first[s]] = last[s]; link[last[s]] = first[s]
    if n_ops > 0:                                        # directed loops, heat-bath exits
        probs = np.zeros(4)
        for _ in range(nloops):
            j0 = np.random.randint(4 * M)
            while ops[j0 // 4] == -1: j0 = np.random.randint(4 * M)
            j = j0; steps = 0
            while True:
                p = j // 4; le = j % 4; base = 4 * p; bb = ops[p] // 2
                leg[base + le] *= -1
                tot = 0.0
                for x in range(4):
                    leg[base + x] *= -1
                    w = vweight(leg[base], leg[base + 1], leg[base + 2], leg[base + 3], Cb[bb], hb[bb, 0], hb[bb, 1], D)
                    leg[base + x] *= -1
                    probs[x] = w; tot += w
                r = np.random.random() * tot; x = 0; acc = probs[0]
                while acc < r:
                    x += 1; acc += probs[x]
                leg[base + x] *= -1
                j = link[base + x]; steps += 1
                if j == j0 or steps > 100 * M: break
        for p in range(M):
            op = ops[p]
            if op == -1: continue
            b = op // 2; base = 4 * p
            ops[p] = 2 * b if (leg[base] == leg[base + 2] and leg[base + 1] == leg[base + 3]) else 2 * b + 1
    for s in range(N):
        if first[s] != -1: spin[s] = leg[first[s]]
        elif np.random.random() < 0.5: spin[s] *= -1
    return n_ops

@njit(cache=True)
def meas(spin, ops, b):
    n = 0; nod = 0
    for p in range(ops.shape[0]):
        if ops[p] != -1:
            n += 1
            if ops[p] % 2 == 1: nod += 1
    zz = 0.0
    for k in range(b.shape[0]): zz += 0.25 * spin[b[k, 0]] * spin[b[k, 1]]
    return n, nod, 0.5 * spin.sum(), zz

def run(bonds, N, beta, Delta=1.0, h=0.0, ntherm=500, nmeas=3000, seed_=0, nbins=20, eps=EPS, loop_div=10):
    """returns dict of (mean, error) per site: E (incl. Zeeman), Exy, Ezz, Ezz_from_E (= E - Exy - Zeeman, divided by Delta), m, |m|"""
    b = np.ascontiguousarray(bonds, dtype=np.int64); nb = len(b)
    z = np.bincount(b.ravel(), minlength=N).astype(float)
    hb = np.ascontiguousarray(np.stack([h / z[b[:, 0]], h / z[b[:, 1]]], 1))
    Cb = 0.25 * Delta + 0.5 * (hb[:, 0] + hb[:, 1]) + eps
    np.random.seed(seed_); seed(seed_)
    spin = np.where(np.random.random(N) < 0.5, 1, -1).astype(np.int64)
    ops = np.full(max(40, int(beta * nb * 0.3)), -1, np.int64); n = 0; nl = 10
    for t in range(ntherm):
        n = sweep(spin, ops, b, beta, nb, n, Cb, hb, nl, Delta)
        M = len(ops); newM = int(1.3 * n) + 40
        if newM > M:
            new = np.full(newM, -1, np.int64); pos = np.sort(np.random.choice(newM, M, replace=False)); new[pos] = ops; ops = new
        if t > 20: nl = max(10, int(2 * n / loop_div))
    R = np.empty((nmeas, 4))
    for t in range(nmeas):
        n = sweep(spin, ops, b, beta, nb, n, Cb, hb, nl, Delta); R[t] = meas(spin, ops, b)
    R = R[:nmeas // nbins * nbins].reshape(nbins, -1, 4).mean(1)
    E = (-R[:, 0] / beta + Cb.sum()) / N; Exy = -R[:, 1] / beta / N; m = R[:, 2] / N; Ezz = R[:, 3] / N
    f = lambda x: (float(x.mean()), float(x.std() / np.sqrt(nbins)))
    return dict(E=f(E), Exy=f(Exy), Ezz=f(Ezz), Ezz_from_E=f((E - Exy + h * m) / Delta), m=f(m), absm=f(np.abs(m)))
