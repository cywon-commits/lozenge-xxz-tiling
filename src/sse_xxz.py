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
    visited = 0
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
            visited += steps
        for p in range(M):
            op = ops[p]
            if op == -1: continue
            b = op // 2; base = 4 * p
            ops[p] = 2 * b if (leg[base] == leg[base + 2] and leg[base + 1] == leg[base + 3]) else 2 * b + 1
    for s in range(N):
        if first[s] != -1: spin[s] = leg[first[s]]
        elif np.random.random() < 0.5: spin[s] *= -1
    return n_ops, visited

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

def tau_int(x, c=6.0):
    """integrated autocorrelation time (in sweeps) with Sokal's automatic window W >= c tau"""
    x = np.asarray(x, float) - np.mean(x); n = len(x)
    if n < 10 or x.var() == 0: return 0.5
    f = np.fft.rfft(x, 2 * n); acf = np.fft.irfft(f * np.conj(f))[:n] / (x.var() * np.arange(n, 0, -1))
    tau = 0.5
    for W in range(1, n // 2):
        tau += acf[W]
        if W >= c * tau: break
    return float(max(tau, 0.5))

def run(bonds, N, beta, Delta=1.0, h=0.0, ntherm=500, nmeas=3000, seed_=0, nbins=20, eps=EPS, adapt=True, loop_div=10):
    """returns dict of (mean, error) per site: E (incl. Zeeman), Exy, Ezz, Ezz_from_E (= E - Exy - Zeeman, divided by Delta), m, |m|,
    plus diagnostics: nl (loops per sweep), visits (vertices visited per sweep / 2n), tau_E, tau_m (integrated autocorrelation
    times in sweeps), block (sweeps per block), sec_per_sweep, E_halves / m_halves (first / second half of the measurement).
    adapt=True: during thermalisation nl is tuned so that one sweep visits ~2n vertices (n = number of operators), then frozen.
    adapt=False: the old rule nl = max(10, 2n/loop_div).
    h = 0: a global spin flip (exact, weight-preserving) is attempted with probability 1/2 after every sweep, because tunnelling
    between the +M and -M sectors of a ferrimagnet is otherwise exponentially slow at Delta > 1."""
    import time
    b = np.ascontiguousarray(bonds, dtype=np.int64); nb = len(b)
    z = np.bincount(b.ravel(), minlength=N).astype(float)
    hb = np.ascontiguousarray(np.stack([h / z[b[:, 0]], h / z[b[:, 1]]], 1))
    Cb = 0.25 * Delta + 0.5 * (hb[:, 0] + hb[:, 1]) + eps
    np.random.seed(seed_); seed(seed_)
    spin = np.where(np.random.random(N) < 0.5, 1, -1).astype(np.int64)
    ops = np.full(max(40, int(beta * nb * 0.3)), -1, np.int64); n = 0; nl = 10
    acc_n = 0; acc_v = 0; acc_k = 0; acc_L = 0
    zflip = (h == 0.0)                                       # h = 0: global spin flip is an exact update (weights invariant)
    for t in range(ntherm):
        n, v = sweep(spin, ops, b, beta, nb, n, Cb, hb, nl, Delta)
        if zflip and np.random.random() < 0.5: spin *= -1
        M = len(ops); newM = int(1.3 * n) + 40
        if newM > M:
            new = np.full(newM, -1, np.int64); pos = np.sort(np.random.choice(newM, M, replace=False)); new[pos] = ops; ops = new
        if not adapt:
            if t > 20: nl = max(10, int(2 * n / loop_div))
            continue
        acc_n += n; acc_v += v; acc_k += 1; acc_L += nl
        # Loop lengths are broad (gapless cases: either ~10 or ~10^4 steps), so the mean loop length is estimated as
        # (summed visits)/(summed loops) over a window of >= 50 loops or >= 20 x 2n visits, and nl may grow by at most x2
        # per update (x4 at the final update).  First half of the thermalisation: windowed updates; second half: one window,
        # one update at the end.  nl is frozen during the measurement.
        last = (t == ntherm - 1)
        if (t < ntherm // 2 and (acc_L >= 50 or acc_v >= 40 * acc_n / acc_k)) or last:
            if acc_v > 0:
                target = 2.0 * (acc_n / acc_k) / (acc_v / acc_L)
                nl = max(1, int(round(min(target, (4 if last else 2) * nl))))
            else:
                nl = 2 * nl
            acc_n = acc_v = acc_k = acc_L = 0
        elif t == ntherm // 2:
            acc_n = acc_v = acc_k = acc_L = 0
    R = np.empty((nmeas, 4)); vis = 0.0; nsum = 0.0; t0 = time.time()
    for t in range(nmeas):
        n, v = sweep(spin, ops, b, beta, nb, n, Cb, hb, nl, Delta)
        if zflip and np.random.random() < 0.5: spin *= -1
        R[t] = meas(spin, ops, b); vis += v; nsum += n
    sec = (time.time() - t0) / max(nmeas, 1)
    Eser = (-R[:, 0] / beta + Cb.sum()) / N; mser = R[:, 2] / N
    half = nmeas // 2
    def hv(x):                                          # (mean, err) of each half, nbins/2 blocks per half
        out = []
        for y in (x[:half], x[half:]):
            k = max(nbins // 2, 2); y = y[:len(y) // k * k].reshape(k, -1).mean(1)
            out.append([float(y.mean()), float(y.std() / np.sqrt(k))])
        return out
    tauE, taum = tau_int(Eser), tau_int(mser)
    Rb = R[:nmeas // nbins * nbins].reshape(nbins, -1, 4).mean(1)
    E = (-Rb[:, 0] / beta + Cb.sum()) / N; Exy = -Rb[:, 1] / beta / N; m = Rb[:, 2] / N; Ezz = Rb[:, 3] / N
    f = lambda x: (float(x.mean()), float(x.std() / np.sqrt(nbins)))
    return dict(E=f(E), Exy=f(Exy), Ezz=f(Ezz), Ezz_from_E=f((E - Exy + h * m) / Delta), m=f(m), absm=f(np.abs(m)),
                nl=int(nl), visits=float(vis / max(2 * nsum, 1)), tau_E=tauE, tau_m=taum, block=nmeas // nbins,
                sec_per_sweep=sec, E_halves=hv(Eser), m_halves=hv(mser))
