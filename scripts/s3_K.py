"""Anisotropy stiffness K = e_perp - e_par of the Table-I crystals from the QMC energy decomposition of paper I
(data/paper/staircase_v8.json: Exy = in-plane, Ezz = normal exchange energy per site in the state S^z = M).
K_raw = (Exy + Ezz - 3 Ezz)/2.  Finite-cluster rotor correction (zero-point motion of the director in |J, J>):
  upper bound  K_raw / P_J  (rigid top, P_J = J(2J-1)/((J+1)(2J+3)));  estimate  K_raw (1 - r 3/J)/(1 - 3/J),
  r = 2/(1 + sqrt(1 + (c kappa/Delta_+)^2)), kappa = sqrt(4 pi/N): share of the k = 0 mode that survives in the thermodynamic limit.
Step slopes (exact identities): s_par = dh/dDelta(1+, h||z) = h/3 - (2/3) dK/dm,  s_perp = dh/dDelta(1-, in-plane h) = h/3 + (1/3) dK/dm."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import json, numpy as np
from common import REPO, PJ
P = json.load(open(REPO + '/data/paper/staircase_v8.json'))['points']
rows = [('3/47','M347',1175,3/47,-1.46,None),('1/11','M111',825,1/11,-2.25,0.12),('1/9','M19',972,1/9,-2.49,0.15),('1/7','M17',1050,1/7,-2.79,None),
        ('1/6','M6',1296,1/6,-3.13,None),('1/5','M15',1080,1/5,-3.54,None),('1/4','M14',864,1/4,-3.84,0.32),('1/3','M3',1296,1/3,-4.53,0.43),
        ('1/2','M2',1296,1/2,-4.98,0.66),('3/5','M35',1280,3/5,-5.25,0.82),('2/3','M23',1296,2/3,-4.37,0.90),('dice','dice',576,1.0,0.0,1.50)]
esq, dd, c = -0.669437, 0.031577, 1.66
T = {}
print('crystal    m       J     Exy       Ezz      K_raw   K_raw/P_J  K_est   (e0 Table I)')
for nm, key, N, f, dev, gap in rows:
    p = P[key]; m = f / 6; J = m * N; e0 = p['Exy'] + p['Ezz']; Kr = (e0 - 3 * p['Ezz']) / 2
    gap = gap or 8 * m; kap = np.sqrt(4 * np.pi / N); r = 2 / (1 + np.sqrt(1 + (c * kap / gap) ** 2))
    Ke = Kr * (1 - r * 3 / J) / (1 - 3 / J); e0T = esq + f * dd + dev * 1e-3
    T[nm] = dict(m=m, J=J, N=N, Exy=p['Exy'], Ezz=p['Ezz'], K_raw=Kr, K_top=Kr / PJ(J), K_est=Ke, e0=e0T, gap=gap)
    print(f'{nm:6s} {m:.5f} {J:6.1f} {p["Exy"]:.5f} {p["Ezz"]:.5f}  {Kr:.4f}   {Kr/PJ(J):.4f}    {Ke:.4f}   {e0T:.5f}')
sq = P['square']; print('square spin-flop runs (L=24): Ezz =', np.round(sq['Ezz'], 4), ' -> K_sq = 3 Ezz - e_sq =', np.round(3 * np.array(sq['Ezz']) - esq, 3))
ms = np.array([T[k]['m'] for k in T if k != '3/47']); 
for lab in ('K_raw', 'K_est', 'K_top'):
    Ks = np.array([T[k][lab] for k in T if k != '3/47']); a = np.polyfit(ms, Ks, 1); a2 = np.polyfit(ms[:5], Ks[:5], 1)
    print(f'{lab}: linear fit K = {a[1]:.3f} + {a[0]:.2f} m (all);  dilute part (m <= 1/24): K = {a2[1]:.3f} + {a2[0]:.2f} m')
T['square'] = dict(m=0.0, e0=esq)
print('\nstep slopes along the Table-I path (identities), for K_raw / K_est / K_top; square uses the dilute-fit intercept')
path = ['square', '1/11', '1/5', '1/3', '3/5', 'dice']; S = {}
for lab in ('K_raw', 'K_est', 'K_top'):
    Ks = np.array([T[k][lab] for k in T if k not in ('3/47', 'square')]); T['square'][lab] = np.polyfit(ms[:5], Ks[:5], 1)[1]
for A, B in zip(path[:-1], path[1:]):
    dm = T[B]['m'] - T[A]['m']; h = (T[B]['e0'] - T[A]['e0']) / dm; s = []
    for lab in ('K_raw', 'K_est', 'K_top'):
        dK = (T[B][lab] - T[A][lab]) / dm; s.append((h / 3 - 2 * dK / 3, h / 3 + dK / 3))
    S[A + '->' + B] = dict(h=h, s=s)
    print(f'  {A:6s}->{B:5s} h = {h:.3f}   s_par = ' + ' / '.join(f'{x[0]:+.2f}' for x in s) + '   s_perp = ' + ' / '.join(f'{x[1]:+.2f}' for x in s))
json.dump(dict(table=T, steps=S), open('K_table.json', 'w'), indent=1, default=float)
