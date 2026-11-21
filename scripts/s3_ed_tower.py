"""ED tests of the multiplet (rotor) results at Delta = 1.
T1  ferrimagnetic tilings, state |J, S^z = J>:  e_par(meas) = <sum SzSz>/N,  K_meas = (e0 - 3 e_par)/2,  K = K_meas / P_J
T2  Wigner-Eckart tower: <sum SzSz> in |J, M> is linear in 3M^2 - J(J+1)
T3  square-type clusters (antiferromagnet, K-quantum-number 0): K_J = (3/2)(e_zz - e_J/3)(2J+3)/J from the tower states |J,J>"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import json
from common import *
out = {'T1': [], 'T2': [], 'T3': []}
def pick(cell, d, hist):
    C, cl = classes(cell)
    return C, [v for k, v in cl.items() if k[0] == d and dict(k[1]) == hist]
print('T1  cell N  NB-NA  hist                      J    e0        e_par     K_meas   P_J     K_meas/P_J')
cases = [((6,2,2),4,{3:8,6:4},'dice'),((6,3,3),6,{3:12,6:6},'dice'),((3,7,1),7,{3:14,6:7},'dice'),((6,4,1),8,{3:16,6:8},'dice'),
         ((10,2,1),4,{3:8,4:4,5:8},'3/5'),((10,2,1),2,{3:6,4:8,5:6},''),((10,2,1),2,{3:4,4:12,5:4},''),
         ((6,3,3),4,{3:9,4:3,5:3,6:3},'2/3'),((6,3,3),2,{3:6,4:6,5:6},'1/3'),((6,3,3),2,{3:4,4:10,5:4},'1/3'),
         ((8,3,2),6,{3:12,4:3,5:6,6:3},'3/4'),((8,3,2),4,{3:9,4:7,5:7,6:1},'1/2'),((6,4,2),4,{3:8,4:9,5:6,6:1},'1/2')]
for cell, d, hist, lab in cases:
    C, ts = pick(cell, d, hist)
    for t, b, a in ts[:1]:
        N = C.N; J = d / 2; nup = (N + d) // 2
        w, v, zz = ground(N, b, nup)
        e0 = w[0] / N; epar = float(v[:, 0] ** 2 @ zz) / N; Km = (e0 - 3 * epar) / 2
        print(f'{str(cell):11s}{N:3d} {d:3d}  {str(hist):26s}{J:4.1f} {e0:9.5f} {epar:9.5f} {Km:8.4f} {PJ(J):7.4f} {Km/PJ(J):8.4f}  {lab}')
        out['T1'].append(dict(cell=cell, N=N, d=d, hist=hist, J=J, e0=e0, epar=epar, Kmeas=Km, PJ=PJ(J), K=Km / PJ(J), label=lab))
print('\nT2  dice N=18 (J=3) and N=24 (J=4): <sum SzSz>/N in |J,M> vs a + b[3M^2 - J(J+1)]')
for cell, d, hist in [((6,3,3),6,{3:12,6:6}),((6,4,1),8,{3:16,6:8})]:
    C, ts = pick(cell, d, hist); t, b, a = ts[0]; N = C.N; J = d / 2; M = np.arange(J, -0.1, -1); ez = []; E = []
    for m_ in M:
        w, v, zz = ground(N, b, int(N / 2 + m_)); ez.append(float(v[:, 0] ** 2 @ zz) / N); E.append(w[0])
    x = 3 * M ** 2 - J * (J + 1); A = np.polyfit(x, ez, 1); res = np.abs(np.polyval(A, x) - ez).max()
    print(f'  N={N}: energies degenerate to {np.ptp(E):.1e};  e_zz(M) =', np.round(ez, 5), f' linear-fit max residual {res:.1e}')
    out['T2'].append(dict(N=N, J=J, M=M.tolist(), ezz=ez, resid=res))
print('\nT3  square-type clusters: tower states |J,J>')
for cell in [(4,4,0),(6,3,3),(10,2,1),(6,4,2)]:
    C, ts = pick(cell, 0, {4: C.N if False else Cell(*cell).N}); t, b, a = ts[0]; N = C.N; row = []
    for J in (0, 1, 2, 3):
        w, v, zz = ground(N, b, N // 2 + J); eJ = w[0] / N; ezz = float(v[:, 0] ** 2 @ zz) / N
        KJ = 1.5 * (ezz - eJ / 3) * (2 * J + 3) / J if J else float('nan'); row.append((J, eJ, ezz, KJ))
    print(f'  {cell} N={N}: ' + '  '.join(f'J={J}: e={e:.5f} ezz={z:.5f} K={k:.4f}' for J, e, z, k in row))
    out['T3'].append(dict(cell=cell, N=N, rows=row))
json.dump(out, open('ed_tower.json', 'w'), indent=1, default=float)
