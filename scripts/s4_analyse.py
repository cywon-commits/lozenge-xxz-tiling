"""Stage-4 tables from the JSONL files in the current directory (run from data/stage4).
(1) easy axis (Delta > 1, h || z): e_par(Delta) = Ezz = de/dDelta per crystal, step slopes dh/dDelta = Delta(Ezz)/Delta(m),
    zero-field energies e0 = E + h m, hull steps at each Delta;
(2) size scaling of K_raw for the 1/11 crystal at Delta = 1;  (3) m(h) and energies at Delta < 1, h || z."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import json, numpy as np, collections
from common import PJ
def rd(f):
    return [json.loads(l) for l in open(f)] if _os.path.exists(f) else []
PATH = ['square', '1/11', '1/5', '1/3', '3/5', 'dice']; out = {}
R = collections.defaultdict(dict)
for r in rd('ezz_easy_axis.jsonl'): R[r['Delta']][r['name']] = r
if R:
    print('(1) easy axis, h || z.  e_par = <sum SzSz>/N (= de/dDelta), e0 = E + h m')
    for D in sorted(R):
        T = R[D]; names = [n for n in PATH if n in T]
        print(f'  Delta = {D}: ' + '  '.join(f"{n}: e_par={T[n]['Ezz_from_E'][0]:.5f}({T[n]['Ezz_from_E'][1]*1e5:.0f}) e0={T[n]['E'][0]+T[n]['h']*T[n]['m'][0]:.5f}({T[n]['E'][1]*1e5:.0f})" for n in names))
        rows = []
        for a, b in zip(names[:-1], names[1:]):
            dm = T[b]['m_lieb'] - T[a]['m_lieb']
            s = (T[b]['Ezz_from_E'][0] - T[a]['Ezz_from_E'][0]) / dm; se = np.hypot(T[b]['Ezz_from_E'][1], T[a]['Ezz_from_E'][1]) / dm
            e = lambda n: T[n]['E'][0] + T[n]['h'] * T[n]['m'][0]
            hh = (e(b) - e(a)) / dm; he = np.hypot(T[b]['E'][1], T[a]['E'][1]) / dm
            rows.append((a, b, s, se, hh, he))
        print('     step slopes dh/dDelta: ' + '  '.join(f'{a}->{b}: {s:+.3f}({se:.3f})' for a, b, s, se, hh, he in rows))
        print('     step fields h        : ' + '  '.join(f'{a}->{b}: {hh:.4f}({he:.4f})' for a, b, s, se, hh, he in rows))
        out[f'easy_axis_{D}'] = [dict(a=a, b=b, slope=s, slope_err=se, h=hh, h_err=he) for a, b, s, se, hh, he in rows]
S = rd('ezz_size_1o11.jsonl')
if S:
    print('\n(2) 1/11 crystal at Delta = 1: K_raw = (e0 - 3 e_par)/2 vs cluster spin J = m N')
    for r in sorted(S, key=lambda r: r['N']):
        J = r['m_lieb'] * r['N']; e0 = r['E'][0] + r['h'] * r['m'][0]; ez = r['Ezz_from_E']; K = (e0 - 3 * ez[0]) / 2; Ke = 1.5 * ez[1]
        print(f"  k={r['k']} N={r['N']:5d} J={J:5.1f}  m_meas/m={r['m'][0]/r['m_lieb']:.4f}  e_par={ez[0]:.5f}({ez[1]*1e5:.0f})  K_raw={K:.4f}({Ke:.4f})  K_raw/P_J={K/PJ(J):.4f}")
    out['size_1o11'] = S
Mh = collections.defaultdict(dict)
for r in rd('mh_easy_plane_hz.jsonl'): Mh[r['h']][r['name']] = r
if Mh:
    names = [n for n in ['square', '1/11', '1/5', '1/3', '1/2', '3/5', 'dice'] if any(n in v for v in Mh.values())]
    print('\n(3) Delta = %.2f, h || z: magnetization m_z / m_Lieb (square: m_z) and energy per site; * = lowest energy' % next(iter(next(iter(Mh.values())).values()))['Delta'])
    print('   h     ' + ''.join(f'{n:>22s}' for n in names))
    for h in sorted(Mh):
        T = Mh[h]; emin = min(T[n]['E'][0] for n in names if n in T); row = ''
        for n in names:
            if n not in T: row += ' ' * 22; continue
            r = T[n]; mm = r['m'][0] / r['m_lieb'] if r['m_lieb'] > 0 else r['m'][0]
            row += f"  {mm:6.3f} {r['E'][0]:.5f}({r['E'][1]*1e5:.0f})" + ('*' if r['E'][0] == emin else ' ')
        print(f'  {h:.2f}  ' + row)
json.dump(out, open('stage4_summary.json', 'w'), indent=1, default=float)
