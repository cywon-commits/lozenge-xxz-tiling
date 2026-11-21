"""A0 benchmark: one SSE run with full diagnostics (sec/sweep, tau_E, tau_m, halves); adapt=0 reproduces the old loop rule.
usage: python s4_bench.py <name> <k> <Delta> <beta> <h> <ntherm> <nmeas> <seed> <adapt 0|1> <out.jsonl>"""
import os, sys, json, time, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from structs import load_cell
import sse_xxz
a = sys.argv[1:]; name, k, D, beta, h, nt, nm, sd, ad, out = a[0], int(a[1]), float(a[2]), float(a[3]), float(a[4]), int(a[5]), int(a[6]), int(a[7]), bool(int(a[8])), a[9]
N, b, z, m, eta = load_cell(name, k)
sse_xxz.run(b, N, 1.0, ntherm=2, nmeas=2)                     # compile outside the timing
t0 = time.time(); r = sse_xxz.run(b, N, beta, D, h, ntherm=nt, nmeas=nm, seed_=sd, adapt=ad)
commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True, text=True).stdout.strip()
r.update(name=name, k=k, N=N, m_lieb=m, Delta=D, beta=beta, h=h, ntherm=nt, nmeas=nm, seed=sd, adapt=ad, seconds=round(time.time() - t0), commit=commit)
open(out, 'a').write(json.dumps(r) + '\n'); print(json.dumps(r))
