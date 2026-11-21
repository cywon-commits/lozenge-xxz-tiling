"""Stage 4: XXZ SSE on a Table-I crystal (k x k copies of its supercell). Appends one JSON line to the output file.
usage: python s4_qmc.py <name> <k> <Delta> <beta> <h> <ntherm> <nmeas> <seed> <out.jsonl>"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import sys, json, time, subprocess
from structs import load_cell
import sse_xxz
name, k, D, beta, h, nt, nm, sd, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]), sys.argv[9]
try: COMMIT = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=_os.path.dirname(_os.path.abspath(__file__)), capture_output=True, text=True).stdout.strip()
except Exception: COMMIT = 'unknown'
N, b, z, m, eta = load_cell(name, k); t0 = time.time()
r = sse_xxz.run(b, N, beta, D, h, ntherm=nt, nmeas=nm, seed_=sd)
r.update(name=name, k=k, N=N, m_lieb=m, Delta=D, beta=beta, h=h, ntherm=nt, nmeas=nm, seed=sd, seconds=round(time.time() - t0), commit=COMMIT)
open(out, 'a').write(json.dumps(r) + '\n'); print(json.dumps(r))
