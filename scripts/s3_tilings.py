"""List the structural classes of small supercells used for ED."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import pickle, time
from common import *
R = {}
for cell in [(6,2,2),(4,3,1),(6,3,3),(9,2,3),(6,3,0),(10,2,1),(7,3,2),(6,4,2),(12,2,6),(8,3,2)]:
    t0 = time.time(); C, cl = classes(cell)
    R[cell] = cl
    summ = {}
    for (d, hist, _), v in cl.items(): summ.setdefault((d, hist), 0); summ[(d, hist)] += 1
    print(cell, 'N=%d' % C.N, 'classes=%d' % len(cl), '%.0fs' % (time.time() - t0))
    for (d, hist), n in sorted(summ.items()): print('    NB-NA=%d  hist=%s  x%d' % (d, dict(hist), n))
pickle.dump(R, open('tilings_small.pkl', 'wb'))
