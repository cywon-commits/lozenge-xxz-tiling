"""A1 job list: easy-plane m(h), h || z.  One line per run in the format of run_jobs.sh / s4_qmc.py.
usage: python s4_a1_jobs.py <Delta list, comma> <size: small|large> <beta> <ntherm> <nmeas> <seeds, comma> <out.jsonl> > jobs.txt
Sizes (N close to 500 / 1000 with a size ratio near 2):"""
import sys
SIZES = {'square': (8, 12), '1/11': (4, 6), '1/5': (4, 6), '1/3': (4, 6), '1/2': (4, 6), '3/5': (5, 7), 'dice': (8, 11)}
# N:      square 576/1296, 1/11 528/1188, 1/5 480/1080, 1/3 576/1296, 1/2 384/864, 3/5 500/980, dice 576/1089
Ds = [float(x) for x in sys.argv[1].split(',')]; cls = {'small': 0, 'large': 1}[sys.argv[2]]
beta, nt, nm = float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]); seeds = [int(x) for x in sys.argv[6].split(',')]; out = sys.argv[7]
hs = [round(0.02 * i, 2) for i in range(1, 21)]
jobs = []
for D in Ds:
    for name, ks in SIZES.items():
        for h in hs:
            for s in seeds:
                jobs.append((ks[cls] ** 2, f'{name} {ks[cls]} {D} {beta:g} {h} {nt} {nm} {s} {out}'))
for _, j in sorted(jobs, key=lambda x: -x[0]):            # largest first for better load balance
    print(j)
