# XXZ antiferromagnet on lozenge tilings: anisotropy and field direction

Follow-up to *Quantum order by disorder among lozenge tilings* (paper I, https://github.com/cywon-commits/lozenge-quantum-tiling).

Model: `H = J sum_<ij> [SxSx + SySy + Delta SzSz] - h.S`, S = 1/2, on the bipartite graphs defined by lozenge tilings.
Question: how the magnetization staircase of paper I changes with the exchange anisotropy `Delta` and the field direction.

Research plan and results (Korean, Claude Docs): https://claude.ai/artifact/87xmDi4KURMsj4ujGF6L9r
Status: stages 1-3 done (symmetry, analytic formulation, small calculations); stage 4 (QMC) in progress. Unpublished work.

## Layout

```
src/        common.py (supercell tilings, fixed-S^z ED), structs.py (graphs of the paper-I crystals), sse_xxz.py (XXZ SSE)
scripts/    s3_*.py stage-3 checks, s4_*.py stage-4 QMC and analysis, run_jobs.sh
tests/      test_sse_xxz.py (SSE vs exact finite-temperature averages)
data/       stage3/ and stage4/: outputs (JSON, JSONL) and job lists
docs/       work_order_stage4.md (code work order for the larger runs), STATUS_KO.md
```

## Requirements

Python >= 3.10, numpy, scipy, numba. A clone of the paper-I repository; set `LOZENGE_REPO` to its path
(default `~/cywon-commits/lozenge-quantum-tiling`). Its `code/` modules (`periodic.py`, `lswt.py`, `cellgraph.py`) and
`data/` files (crystal cells, QMC decomposition) are read, never modified.

## Reproduce

```bash
export LOZENGE_REPO=/path/to/lozenge-quantum-tiling
python tests/test_sse_xxz.py                     # unit test, about 2 minutes

cd data/stage3                                    # stage 3: each script writes its JSON here
python ../../scripts/s3_ed_tower.py               # Wigner-Eckart structure of the Lieb multiplet
python ../../scripts/s3_K.py                      # K = e_perp - e_par from the QMC decomposition of paper I; step slopes
python ../../scripts/s3_ed_delta.py               # step fields vs Delta on N = 18, 20 clusters
python ../../scripts/s3_lswt.py                   # XXZ spin-wave hull, both branches
python ../../scripts/s3_chi_classical.py          # classical transverse-susceptibility identity
python ../../scripts/s3_ed_hx.py                  # field along x: quasi-plateau
python ../../scripts/s3_model.py                  # effective director model (needs K_table.json, chi_classical.json)

cd ../stage4                                      # stage 4: QMC
../../scripts/run_jobs.sh jobs_b1.txt 2           # Delta = 1.02, 1.05, 1.10, field || z; size scaling of the 1/11 crystal
../../scripts/run_jobs.sh jobs_b2.txt 2           # Delta = 0.97, field || z: m(h) curves
python ../../scripts/s4_analyse.py                # tables from the JSONL files
```

## Main results so far

- Symmetry: exact plateaus only for h || z (or Delta = 1); for h || x a Z2 (Ising) transition inside a fixed tiling is possible.
- To first order in x = Delta - 1 everything is fixed by the anisotropy stiffness K = e_perp - e_par of the Heisenberg ground state
  (Wigner-Eckart): multiplet splitting E(Sz) - E(M) = x N K (1 - Sz^2/M^2); step slopes
  s_par = h/3 - (2/3) dK/dm (Delta > 1, h || z) and s_perp = h/3 + (1/3) dK/dm (Delta < 1, in-plane h).
- K from the QMC decomposition of paper I: 0.107 J (1/11 crystal) to 0.2345 J (dice), K ~ 0.108 + 0.77 m.
- All steps move to lower fields for both signs of the anisotropy; the width of the 3/5 plateau is unchanged to first order.
- Field along the hard direction: plateaus are cut to a window [h-, h+], h- ~ 2|x|K/m; low-moment crystals disappear first.
- Spin-wave hull: the 3/5 hub-free crystal remains the deepest vertex for Delta = 0 ... 4.

Acknowledgment of support and AI assistance as in paper I. License: MIT (to be confirmed before the repository is made public).
