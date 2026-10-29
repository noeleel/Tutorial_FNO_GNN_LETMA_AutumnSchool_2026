# Instructor guide — LETMA 2026 ML labs (3 × 30 minutes)

## Format

Three labs, **30 minutes each**, one shared skeleton: state every hypothesis (numbered H1, H2, …)
→ check on data what can be checked → build/train → watch animations → evaluate honestly →
transfer checklist. Student versions contain the exercises with `TODO`s; solution versions are
executed end-to-end with every figure and animation embedded, so **you can teach entirely from
the solutions** and let students run the student versions.

## The 30-minute contract (why nobody gets left behind)

- The notebooks no longer carry timing checkpoints: pace the room yourself against the
  section structure (the whole set still runs in about 30 minutes per lab).
- **Exercises cannot block.** Each exercise has an automatic checker; on failure the reference
  implementation takes over silently. Give each exercise its small budget (about 5 minutes) and
  move on: "not passing? read the hint tonight."
- **Optional deep dives** (Lab 1 §4.5 3-D landscape and §4.6 hyperparameter corner plot;
  Lab 2 latent-field visualizations and §4.3 ML-pathology demos) are fully executed in the
  solutions. If the room is fast, demo them live (the seed-spread demo takes seconds on a GPU);
  otherwise point to the solutions.
- The main line carries both a plain-language explanation and the rigorous statement of each
  step side by side; the mathematical notes are optional reference material for after class.

## Suggested running order

1 → 2 → 3. Lab 1 introduces the loss, the landscape and the optimizer race; Lab 2 reuses both on
a physics-rich operator and dissects the latent space; Lab 3 changes the geometry (irregular
graphs) and teaches oversmoothing. Compute is never the pacing bottleneck (~1 min per lab on a
GPU, ~2–8 min on a laptop CPU).

## Practical notes

- **Data**: everything is generated on the fly (Burgers pseudo-spectral solver, infrasound
  parabolic-equation marcher, synthetic seismic picks). No downloads, no stored datasets.
- **Devices**: the setup cell auto-detects CUDA and falls back to CPU. Both were validated
  (identical results to ~3 decimals).
- **Hedy cluster**: use the reservation/MIG pattern from the school planning docs; the two
  scripts in `slurm/` re-execute all labs (PyTorch) and both JAX extensions headlessly:
  adjust `--account`, `--partition`, `--reservation`, `--nodelist`, and the venv paths
  (`~/envs/fno-gnn`, `~/envs/jax-hedy`). Batch jobs must export
  `PYTHONPATH=$HOME/<repo-root>` for the JAX notebooks (they import the in-repo profiler).
- **JAX extensions** (full mirrors of Labs 1 and 3 in JAX/Equinox/Optax — vmap/jit/pytrees,
  pytree loss landscapes, padded+masked graph batching; data shipped in
  `data/burgers_jax.npz` and `data/seismic_jax.npz`, so torch is not needed) are optional
  material for after class; install in a separate venv
  (`pip install "jax[cuda12]" equinox optax matplotlib ipykernel nbconvert`) — never during
  the session.

## Timing evidence

The executed solutions carry timing prints; on a MIG 1g.20gb GPU instance of an H100 the full
set (3 labs + Darcy scale-up + the 4 JAX notebooks — solutions and students) runs in ~25 minutes
of compute (labs 104+159+49 s, Darcy ~9 min, JAX FNO 128 s / GNN 93 s, students ~same); the
30-minute budget is reading-and-watching time, and the pacing tables account for that.

## Scale-up extension (optional, after class)

`notebooks/extensions/scaleup_darcy_fno2d.ipynb`: the small FFNO2d (width 20, modes 14²,
5 blocks, 123k params) on the shipped 2304-pair 96×96 Darcy database (`data/darcy96_big.pt`,
fp16, 85 MB), trained **200 epochs**: the long-schedule regime. Validated on the cluster:
**3 min on a full H100** (~10 min on the MIG 1g.20gb slice), **test rel-L2 0.0083** — better
than the literature's ~0.01, from a model smaller than the lab network. The notebook also
dissects the learned Fourier weights per layer (x/y responses match within 3%: the isotropy
check). Hours on CPU — set `DARCY_EPOCHS=10`–`20` there or read the executed solution.
Regenerating the database needs `scipy` (~2 min; not required).


## Participants' environment (Hedy)

Homes are private on Hedy: participants build their OWN environment with
`bash scripts/setup_participant.sh` (pinned requirements, kernel registered, ~6 min; validated
end-to-end on the cluster on 28 Oct 2026: fresh venv, torch 2.15+cu132 + PyG 2.8.1, imports OK).
If many participants set up simultaneously, expect PyPI bandwidth contention — staggering the
downloads, or asking the mesocentre for a shared account directory (pip cache / a shared
read-only venv / an Apptainer `.sif`), removes it.
