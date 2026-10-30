# LETMA 2026 — Machine Learning on Geophysical Data: three hands-on labs

<img src="assets/logos/amiad.png" width="120"/> <img src="assets/logos/cea.png" width="120"/> <img src="assets/logos/lmps.png" width="120"/> <img src="assets/logos/borelli.png" width="120"/>

**Authors:** E. Noëlé (1, 2, 3), C. Millet (2, 4), F. Gatti (3), D. Clouteau (3)

**Affiliations:**
1. AMIAD, Pôle Recherche, France
2. CEA DAM DIF
3. Université Paris-Saclay, CentraleSupélec, ENS Paris-Saclay, CNRS, LMPS — Laboratoire de Mécanique Paris-Saclay, 91190 Gif-sur-Yvette, France
4. Centre Borelli, ENS Paris-Saclay, Université Paris-Saclay
   
**Contact:** elodie.noele@polytechnique.edu

Feel free to contact me if you have any questions. I'll try my 

**Fourier Neural Operators (FNO) · Factorized FNO for infrasound (FFNO) · Graph Neural Networks (GNN) · a Darcy-flow scale-up**

Three self-contained lab notebooks: every assumption is
stated as a numbered hypothesis (H1, H2, …) before it is used, every exercise is fail-safe
(automatic checker + reference fallback), and every lab is driven by animations — from PDE
dynamics and **boundary-layer infrasound propagation** (wavefields, rays, hodographs) to
**animated loss landscapes with multiple minima, optimizer races to the final loss, and the
Dirichlet-energy collapse of deep GNNs**.

You can either run this tutorial on Hedy (only during the Autumn School) Binder [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/noeleel/Tutorial_FNO_GNN_LETMA_AutumnSchool_2026.git/HEAD), on Colab (see notebooks, below, on the right of the board) or on your laptop (See **Quick start** section)

| Lab | Question | Method | Data | Exercise | Colab Notebooks |
|---|---|---|---|---|:---:|
| **1 · FNO for Burgers** | Can a network learn a PDE's solution operator? | Fourier Neural Operator (Li et al., ICLR 2021) | 1-D viscous Burgers, generated on the fly | implement `SpectralConv1d` | [![Student](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/noeleel/Tutorial_FNO_GNN_LETMA_AutumnSchool_2026/blob/master/notebooks/labs/lab_fno_student.ipynb) <br> [![Solution](https://img.shields.io/badge/Colab-Solution-success?logo=googlecolab)](https://colab.research.google.com/github/noeleel/Tutorial_FNO_GNN_LETMA_AutumnSchool_2026/blob/master/notebooks/labs/lab_fno_solution.ipynb) |
| **2 · FFNO for infrasound** | Can a network predict ground waveforms from the boundary-layer profile that shaped them? | Factorized FNO (Tran et al., ICLR 2023) | narrow-angle parabolic-equation solver, synthetic atmospheres | implement the trace spectrum | [![Student](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/noeleel/Tutorial_FNO_GNN_LETMA_AutumnSchool_2026/blob/master/notebooks/labs/lab_ffno_infrasound_student.ipynb) <br> [![Solution](https://img.shields.io/badge/Colab-Solution-success?logo=googlecolab)](https://colab.research.google.com/github/noeleel/Tutorial_FNO_GNN_LETMA_AutumnSchool_2026/blob/master/notebooks/labs/lab_ffno_infrasound_solution.ipynb) |
| **3 · GNN for a sensor network** | How do we learn on stations scattered on a map — any number of them, some always down? | Graph Neural Network (PyTorch Geometric) | synthetic seismic network with station dropout | implement the message-passing layer | [![Student](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/noeleel/Tutorial_FNO_GNN_LETMA_AutumnSchool_2026/blob/master/notebooks/labs/lab_gnn_student.ipynb) <br> [![Solution](https://img.shields.io/badge/Colab-Solution-success?logo=googlecolab)](https://colab.research.google.com/github/noeleel/Tutorial_FNO_GNN_LETMA_AutumnSchool_2026/blob/master/notebooks/labs/lab_gnn_solution.ipynb) |
| **Extension · Darcy scale-up** | The same method, long-schedule regime | **small FFNO2d** (123k params) × **200 epochs** | **2304 pre-generated 96×96 Darcy pairs** (shipped in `data/`) | — | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/noeleel/Tutorial_FNO_GNN_LETMA_AutumnSchool_2026/blob/master/notebooks/extensions/scaleup_darcy_fno2d.ipynb) |
| **Extension · Lab 1 in JAX** | The same FNO lab, in the other framework | JAX + Equinox + Optax (vmap, jit, pytree landscapes) | Lab 1's Burgers samples (shipped as `data/burgers_jax.npz`) | `SpectralConv1d` + the forward pass | [![Student](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/noeleel/Tutorial_FNO_GNN_LETMA_AutumnSchool_2026/blob/master/notebooks/extensions/jax_fno_student.ipynb) <br> [![Solution](https://img.shields.io/badge/Colab-Solution-success?logo=googlecolab)](https://colab.research.google.com/github/noeleel/Tutorial_FNO_GNN_LETMA_AutumnSchool_2026/blob/master/notebooks/extensions/jax_fno_solution.ipynb) |
| **Extension · Lab 3 in JAX** | The same GNN lab, without PyTorch Geometric | JAX + Equinox + Optax (padded + masked batching) | the same seismic events (shipped as `data/seismic_jax.npz`) | masked mean-aggregation + the full locator | [![Student](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/noeleel/Tutorial_FNO_GNN_LETMA_AutumnSchool_2026/blob/master/notebooks/extensions/jax_gnn_student.ipynb) <br> [![Solution](https://img.shields.io/badge/Colab-Solution-success?logo=googlecolab)](https://colab.research.google.com/github/noeleel/Tutorial_FNO_GNN_LETMA_AutumnSchool_2026/blob/master/notebooks/extensions/jax_gnn_solution.ipynb) |

Each lab exists in two versions: `*_student.ipynb` (exercises with `TODO`s) and
`*_solution.ipynb` (executed end-to-end, all figures and animations embedded — open it to see
everything without running a single cell). Each notebook has a core path and parts marked
*optional* that go deeper (derivations, extra experiments); skipping them breaks nothing that
follows. Every lab ends with a **Going further** section of references.

**Handouts.** [`docs/LETMA2026_labs_handout.pdf`](docs/LETMA2026_labs_handout.pdf) — a printable
summary of what will be done in each lab, with the scientific question and the exercises.
[`docs/LETMA2026_solutions_handoff.pdf`](docs/LETMA2026_solutions_handoff.pdf) — the exercise
solutions and the numbers every run reproduces (LaTeX sources alongside).

---

## What you will actually see (illustrations)

**Lab 1 — the operator being learned, and its loss landscape with several minima.**
Left: the statistical distributions of the inputs and outputs (per-sample norms vary ×6 — the
imbalance the loss must handle). Right: the same problem trained twice from different seeds
gives two different minima; three Adam initializations each fall into their own basin.

| | |
|---|---|
| ![Distributions](docs/images/lab1_distributions.png) | ![Multiple minima](docs/images/lab1_multiple_minima.png) |

The single-minimum view of the same landscape, with four optimizers racing **to the final trained
loss**: Adam arrives in 79 steps, momentum in 604, plain fixed-step SGD never leaves the plateau.
A hyperparameter **corner plot** (modes, width, depth, projection width) and a **3-D view**
complete the section. The payoff of Fourier layers: parameters live on **modes, not grid
points**, so the model trained at $N=64$ evaluates zero-shot at $N=256$ with the same error.

| | |
|---|---|
| ![Optimizer race](docs/images/lab1_optimizer_race.png) | ![Corner plot](docs/images/lab1_corner_plot.png) |

**Lab 2 — one operator, profile in → waveforms out, from three complementary viewpoints.**
Top: the input profile (vertical: altitude on $y$, celerity on $x$) and the ray fan it produces.
Bottom: the **hodograph** (each recorded trace at its own absolute arrival times, with the
ray-predicted arrivals) and the value distributions of inputs and outputs.

| | |
|---|---|
| ![Vertical profiles](docs/images/lab2_vertical_profiles.png) | ![Ray tracing](docs/images/lab2_raytracing.png) |
| ![Hodograph](docs/images/lab2_hodograph.png) | ![Distributions](docs/images/lab2_distributions.png) |

The FFNO predicts **all four ground receivers jointly, in one evaluation** — the spectrum of a
recorded trace (Exercise F2) confirms the band the operator must reproduce, and every held-out
prediction comes with its input profile.

| | |
|---|---|
| ![Trace spectrum](docs/images/lab2_spectra.png) | ![Held-out waveforms](docs/images/lab2_test_waveforms.png) |

**Lab 3 — a graph built by k-NN, and its specific failure mode, quantified.**
Left: the input distributions (per-event station counts change the graph size every event — H3).
Right: oversmoothing made quantitative — node colours homogenize *while the Dirichlet energy
collapses*, with the energy curve animating alongside the map.

| | |
|---|---|
| ![Lab 3 distributions](docs/images/lab3_distributions.png) | ![Smoothing-operator spectrum](docs/images/lab3_spectrum.png) |

The notebook itself animates, step by step, **how k-nearest-neighbours constructs the graph node
by node** (candidates in grey, the 6 selected edges in red, committed edges in blue).

**Extension — a small factorized network, trained long, on a big database.** The FFNO2d of
Lab 2 — deliberately **small** (width 20, modes 14², 5 blocks, 123k parameters, ×6 fewer
spectral parameters per layer than a full 2-D FNO) — trained for **200 epochs** on **2304
pre-generated 96×96 Darcy pairs** (finite-volume ground truth, shipped as a half-precision
85 MB file): the long-schedule regime. Input/output distributions, the spectral check (H5), a
prediction-improving animation, the **learned Fourier weights dissected layer by layer**
(the x/y responses match at ratio 1.03, exactly as the isotropy of the law predicts), and a
worst-case gallery. The executed solution trained the full budget **in 3 minutes on an H100
(~10 minutes on the school's MIG 1g.20gb slice)** and reaches test rel-$L^2$ **0.0083** —
comparable in magnitude to published FNO Darcy errors, on a different dataset (the repository's
own 96×96 finite-volume generator, not the Li et al. benchmark), from a model smaller than the
Burgers lab network. On a laptop CPU set `DARCY_EPOCHS=10`–`20`, or read
the executed solution.

| | | |
|---|---|---|
| ![Darcy distributions](docs/images/scaleup_darcy_distributions.png) | ![Fourier weights](docs/images/scaleup_fourier_weights.png) | ![Darcy predictions](docs/images/scaleup_darcy_predictions.png) |

More animations (message passing hop by hop, training converging epoch by epoch, code-space
interpolation) live **inside the executed solution notebooks**; a few legacy GIFs are in
[`docs/images/animations/`](docs/images/animations).

---

## Quick start

```bash
git clone https://github.com/noeleel/Tutorial_FNO_GNN_LETMA_AutumnSchool_2026.git letma2026-ml-labs
cd letma2026-ml-labs
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # CPU is enough; GPU optional
jupyter lab                               # or: jupyter notebook
```

Open `notebooks/labs/lab_fno_student.ipynb` and run top to bottom. The three labs generate all
their data on the fly (Burgers pseudo-spectral solver, infrasound parabolic-equation marcher,
synthetic seismic picks) — no downloads. The Darcy scale-up loads its 2304-pair 96×96 database from the shipped
`data/darcy96_big.pt` (half precision, 85 MB; regenerating it requires `scipy` and ~2 minutes).
The two JAX extensions read `data/burgers_jax.npz` (153 kB: Lab 1's Burgers samples, same seeds,
at N=64 and N=256) and `data/seismic_jax.npz` (0.4 MB: Lab 3's seismic events as padded arrays) —
so they need **neither torch nor PyG**, only JAX (regenerate the files with
`python scripts/make_burgers_npz.py` and `python scripts/make_seismic_npz.py` in the PyTorch
environment).

**Binder and Colab.** Binder builds [`binder/environment.yml`](binder/environment.yml): the
torch stack of `requirements.txt` on CPU (PyTorch, PyTorch Geometric, NumPy, Matplotlib). The
JAX extensions are optional and are not installed there (JAX is in the commented-out section of
`requirements.txt`; install it in a separate environment). On Colab, every notebook's first
code cell detects the Colab runtime, clones this repository into `/content` once, installs the
only packages Colab lacks (`torch_geometric` for Lab 3, `equinox`/`optax` for the JAX
extensions) and points the notebook at the clone; nothing of this runs anywhere else. Labs 1
and 3 run on Colab's free CPU in a few minutes each; Lab 2's training and the **Darcy scale-up
are slow on the free CPU** — select a GPU runtime (*Runtime → Change runtime type*), or for the
scale-up set `DARCY_EPOCHS=10`–`20`.
Animations play as JavaScript widgets on a laptop, on Binder and on Colab; only on Hedy, where
Euporie cannot play them, they are shown as a static six-frame filmstrip.

## On the Hedy cluster (school sessions)

The labs are pre-staged in the account's shared folder. One hop, X11 forwarding on:

```bash
ssh -Y hedy                     # the single hop (-Y: X11 forwarding for live plots)

# GPU session on a MIG slice: ONE command allocates the slice, forwards X11 and drops
# you into an interactive shell ON the GPU node:
salloc \
    --account=m26266 \
    --partition=gpu \
    --reservation=test_mig_letma \
    --nodelist=m-gpu[07-08] \
    --gres=gpu:nvidia_h100_80gb_hbm3_1g.20gb:1 \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --time=01:00:00 \
    bash -c '
        NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
        echo "Allocated job: $SLURM_JOB_ID on $NODE"
        exec ssh -Y -C -t "$NODE" \
            "srun --jobid=$SLURM_JOB_ID --overlap --ntasks=1 --pty bash -i"
    '
# release the allocation afterwards with: scancel <jobid>   (see squeue -u $USER)

# then, ON the GPU node: first CHECK YOUR SLICE - you should see one MIG device
# with ~20 GB of memory:
nvidia-smi
#   MIG devices:
#   GPU  GI  CI  MIG |              Shared Memory-Usage
#     0   3   0   0  |               9MiB / 20096MiB     <- your 1g.20gb slice
# (if the MIG section is absent or shows another size, exit and resubmit)

source /home/projects/m26266/letma2026-ml-labs/env/fno-gnn/bin/activate   # or your own venv
rsync -a --exclude env /home/projects/m26266/letma2026-ml-labs/ ~/letma2026-ml-labs/  # once
cd ~/letma2026-ml-labs/notebooks/labs
# Hedy has no SSH port forwarding, so there is no browser interface: use euporie,
# the notebook terminal interface (on your own machine, `jupyter lab` works normally)
euporie-notebook --kernel-name python3 lab_fno_student.ipynb
# optional, with the -Y forwarding: pop-up matplotlib windows instead of inline plots
# (tk needs a WORKING X display: the labs test it and fall back to inline plots automatically)
LAB_PLOT_MODE=tk euporie-notebook --kernel-name python3 lab_fno_student.ipynb
```

**Launch time: ~5 s on a warmed node, 1–2 min on a cold one.** Everything (euporie, jupyter,
torch, numpy, matplotlib, …) is paged in from the cluster's shared filesystem on the first
launch of a node, and the screen just shows `Loading…` / `No Kernel` meanwhile — this is not
a freeze. The nodes are warmed up once before the session (the Linux page cache is shared
by all users of a node):

```bash
ssh hedy                # no -Y needed here: this submits a batch job, nothing is displayed
sbatch /home/projects/m26266/letma2026-ml-labs/slurm/prewarm_nodes.sbatch   # ~3 min, both nodes
```

After that, wait only until the kernel badge `Python 3 (ipykernel)` appears in the
bottom-right status bar — the notebook is then fully interactive. The `Rendering… (61%)`
indicator may stay at a partial value: it counts the whole notebook, and cells below the
visible area are rendered lazily when you scroll.

**On Hedy, the animations are automatically replaced by static filmstrips.** Euporie
cannot play the JavaScript animation widget, so each animation cell draws a strip of six
key frames instead — same content, no motion. This is detected at run time (the Hedy
session profiler is available only on the cluster) and happens only there; on your own
machine, in Jupyter or JupyterLab, the animations play normally.

**Euporie quick reference** (verified against the installed version — navigate with `↑/↓`
or `j/k`, `Home/End`, `PgUp/PgDn`; `Tab` cycles the focus between notebook, side pane and the
command bar at the bottom):

| action | keys |
|---|---|
| run the selected cell / run and go to next | `Ctrl+Enter` / `Shift+Enter` |
| run and insert a cell below | `Alt+Enter` |
| enter edit mode / leave edit mode | `Enter` / `Esc` |
| insert cell above / below; delete (`d d`); undelete (`z`) | `a` / `b` / `d d` / `z` |
| cut, copy, paste cells | `x` / `c` / `v` |
| open the command palette and look up for a command |  `Ctrl+Space` |
| cell to markdown / to code | `m` / `y` |
| move cells up / down | `Alt+↑` / `Alt+↓` |
| merge cells / split cell | `M` / `Ctrl+\` |
| expand or collapse output; wrap output; line numbers | `w` / `W` / `l` |
| interrupt kernel / restart kernel | `i i` / `0 0` |
| save / save as / open / new notebook / close tab | `Ctrl+S` / `Alt+S` / `Ctrl+O` / `Ctrl+N` / `Ctrl+W` |
| quit (prompts to save unsaved changes) | `Ctrl+Q` |

Less common actions can be typed in the command bar (Tab to focus it). Figures render inline —
terminal graphics on Kitty/Sixel-capable terminals, character previews elsewhere. The
`python3` kernel is the activated venv's interpreter. The JAX extensions use the sibling venv
`env/jax-hedy` the same way. `slurm/PROCEDURE_TESTS.md` documents the headless re-validation
(`sbatch slurm/test_all_mig.sbatch`).

## Requirements and measured hardware footprint

**Software**: Python ≥ 3.10 (tested on 3.12/3.13); see [`requirements.txt`](requirements.txt).
The three labs need only PyTorch + PyTorch Geometric + NumPy/Matplotlib/Jupyter. Optional:
`scipy` (to regenerate the Darcy database), `jax[cuda12]`/`equinox`/`optax` (the JAX
extensions, separate venv advised).

**Hardware** (measured on the executed solutions):

| Resource | CPU-only (laptop) | GPU (tested on a MIG slice of an H100) |
|---|---|---|
| Processor | any ≥ 2 cores (4+ comfortable) | any CUDA 12 GPU |
| GPU memory | — | **≥ 4 GB** — everything runs on a 19.6 GB `MIG 1g.20gb` instance |
| System RAM | **~1 GB free** (labs), **~3 GB** (scale-up: 1200×2×64² float32) | ~1–3 GB |
| Disk | ~150 MB repo (incl. the 85 MB shipped Darcy database) | same |
| Lab 1 / 2 / 3 runtime | ≈ 8 / 10 / 2 min | **104 s / 159 s / 49 s** |
| Darcy scale-up (200 epochs) | hours — set `DARCY_EPOCHS=10`–`20` or read the solution | **3.0 min** (full H100) / ~10 min (`MIG 1g.20gb`) |
| JAX extensions (full FNO + GNN mirrors) | ≈ 8 / 6 min on CPU | **128 s + 93 s** (solution notebooks; students ≈ same) |

Nothing requires internet access at run time (except the one-time clone on Colab). Results are deterministic (fixed seeds); GPU and
CPU runs agree to ~3 decimals (e.g. Darcy test rel-L2 0.0405 CPU vs 0.0404 GPU).

## Repository layout

```text
letma2026-ml-labs/
├── README.md                     ← this file
├── scripts/
│   ├── setup_participant.sh      ← one-time participant setup (Hedy or laptop)
│   ├── make_burgers_npz.py        ← regenerates data/burgers_jax.npz  (torch env, run once)
│   └── make_seismic_npz.py        ← regenerates data/seismic_jax.npz  (torch env, run once)
├── binder/environment.yml        ← Binder build (torch stack, CPU; JAX not included)
├── assets/logos/                 ← partner logos shown in the notebook headers
├── requirements.txt               ← install file (+ optional extras commented)
├── data/
│   ├── darcy96_big.pt            ← pre-generated 96×96 Darcy database (2304 pairs, 85 MB, fp16)
│   ├── burgers_jax.npz           ← Lab 1's Burgers samples as plain arrays (JAX extension, 153 kB)
│   └── seismic_jax.npz           ← Lab 3's seismic events as plain arrays (JAX extension, 0.4 MB)
├── fno_tuto/                      ← Burgers/Darcy data generators, losses, FNO/FFNO models
│   └── infrasound.py              ← parabolic-equation solver + synthetic atmospheres
├── gnn_tuto/                      ← seismic simulator, graph construction, GNN models, checkers
├── hedy_euporie/                  ← tiny optional session-profiler used by the labs
├── notebooks/
│   ├── labs/                      ← the three labs (student + executed solution)
│   └── extensions/                ← Darcy scale-up + the two full JAX reimplementations
├── docs/                         ← handouts (PDF + LaTeX), README illustrations
└── slurm/                         ← single-hop validation scripts (see PROCEDURE_TESTS.md)
```

## Going further

- **Lab 1 (FNO):** Z. Li et al., *Fourier Neural Operator for Parametric Partial Differential
  Equations*, ICLR 2021 ([arXiv:2010.08895](https://arxiv.org/abs/2010.08895)); N. Kovachki et
  al., *Neural Operator: Learning Maps Between Function Spaces with Applications to PDEs*, JMLR
  2023 ([arXiv:2108.08481](https://arxiv.org/abs/2108.08481)); H. Li et al., *Visualizing the
  Loss Landscape of Neural Nets*, NeurIPS 2018 ([arXiv:1712.09913](https://arxiv.org/abs/1712.09913)).
- **Lab 2 (FFNO, infrasound):** A. Tran et al., *Factorized Fourier Neural Operators*, ICLR
  2023 ([arXiv:2111.13802](https://arxiv.org/abs/2111.13802)); M. D. Collins, *A split-step Padé
  solution for the parabolic equation method*, J. Acoust. Soc. Am. 93, 1736–1742 (1993)
  ([doi](https://doi.org/10.1121/1.406739)); D. P. Drob, J. M. Picone, M. Garcés, *Global
  morphology of infrasound propagation*, J. Geophys. Res. 108(D21) (2003)
  ([doi](https://doi.org/10.1029/2002JD003307)).
- **Lab 3 (GNN):** J. Gilmer et al., *Neural Message Passing for Quantum Chemistry*, ICML 2017
  ([arXiv:1704.01212](https://arxiv.org/abs/1704.01212)); K. Oono, T. Suzuki, *Graph Neural
  Networks Exponentially Lose Expressive Power for Node Classification*, ICLR 2020
  ([arXiv:1905.10947](https://arxiv.org/abs/1905.10947)); M. P. A. van den Ende, J.-P. Ampuero,
  *Automated Seismic Source Characterization Using Deep Graph Neural Networks*, GRL 2020.
- **Fourier neural operators for seismic waves:** F. Lehmann, F. Gatti, M. Bertin, D. Clouteau,
  *3D elastic wave propagation with a Factorized Fourier Neural Operator (F-FNO)*, CMAME 420,
  116718 (2024) ([doi](https://doi.org/10.1016/j.cma.2023.116718)); F. Lehmann, F. Gatti,
  D. Clouteau, *Multiple-input Fourier Neural Operator (MIFNO) for source-dependent 3D
  elastodynamics*, JCP 527, 113813 (2025) ([doi](https://doi.org/10.1016/j.jcp.2025.113813));
  F. Lehmann et al., *the HEMEW<sup>S</sup>-3D database*, ESSD 16, 3949–3972 (2024)
  ([doi](https://doi.org/10.5194/essd-16-3949-2024)).
- **A related tutorial:** [Tutorial_GNN_Summer_School_CEA_EDF_INRIA2025](https://github.com/noeleel/Tutorial_GNN_Summer_School_CEA_EDF_INRIA2025) — given at the CEA–EDF–INRIA summer school (June 2025):
  dynamic graph neural networks for seismic characterization, with a dedicated study of
  oversmoothing. A natural next step after Lab 3.

Each notebook ends with a longer, annotated **Going further** list.

## License

MIT — see [`LICENSE`](LICENSE). If you use these labs for teaching, an acknowledgement of the
LETMA 2026 summer school is appreciated.
