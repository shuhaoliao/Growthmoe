# Growth MoE RL Prototype

Minimal research prototype for testing growth-style MoE reinforcement learning on a lightweight continuous-control navigation task.

This repo studies four questions:

- Q1. Does growth-style action restriction improve early training stability?
- Q2. Does MoE specialize across different terrain dynamics?
- Q3. After maturation and partial freezing, does routing become lower-entropy and behavior more stable on old tasks?
- Q4. Is a matured policy harder to adapt on new tasks than a more plastic policy?

## What Is In This Repo

Core files:

- `envs/multi_region_nav_env.py`: multi-goal 2D navigation environment
- `models/moe_policy.py`: MLP actor and MoE actor
- `models/critic.py`: critic network
- `algos/ppo.py`: lightweight PPO trainer
- `evaluate.py`: evaluation, terrain-activation stats, GIF export
- `run_experiments.py`: batch run the 4 core experiments with a timestamped output folder
- `summarize_results.py`: build reward comparison plots, GIFs, and JSON summaries

## Task Design

### Old Task Distribution

`old` is an unordered multi-goal coverage task on maps with:

- `normal`
- `slippery`
- `damping`

Each episode places one goal in each primary terrain. The policy succeeds only after visiting all terrain-specific targets, but the order does not matter. The reward therefore favors:

- covering all goals
- short paths
- low control effort

rather than following a fixed waypoint sequence.

### New Task Distribution

`new` is the same unordered coverage problem, but with one additional target in:

- `disturbance`
- `slippery + disturbance`

This makes adaptation harder because the policy must still cover all goals while handling persistent external perturbations.

### What MoE Activation Means Here

The environment records terrain IDs at every step. During training and evaluation, the MoE policy logs:

- overall expert usage
- expert usage by terrain type
- gate entropy
- goal coverage ratio

This lets you inspect whether different experts are more active on `normal`, `slippery`, `damping`, or disturbance-heavy regions.

## Training Stages

- `Stage A / acquisition`: learn multi-goal navigation on `old`
- `Stage B / maturation`: continue on `old`, reduce routing temperature, shrink top-k, freeze part of the model
- `Stage C / relearning`: switch to `new` and compare plastic vs mature adaptation

## Upload This Repo To GitHub

Use the following commands after creating an empty GitHub repo:

```bash
git add .
git commit -m "Add growth-style MoE RL prototype"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

Example remote URL forms:

```bash
git remote add origin git@github.com:<user>/<repo>.git
git remote add origin https://github.com/<user>/<repo>.git
```

## Ubuntu Server Setup

The commands below assume:

- Ubuntu with NVIDIA driver already installed
- Python 3.12 available through conda or system Python
- you want GPU training

### Option A: Conda

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd growthmoe

conda create -n growthmoe python=3.12 -y
conda activate growthmoe

pip install --upgrade pip
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Verify CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

### Option B: Python venv

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd growthmoe

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

If the server is CPU-only, replace the torch install with:

```bash
pip install torch
```

## Run The 4 Experiments

### Recommended final run: batch run with timestamped output

This script automatically creates a folder like `runs/20260318_212357_compare4_seed42/`.

```bash
python run_experiments.py --preset full --device cuda --tag compare4
```

This runs:

- `baseline`
- `gpo_only`
- `moe_only`
- `full`

and then automatically generates summary plots, GIFs, and JSON outputs.

For a shorter smoke or trend run, use:

```bash
python run_experiments.py --preset quick --device cuda --tag compare4_quick
```

### Run each experiment manually

```bash
python train.py --exp baseline --preset full --stage acquisition --device cuda
python train.py --exp gpo_only --preset full --stage acquisition --device cuda
python train.py --exp moe_only --preset full --stage acquisition --device cuda
python train.py --exp full --preset full --stage all --device cuda
```

## Summarize And Visualize Results

If you used `run_experiments.py`, summary is generated automatically. If you want to regenerate it:

```bash
python summarize_results.py --group-dir runs/<TIMESTAMP>_compare4_seed42 --device cuda
```

This produces:

- one reward comparison figure for the 4 methods
- one old-task GIF for each method
- `full` plastic vs mature GIFs on `new`
- JSON summary with evaluation metrics, coverage, and output paths

## Where Final Visualizations Appear

Inside:

```text
runs/<TIMESTAMP>_compare4_seed42/summary/
```

Key outputs:

- `reward_comparison_acquisition.png`
- `coverage_curve.png` inside each run's `plots/`
- `baseline_old.gif`
- `gpo_only_old.gif`
- `moe_only_old.gif`
- `full_old.gif`
- `full_plastic_new.gif`
- `full_mature_new.gif`
- `comparison_summary.json`

## Export Extra GIFs Manually

Example:

```bash
python evaluate.py \
  --checkpoint runs/<RUN>/full/maturation/stage_b_mature.pt \
  --mode old \
  --device cuda \
  --gif-path runs/<RUN>/media/full_old.gif \
  --map-path runs/<RUN>/media/full_old_map.png
```

## Notes About GPO-Style Action Growth

The action-growth part follows the paper-style form:

```text
a = clip(a, -a_limit, a_limit)
a_tilde = beta_t * tanh(a / beta_t)
beta_t = a_limit * exp(-exp(-k * t))
```

The code extends the original GPO idea in three directions:

- action growth
- routing-capacity growth
- plasticity/maturation growth

## Current Goal Semantics

The current environment no longer uses ordered waypoint tracking.

- `old`: visit one goal in each of `normal`, `slippery`, and `damping`
- `new`: visit those three goals plus one disturbance-heavy goal
- success: all goals visited
- order: irrelevant
- reward: goal coverage bonus + nearest-unvisited progress shaping - path/control/time costs

## Current Default Budget

Quick preset:

- acquisition: `80k`
- maturation: `20k`
- relearning: `40k`

Full preset:

- acquisition: `200k`
- maturation: `50k`
- relearning: `100k`

## Useful Commands

Check a trained run:

```bash
python evaluate.py --checkpoint runs/<RUN>/moe_only/acquisition/stage_a_end.pt --mode old --device cuda
```

Run only the MoE baseline:

```bash
python train.py --exp moe_only --preset quick --stage acquisition --device cuda
```

## Ignore Rules

The repo ignores local artifacts such as:

- `runs/`
- `paper/`
- `__pycache__/`
- `.vscode/`

so GitHub uploads contain code and config, not local experiment outputs.
