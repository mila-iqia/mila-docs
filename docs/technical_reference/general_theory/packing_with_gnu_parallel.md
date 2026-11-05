# Packing ML Jobs with GNU `parallel` + CUDA MPS on the Mila SLURM Cluster

## Jobs vs Experiments

Computer-science research often involves running many computationally-costly
experiments, more often called "runs". Researchers will often have a script, or a
command, that runs the bulk of the code they need for each such run. A common
requirement is to rerun experiments several times with different "seeds" for the
random-number generators to assure oneself that a result is not a fluke.

The Slurm scheduler, however, thinks in terms of _jobs_. A job is a an allocation of
a certain set of resources for a certain amount of time. Chiefly, these resources
are CPU cores, GPUs and memory, specified with `sbatch` and `salloc` flags like
`--cpus-per-task=`, `--gres=gpu:a100l:1` and `--mem=16G`; while the resource of time
is requested with a flag such as `--time=08:00:00`.

**It is important to recognize that** _jobs_ *need not be in 1:1 correspondence with*
_experiments_. While a 1:1 correspondence may simplify running experiments, it
sometimes leads to a great waste of resources. This is especially the case if
experiments are lightweight while the allocated GPUs are heavyweight, making for a
poor match.


## Why job packing?

Many everyday ML workloads — seed sweeps, small hyperparameter grids, inference over
many files, many reinforcement learning environments — do **not** saturate a modern
GPU. A single training run can leave most of a GPU's resources (streaming
multiprocessors, memory and I/O bandwidth) unutilized, Yet SLURM charges you for the
whole allocation from the moment it starts, whether the GPU is busy or not. On a shared
lab cluster, the cost of running one experiment per `sbatch` job is threefold:

- **Wasted compute.** Dozens of low-utilization jobs add up to GPUs that look "busy" in
  the queue while doing a fraction of the work they could (see the *Identifying GPU
  waste* page in the MILA docs).
- **Queue overhead.** Each small job waits in line on its own, so a 20-seed sweep can
  spend more wall-clock time *queued* than training, and pays job startup (environment
  setup, dataset staging) 20 times.
- **Scheduler pressure.** Thousands of tiny jobs crowd the queue for everyone; a few
  well-packed jobs get the same science done with less contention — and you keep your
  fair-share budget for runs that genuinely need a whole node.

**Job packing** is the practice of requesting *one* allocation and running *many*
experiments inside it concurrently. GNU `parallel` is the packing tool: it launches your
experiments a few at a time, buffers their logs, records statuses, and retries failures.
CUDA **MPS** (Multi-Process Service) makes the packing *efficient*: without it, packed
processes each create their own CUDA context and the GPU time-slices between them;
with MPS running, the processes share a single GPU context and their kernels execute
concurrently, which is exactly what small models need to fill the card.

The three scripts below combine both: one allocation on `--partition=long`, an MPS
server started at the beginning, and `parallel` running up to `PARALLELISM` experiments
at a time.

## Conventions

These job scripts follow the conventions of the example scripts on
[docs.mila.quebec](https://docs.mila.quebec/) (see *Train Your First Model* and the
*Good Practices* examples *Launch many tasks on the same GPU* and *Launch many jobs from
the same shell script*): `#SBATCH` directives requesting GPUs with `--gpus-per-task`,
staging datasets into `$SLURM_TMPDIR`, and running training with `uv run python`.

Three cluster-wide conventions used everywhere below:

- **Partition.** All examples request `--partition=long` (Mila's default partition)
  explicitly. Declare a realistic `--time` so the scheduler can backfill your job.
- **`PARALLELISM`.** Each script reads a single environment variable to control packing,
  with a safe default. `sbatch` forwards your environment by default, so:

  ```bash
  sbatch job.sh                                        # uses the script's default
  PARALLELISM=4 sbatch job.sh                          # forwarded by sbatch (--export=ALL)
  sbatch --export=ALL,PARALLELISM=4,N=1,M=20 job.sh    # explicit, survives --export=NONE setups
  ```

- **CUDA MPS lifecycle.** Start the daemon *before* any CUDA process, point its pipe/log
  directories at `$SLURM_TMPDIR` (per-job, writable scratch), and shut it down on exit so
  no server is left running on the node:

  ```bash
  export CUDA_MPS_PIPE_DIRECTORY=$SLURM_TMPDIR/mps_pipe
  export CUDA_MPS_LOG_DIRECTORY=$SLURM_TMPDIR/mps_log
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  nvidia-cuda-mps-control -d
  trap 'echo quit | nvidia-cuda-mps-control 2>/dev/null || true' EXIT
  ```

  `parallel`-spawned jobs inherit `CUDA_MPS_PIPE_DIRECTORY`/`CUDA_MPS_LOG_DIRECTORY` and
  automatically become MPS clients.

---

## 1. Seed Sweep: `--seed` from N to M on a Single GPU (with MPS)

Pack a range of random seeds onto one GPU. Small models (e.g., the docs' ResNet18 on
CIFAR-10) leave plenty of GPU memory headroom and issue small kernels — exactly the
workload CUDA MPS is made for.

**`job.sh`**

```bash
#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=seed-sweep
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=l40s:1
## Or --gpus-per-task=rtx8000:1 to request a different GPU model
## Or --gpus-per-task=1 for any GPU model
#SBATCH --mem-per-gpu=16G
#SBATCH --time=08:00:00

set -e
module load parallel

# --- Packing control -------------------------------------------------------
PARALLELISM=${PARALLELISM:-2}   # concurrent training runs on this GPU
N=${N:-1}                       # first seed
M=${M:-10}                      # last seed
# -------------------------------------------------------------------------

echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Packing seeds ${N}..${M}, ${PARALLELISM} at a time"

# --- CUDA MPS server (one per job; clients are the parallel workers) ------
export CUDA_MPS_PIPE_DIRECTORY=$SLURM_TMPDIR/mps_pipe
export CUDA_MPS_LOG_DIRECTORY=$SLURM_TMPDIR/mps_log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
trap 'echo quit | nvidia-cuda-mps-control 2>/dev/null || true' EXIT
# Optionally cap each client's share of the SMs for fairness between runs:
# echo set_default_active_thread_percentage 50 | nvidia-cuda-mps-control
# -------------------------------------------------------------------------

# Stage dataset into $SLURM_TMPDIR (fast local scratch on the compute node)
mkdir -p $SLURM_TMPDIR/data
cp /network/datasets/cifar10/cifar-10-python.tar.gz $SLURM_TMPDIR/data/

mkdir -p logs

# One job per seed; each writes its own log so outputs don't interleave.
seq "$N" "$M" | parallel -j "$PARALLELISM" --line-buffer \
  "uv run python main.py --seed {1} > logs/seed_{1}.log"
```

Submit:

```bash
sbatch job.sh                                  # seeds 1..10, 2 concurrent
PARALLELISM=4 N=1 M=30 sbatch job.sh           # more seeds, packed tighter
```

Tips:

- Size `PARALLELISM` to the GPU memory (check with `nvidia-smi` in a test job); MPS
  improves *compute* sharing but does not multiply memory — every process still
  allocates its own weights, activations, and CUDA buffers.
- Divide your CPUs among the packed runs: leave roughly
  `cpus-per-task / PARALLELISM` `DataLoader` workers per run.
- Add `--joblog sweep.log` to the `parallel` command and rerun with `parallel --resume`
  to retry only the seeds that failed.

---

## 2. Hyperparameter Grid from Three Nested `for` Loops (with MPS)

Generate the command list with nested Bash loops (model × learning rate × seed), write it
to a file, and let `parallel` consume it — all processes sharing the GPU through MPS.

**`job.sh`**

```bash
#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=grid-sweep
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=rtx8000:1
#SBATCH --mem-per-gpu=24G
#SBATCH --time=24:00:00

set -e
module load parallel

# --- Packing control -------------------------------------------------------
PARALLELISM=${PARALLELISM:-3}   # concurrent grid points on this GPU
# -------------------------------------------------------------------------

echo "Date: $(date)"
echo "Hostname: $(hostname)"

# --- CUDA MPS server --------------------------------------------------------
export CUDA_MPS_PIPE_DIRECTORY=$SLURM_TMPDIR/mps_pipe
export CUDA_MPS_LOG_DIRECTORY=$SLURM_TMPDIR/mps_log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
trap 'echo quit | nvidia-cuda-mps-control 2>/dev/null || true' EXIT
# -------------------------------------------------------------------------

mkdir -p $SLURM_TMPDIR/data
cp /network/datasets/cifar10/cifar-10-python.tar.gz $SLURM_TMPDIR/data/

# Generate one command per grid point with three nested loops.
commands=$SLURM_TMPDIR/commands.txt
: > "$commands"
for model in resnet18 resnet50; do
  for lr in 0.0001 0.001; do
    for seed in 1 2 3; do
      echo "uv run python main.py --model ${model} --lr ${lr} --seed ${seed}" >> "$commands"
    done
  done
done
wc -l "$commands"   # 12 commands here

# Run the grid through parallel, one line of the file = one job.
# --joblog records status/duration/exit code per command for auditing and retries.
parallel --joblog joblog.tsv --line-buffer -j "$PARALLELISM" :::: "$commands"
```

Submit:

```bash
sbatch job.sh
PARALLELISM=6 sbatch job.sh
```

Tips:

- The `for` loops make the sweep easy to extend (add a dataset loop, filter combos with
  `if`, etc.) without learning parallel's `:::`/`::::` syntax.
- Retry only failed grid points: `parallel --joblog joblog.tsv --resume -j "$PARALLELISM" :::: "$commands"`.
- Because grid points have mixed sizes (`resnet18` vs `resnet50`), MPS keeps the GPU
  busy while the large runs leave SMs idle.

---

## 3. Batch Inference Packed Across a Node's GPUs (one MPS server per GPU)

Embarrassingly parallel scoring of many files, using one `parallel` worker per GPU.
`srun` starts `SLURM_NTASKS` workers (one per requested GPU); each pins itself to its
GPU, starts **its own MPS server** (MPS is per-GPU, so start it *after* setting
`CUDA_VISIBLE_DEVICES`), takes every *n*-th input file, and runs
`PARALLELISM / SLURM_NTASKS` sub-jobs concurrently.

**`job.sh`**

```bash
#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=batch-infer
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:l40s:4
## Or --gres=gpu:rtx8000:4 for a different GPU model
#SBATCH --mem=56G
#SBATCH --time=04:00:00

set -e
module load parallel

# --- Packing control -------------------------------------------------------
PARALLELISM=${PARALLELISM:-16}  # total concurrent predictions across ALL GPUs
# -------------------------------------------------------------------------

echo "Date: $(date)"
echo "Hostname: $(hostname)"

# Total concurrency is split evenly over the one-worker-per-GPU srun tasks.
export PER_TASK=$(( (PARALLELISM + SLURM_NTASKS - 1) / SLURM_NTASKS ))

# Shared MPS directories: srun tasks share the node's $SLURM_TMPDIR.
export CUDA_MPS_PIPE_DIRECTORY=$SLURM_TMPDIR/mps_pipe
export CUDA_MPS_LOG_DIRECTORY=$SLURM_TMPDIR/mps_log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

# Input list (e.g., extracted feature files produced by an earlier job).
find "$SCRATCH/features" -name '*.npy' | sort > "$SLURM_TMPDIR/filelist.txt"
wc -l "$SLURM_TMPDIR/filelist.txt"

export FILELIST="$SLURM_TMPDIR/filelist.txt"
mkdir -p "$SLURM_TMPDIR/predictions"
export PREDIR="$SLURM_TMPDIR/predictions"

srun bash worker.sh

# Collect predictions back to the submission directory.
mkdir -p "$SLURM_SUBMIT_DIR/predictions"
cp "$SLURM_TMPDIR"/predictions/*.json "$SLURM_SUBMIT_DIR/predictions/"

# --- CUDA MPS shutdown (each worker started one server per GPU) ------------
echo quit | nvidia-cuda-mps-control 2>/dev/null || true
```

**`worker.sh`**

```bash
#!/bin/bash
set -e

# Pin this srun task to its own GPU (SLURM_STEP_GPUS holds the GPU ids
# assigned to this task, e.g. "2"), then start a dedicated MPS server for it.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$SLURM_STEP_GPUS}
nvidia-cuda-mps-control -d
trap 'echo quit | nvidia-cuda-mps-control 2>/dev/null || true' EXIT

# Take every SLURM_NTASKS-th file: task SLURM_PROCID of SLURM_NTASKS.
awk -v n="$SLURM_NTASKS" -v p="$SLURM_PROCID" 'NR % n == p' "$FILELIST" \
  | parallel -j "$PER_TASK" --line-buffer \
      "uv run python predict.py --input {} --output ${PREDIR}/{.}.json"
```

Submit:

```bash
sbatch job.sh
PARALLELISM=8 sbatch job.sh     # lighter packing: 2 predictions per GPU
```

Tips:

- Keep `PARALLELISM` consistent with the allocation: 16 sub-jobs on 4 GPUs is fine,
  16 on 1 GPU would just queue inside `parallel` (harmless, but set `--time` accordingly).
- If a worker dies mid-run, add `--joblog "worker_${SLURM_PROCID}.log" --resume` inside
  `worker.sh` and resubmit: files already scored are skipped.
- All workers share one `CUDA_MPS_PIPE_DIRECTORY` here because each server sees a
  different GPU; if you ever see pipe collisions between servers, give each task its own
  subdirectory (`$CUDA_MPS_PIPE_DIRECTORY/$SLURM_PROCID`) set before `nvidia-cuda-mps-control -d`
  and inherited by that task's clients.
- For a CPU-only variant, drop the `--gres` line, the MPS block, and the GPU pinning,
  and default to `PARALLELISM=${PARALLELISM:-$SLURM_CPUS_PER_TASK}`.

---

## Quick Reference

| Goal | How |
| --- | --- |
| Request Mila's long partition | `#SBATCH --partition=long` (also the cluster default) |
| Packing control knob | `PARALLELISM=${PARALLELISM:-4}` … `parallel -j "$PARALLELISM"` |
| Override at submission | `PARALLELISM=4 sbatch job.sh` or `sbatch --export=ALL,PARALLELISM=4 job.sh` |
| Load GNU parallel | `module load parallel` (fallback: `module load gcc/9.3.0 parallel/20220822`) |
| Start CUDA MPS per job | `CUDA_MPS_PIPE_DIRECTORY=$SLURM_TMPDIR/mps_pipe nvidia-cuda-mps-control -d` |
| Stop CUDA MPS | `echo quit \| nvidia-cuda-mps-control` (via `trap ... EXIT`) |
| Per-GPU MPS on multi-GPU jobs | pin `CUDA_VISIBLE_DEVICES` first, then `nvidia-cuda-mps-control -d` |
| Seed range N..M | `seq "$N" "$M" \| parallel ... --seed {1}` |
| Grid from nested loops | write commands to a file, then `parallel :::: commands.txt` |
| One worker per GPU | `srun bash worker.sh` + `CUDA_VISIBLE_DEVICES=$SLURM_STEP_GPUS` |
| Retry only failures | `--joblog run.log` then `--resume` (or `--resume-failed`) |
| Keep packed-run logs separate | `--line-buffer`, or `--results DIR/` |

> **MPS caveats.** The MPS control daemon must be started *before* any CUDA context
> exists in the allocation, and clients must see the same `CUDA_MPS_PIPE_DIRECTORY` /
> `CUDA_MPS_LOG_DIRECTORY`. Always shut the server down at job end (`trap ... EXIT`) so
> the next tenant of the node starts clean. MPS shares compute, not memory: budget
> `PARALLELISM` against per-process GPU memory with `nvidia-smi`.

> Compute nodes have no internet access. Use `uv run --offline` (and `--locked` for
> reproducibility) once your project environment is set up, as recommended in the
> MILA docs' job scripts.
