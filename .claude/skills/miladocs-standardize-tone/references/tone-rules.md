# Tone Rules for Mila Documentation

Complete rules with before/after examples. Apply all rules when standardizing a page.

---

## Rule 1: Third-Person Pronouns

Replace second-person pronouns ("you", "your", "yourself") with third-person or imperative constructions.

**Strategy A — Imperative mood** (preferred for instructions):
Remove the subject entirely and start with the verb.

| Before | After |
|--------|-------|
| You can run the script with `uv run`. | Run the script with `uv run`. |
| You need to create a virtual environment first. | Create a virtual environment first. |
| You should submit your job using `sbatch`. | Submit the job using `sbatch`. |
| Once you've done that, you can proceed. | Once that is done, proceed. |

**Strategy B — Restructure to third-person** (for descriptive/explanatory sentences):
Use "the user", "researchers", or a noun describing the reader role.

| Before | After |
|--------|-------|
| If you are running a multi-GPU job... | For multi-GPU jobs... |
| Your job will be queued until a node is available. | The job is queued until a compute node is available. |
| This allows you to monitor progress. | This allows monitoring of job progress. |

**Strategy C — Passive or noun phrase** (when neither A nor B fits naturally):

| Before | After |
|--------|-------|
| You will see the output below. | The output appears below. |
| This is what you should expect to see. | Expected output: |

**Do NOT change:**
- "your" in file paths that are meant to be replaced by the user (e.g., `your_script.py`, `/home/your_username/`) — these are placeholders, not pronouns.
- Second-person in quoted or cited content.
- Comments inside code blocks.

---

## Rule 2: Active Voice

Prefer active constructions over passive ones.

| Before (passive) | After (active) |
|------------------|----------------|
| The job should be submitted with `sbatch`. | Submit the job with `sbatch`. |
| A GPU node can be requested using `--gres`. | Request a GPU node using `--gres`. |
| The environment is activated automatically. | The environment activates automatically. *(or: uv activates the environment automatically.)* |
| The results are saved to `$SCRATCH`. | The results save to `$SCRATCH`. *(or: SLURM saves results to `$SCRATCH`.)* |
| It is recommended to use `uv`. | Use `uv` for environment management. |
| Care should be taken to... | Take care to... |

**Exceptions** — passive is acceptable when:
- The agent is unknown or irrelevant: "The cluster is maintained by IDT."
- The subject is the natural focus: "Logs are written to `/tmp/slurm-<jobid>.out`."

---

## Rule 3: Avoid Vague and Condescending Language

Remove words that imply a task is trivial or obvious. These create friction for readers who find the task difficult.

**Words to remove or replace:**

| Word/phrase | Action |
|-------------|--------|
| simply | remove |
| just | remove (e.g., "just run" → "run") |
| easy / easily | remove or rephrase |
| obviously | remove |
| of course | remove |
| clearly | remove |
| straightforward | remove or rephrase |
| trivially | remove |
| as expected | remove unless following an output example |
| needless to say | remove |

**Examples:**

| Before | After |
|--------|-------|
| Simply run `sbatch job.sh` to submit the job. | Run `sbatch job.sh` to submit the job. |
| It's easy to monitor your jobs with `squeue`. | Monitor jobs with `squeue`. |
| Just add `--gres=gpu:1` to your script. | Add `--gres=gpu:1` to the script. |
| Of course, you'll need to load the module first. | Load the module first. |

---

## Rule 4: Present Tense for Instructions

Use present tense (or imperative mood) for instructions. Future tense distances the reader from the action.

| Before (future) | After (present/imperative) |
|-----------------|---------------------------|
| You will create a file called `job.sh`. | Create a file called `job.sh`. |
| This will install all dependencies. | This installs all dependencies. |
| The script will print the GPU name. | The script prints the GPU name. |
| You'll need to activate the environment. | Activate the environment. |
| Running this command will allocate a node. | Running this command allocates a node. |

**Exception** — future tense is fine for:
- Describing consequences: "If the job fails, SLURM will send an email."
- Time-bound outcomes: "The allocation will expire after 1 hour."

---

## Rule 5: Consistent Mila Terminology

Use the canonical terms for Mila infrastructure components.

| Use this | Not this |
|----------|----------|
| `cluster` | server, supercomputer, HPC, system |
| `compute node` | worker, worker node, machine, host |
| `login node` | head node, master node, front-end |
| `job` (for SLURM) | task, workload, process, run |
| `SLURM` (all caps) | Slurm, slurm |
| `$SCRATCH` | scratch, scratch storage, /scratch |
| `$HOME` | home directory, home folder |
| `sbatch` | batch submission, batch job command |
| `salloc` | interactive allocation command |
| `squeue` | job queue command |
| `module load` | loading a module |
| `milatools` | mila-tools, mila tools |
| `mila code` | mila-code, milacode |

**Capitalization:**
- `SLURM` — always all-caps
- `GPU`, `CPU` — always all-caps
- `SSH` — always all-caps
- `MFA` — always all-caps
- `PyTorch`, `JAX` — exact capitalization

---

## Applying Rules to Specific Constructs

### Headings
Headings often use second-person implicitly. Prefer gerund or noun phrases.

| Before | After |
|--------|-------|
| "Getting your first job to run" | "Running your first job" → "Running a first job" |
| "What you will learn" | "What this guide covers" |
| "Your next steps" | "Next steps" |

### Admonition content
Apply all rules inside `!!! note`, `!!! warning`, etc. blocks.

### Grid card text
Apply all rules to the description text inside grid cards.

### Do NOT modify
- Content inside ` ```bash ` or other code fences
- Content inside `<div class="result">` blocks
- YAML frontmatter
- `--8<--` include directives
- `{%  include-markdown ... %}` directives
- URLs and link targets
- SLURM flags and command-line arguments
