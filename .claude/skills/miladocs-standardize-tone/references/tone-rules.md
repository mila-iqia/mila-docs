# Tone Rules for Mila Documentation

Complete rules with before/after examples. Apply all rules when standardizing a page.

## Goal

The target tone is **formal and objective** — the register of a technical reference manual, not a tutorial or conversation. Rules below enforce this: they remove direct address, vague filler, and casual constructions in favor of precise, impersonal language.

---

## Rule 1: Second-Person Pronouns

Replace second-person pronouns ("you", "your", "yourself") using the strategy that produces the most natural result. Do not force a rewrite that is syntactically awkward or harder to parse than the original.

**Strategy A — Imperative mood** (preferred for action steps):
Remove the subject and start with the verb.

| Before | After |
|--------|-------|
| You can run the script with `uv run`. | Run the script with `uv run`. |
| You need to create a virtual environment first. | Create a virtual environment first. |
| You should submit your job using `sbatch`. | Submit the job using `sbatch`. |
| Once you've done that, you can proceed. | Once that is done, proceed. |

**Strategy B — Restructure** (for descriptive/explanatory sentences):
Rewrite using a noun phrase or gerund when no subject can be dropped.

| Before | After |
|--------|-------|
| If you are running a multi-GPU job... | For multi-GPU jobs... |
| Your job will be queued until a node is available. | The job is queued until a compute node is available. |
| This allows you to monitor progress. | This allows monitoring of job progress. |
| You will see the output below. | The output appears below. |

**Strategy C — Possessives ("your X"):**
Choose based on whether the reader genuinely owns or is assigned the item.

- **Personally owned items** — credentials, personal relationships, owned devices (e.g. password, supervisor): **preserve "your"**.
- **Tools and shared resources** — software tools, cluster infrastructure, job artifacts: **use definite or indefinite article**.

| Before | After |
|--------|-------|
| your terminal | a terminal |
| your browser | a browser |
| your job ID | the job ID |
| your script | the script |

**Rule of thumb:** If the item is a credential, a personal relationship, or a personally owned device, keep "your". If it is a tool or infrastructure the reader happens to be using, replace it. Use "the user's X" only as a last resort when neither article fits.

**Do NOT change:**
- "your" in file paths that are meant to be replaced by the user (e.g., `your_script.py`, `/home/your_username/`) — these are placeholders, not pronouns.
- Second-person in quoted or cited content.
- Comments inside code blocks.
- Any case where the rewrite would be syntactically awkward or harder to understand.

---

## Rule 2: Active Voice

Prefer active constructions over passive ones.

| Before (passive) | After (active) |
|------------------|----------------|
| The job should be submitted with `sbatch`. | Submit the job with `sbatch`. |
| A GPU node can be requested using `--gres`. | Request a GPU node using `--gres`. |
| The environment is activated automatically. | The environment activates automatically. *(or: uv activates the environment automatically.)* |
| The results are saved to `$SCRATCH`. | The results save to `$SCRATCH`. *(or: Slurm saves results to `$SCRATCH`.)* |
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
- Describing consequences: "If the job fails, Slurm will send an email."
- Time-bound outcomes: "The allocation will expire after 1 hour."

---

## Rule 5: Consistent Mila Terminology

Use the canonical terms for Mila infrastructure components.

| Use this | Not this |
|----------|----------|
| `cluster` | server, supercomputer, HPC, system |
| `compute node` | worker, worker node, machine, host |
| `login node` | head node, master node, front-end |
| `job` (for Slurm) | task, workload, process, run |
| `Slurm` | SLURM, slurm |
| `$SCRATCH` | scratch, scratch storage, /scratch |
| `$HOME` | home directory, home folder |
| `sbatch` | batch submission, batch job command |
| `salloc` | interactive allocation command |
| `squeue` | job queue command |
| `module load` | loading a module |
| `milatools` | mila-tools, mila tools |
| `mila code` | mila-code, milacode |

**Capitalization:**
- `Slurm` — title case (not all-caps)
  - Exception: `$SLURM_*` environment variable names
    (e.g. `$SLURM_TMPDIR`) must stay all-caps — they are
    shell variable names, not references to the scheduler.
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
- Slurm flags and command-line arguments
