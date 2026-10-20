# Mila Technical Documentation — Reference Summary for LLM Agents

> Source: https://docs.mila.quebec (source repo: https://github.com/mila-iqia/mila-docs),
> summarized from the repository state of 2026-10-20.
> Scope: Mila organisation & support, Mila cluster (access, MFA, SSH, Slurm, partitions,
> nodes, storage, tools), DRAC (Digital Research Alliance of Canada) and PAICE clusters,
> Python environments (uv), experiment tracking, compute-efficiency policy, containers,
> distributed training, FAQ.
>
> This file is meant to **replace** Mila's own `all_docs_condensed.md`: it contains everything
> in it (checked against the docs source; outdated bits fixed) plus more coverage.
>
> **Instructions for the agent using this file:** answer Mila / Mila-cluster / DRAC questions
> from this file first. Policies, quotas, partition limits and allocations change over time:
> when giving numbers, tell the user they can verify them (`sinfo`, `disk-quota`,
> https://docs.mila.quebec, https://docs.alliancecan.ca). If something is not covered,
> say so and point to the support channels in §17.

---

## 0. Key URLs and contacts (quick lookup)

| What | Where |
|---|---|
| Docs | https://docs.mila.quebec |
| Docs source / issues / PRs | https://github.com/mila-iqia/mila-docs |
| Condensed docs for LLMs (official) | https://raw.githubusercontent.com/mila-iqia/mila-docs/master/all_docs_condensed.md |
| IT support portal | https://it-support.mila.quebec (also it-support@mila.quebec) |
| Helpdesk (service desk portal 5) | https://mila-iqia.atlassian.net/servicedesk/customer/portal/5 |
| MyMila support (accounts/onboarding admin) | https://mila-iqia.atlassian.net/servicedesk/customer/portal/8 · MyMila portal https://portal.mila.quebec/ |
| MFA portal | https://mfa.mila.quebec |
| Mila intranet | https://sites.google.com/mila.quebec/mila-intranet (also https://intranet.mila.quebec/) |
| IT Onboarding Training (read + quiz) | https://sites.google.com/mila.quebec/mila-intranet/it-infrastructure/it-onboarding-training |
| Grafana cluster dashboard | https://dashboard.server.mila.quebec |
| Compute Utilization Dashboard (SARC) | https://sarc.mila.quebec/dash/metrics (Mila credentials) |
| JupyterHub (Mila) | https://jupyterhub.server.mila.quebec (Google OAuth, @mila.quebec account) |
| Mila datasets browser | https://datasets.server.mila.quebec/ |
| Comet (Mila org) | https://comet.mila.quebec/ |
| Slack | `#mila-cluster`, `#compute-canada` (DRAC), `#tamia-cluster` |
| Cheat sheet PDF | https://docs.mila.quebec/_static/2026-05-05_Mila_compute_cheat_sheet_v4.pdf (feedback: idt.cheatsheet@mila.quebec) |
| DRAC CCDB (accounts) | https://ccdb.alliancecan.ca |
| DRAC docs / status | https://docs.alliancecan.ca · https://status.alliancecan.ca/ |
| Add to Mila DRAC global allocation | email **ccdb-accounts@mila.quebec** with your CCRI |
| DRAC support | support@tech.alliancecan.ca · renewals: renewals@tech.alliancecan.ca |
| milatools | https://github.com/mila-iqia/milatools |
| cluv | https://mila-iqia.github.io/cluv/ |
| milalib | https://github.com/mila-iqia/milalib |
| Research Project Template | https://mila-iqia.github.io/ResearchTemplate/ |
| Mila Claude Code skills | https://github.com/mila-iqia/skills |

---

## 1. Mila, IDT and the documentation

- The docs cover running scientific/data-intensive computing at Mila and the resources
  available to members. Main audience: Mila researchers with Mila-cluster access.
- **IDT** = Innovation, Development and Technology team. They help with running experiments,
  efficient ML code, ML tools; they run the compute-efficiency program ("Safety net").
- **IT Support** handles accounts, SSH problems, VPN, hardware, access management.
- **MyMila** handles administrative onboarding and community questions.
- **IDT Office Hours**: **Lab A**, **Tuesdays 3pm–5pm** and **Wednesdays 2pm–4pm**, also on
  Google Meet (link in the mila-core Google calendar; see the intranet for how to subscribe).
  Printed cheat sheets are available there.
- **Cheat sheet** (v4, 2026-05-05): updated about once a year (usually around April, when
  DRAC allocations are announced). Known errata:
  - Preemption is more subtle than the sheet says: `long` jobs can be preempted by `main`
    jobs; `main` jobs are never preempted by other `main` jobs (whatever the fair-share usage);
    `unkillable` jobs can preempt everything except other `unkillable` jobs.
  - TamIA also has 8×H200 nodes (not just H100).
  - The cluster "STACC" will be called **Lynx**.
  - The URL for joining the Mila DRAC "mega allocation" is outdated (correct procedure: §12).

### Acknowledging Mila in publications
Recommended wording: *"This research was enabled in part by compute resources provided by
Mila (mila.quebec)."* or *"...by compute resources, software and technical help provided by
Mila (mila.quebec)."* If DRAC resources are used, acknowledge the Alliance as well; for PAICE,
acknowledge DRAC, the regional partner and the AI institute managing the cluster.

---

## 2. Who has access to the Mila cluster

- Professors/PIs are either **Core** or **Affiliated** researchers.
- Students/postdocs inherit access from their supervisor: if the supervisor **or co-supervisor**
  is a Core professor → access granted. If both are Affiliated → no access.
- Access is granted to: **current students of Core professors, Core professors, Staff**.
  Access duration is tied to the stay duration set when the account was created.
- Students of Core professors are also eligible for the Mila DRAC global allocation
  (`rrg-bengioy-ad`), the Mila WandB organization and the Mila Comet organization.

---

## 3. Getting started (onboarding checklist)

### 3.1 Get a Mila account and enable cluster access
1. Ask your supervisor to invite you into the Mila organization → `@mila.quebec` account.
   A confirmation email from IT support follows. Delays → contact MyMila support (portal 8).
2. Read the **IT Onboarding Guide** on the intranet and **submit the quiz**. During onboarding
   you submit your **SSH public key** (max **4 public keys per user**).
3. After passing the quiz, IT sends connection details (incl. **cluster username**) by email or
   Slack. Access can take **up to 48 h**.
4. IT sends an email to activate **MFA**.
5. Accept the **compute cluster terms and conditions** (Google form linked in the docs).

### 3.2 Set up MFA (required before any SSH connection)
- Two factors: SSH key (1st) + a second factor (TOTP, Push or email token).
- Find the email titled *"Votre accès temporaire registrationcode / Your temporary access
  registrationcode"* → contains a **one-time registration token**.
- Go to https://mfa.mila.quebec:
  - **Username = cluster username** (NOT the @mila.quebec email).
  - **Password = the registration token** (NOT the Mila password).
- **Immediately enroll a TOTP token** ("Enroll token" → TOTP → scan QR code in an
  authenticator app: privacyIDEA Authenticator, Authy, Google Authenticator, Microsoft
  Authenticator, any RFC 6238 app). After the first visit the portal **only accepts TOTP**;
  leaving without a TOTP locks you out and you need a new token from IT support.
- Optionally add more tokens: **Push** (privacyIDEA app; recommended for frequent access),
  another TOTP, or email. Hardware keys (YubiKey FIDO2/WebAuthn) are planned.

| Access | First login | Later logins |
|---|---|---|
| MFA web portal | email registration token | TOTP only |
| Cluster SSH | n/a | TOTP, Push or email code |

- If both Push and OTP are enrolled, SSH always shows an OTP prompt: type the code + Enter, or
  approve the push on the phone first and then press Enter with an empty prompt.
- Troubleshooting: TOTP rejected → enable automatic time sync on the phone. Lost phone →
  contact IT support immediately to reset tokens.

### 3.3 Install tools locally and connect
- **Windows**: install WSL2 (`wsl --install Ubuntu` in PowerShell, reboot, open "Ubuntu") and
  run everything inside WSL.
- Install `uv` locally: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Install milatools: `uv tool install --upgrade milatools` (older: `pip install milatools`;
  docs assume milatools ≥ 0.0.17).
- Run `mila init` (asks for your cluster username): creates `~/.ssh/config` entries
  (`mila`, `mila1`..`mila5`, `mila-cpu`, `cn-????` compute-node rules), an SSH key (`~/.ssh/id_ed25519_mila`), and adds the
  public key to `~/.ssh/authorized_keys` on the cluster (needed to reach compute nodes).
- Test: `ssh mila`, enter the 6-digit OTP (not echoed). You land on a **login node**.
  - No OTP prompt → redo milatools install/config. No banner after OTP → redo MFA setup.
- Install `uv` on the cluster too (same curl command, on a login node). If `uv` is not
  found, reconnect to refresh `PATH`.

### 3.4 First job (interactive, VSCode)
```bash
ssh mila 'mkdir -p CODE/my_first_job'          # from your laptop
mila code CODE/my_first_job --alloc --gres=gpu:1 --cpus-per-task=2 --mem=16G --time=01:00:00
# everything after --alloc is passed to salloc; VSCode (or Cursor) opens on a compute node
uv run python main.py                          # in the VSCode terminal
```
Close VSCode and press Ctrl+C in the terminal to release the allocation. If VSCode hangs on
"Opening Remote...", enable `remote.SSH.showLoginTerminal` (it is probably waiting for the OTP).

### 3.5 First batch training (ResNet18 on CIFAR-10)
- Open VSCode on a CPU node to edit: `mila code CODE/train_first_model --alloc --cpus-per-task=2 --mem=16G --time=01:00:00`
- `job.sh` requests resources, **stages data** from `/network/datasets/cifar10/` into
  `$SLURM_TMPDIR/data`, then runs `srun uv run python main.py`.
- `sbatch job.sh` → "Submitted batch job N"; monitor with `squeue --me`; output in
  `slurm-<JOBID>.out`. `ReqNodeNotAvail` = no matching node free right now (not an error):
  wait, or `scancel` and resubmit with another GPU type, e.g. `--gres=gpu:rtx8000:1`.

---

## 4. SSH details (Mila cluster)

- **Public-key auth only** (passwords disabled since March 31, 2025) + MFA.
- Login: host `login.server.mila.quebec`, **port 2222**, user = cluster username,
  `PreferredAuthentications publickey,keyboard-interactive`.
  - `ssh -p 2222 <user>@login.server.mila.quebec` (load-balancer → least-loaded login node)
  - Specific node: `ssh -p 2222 <user>@login-X.login.server.mila.quebec` (X = 1..4, a login-5
    entry exists in the sample config).
- Manual `~/.ssh/config` template:
```sshconfig
Host mila             login.server.mila.quebec
    Hostname          login.server.mila.quebec
Host mila1    login-1.login.server.mila.quebec
    Hostname  login-1.login.server.mila.quebec
# ... mila2..mila5 likewise
Host cn-????
    Hostname             %h.server.mila.quebec
Match host !*login.server.mila.quebec,*.server.mila.quebec
    Hostname                 %h
    ProxyJump                mila
Match host           *login.server.mila.quebec
    Port                     2222
    ServerAliveInterval      120
    ServerAliveCountMax      5
Match host *.server.mila.quebec
    PreferredAuthentications publickey,keyboard-interactive
    AddKeysToAgent           yes
    # ForwardAgent yes        (optional)
    # UseKeychain  yes        (macOS only)
    User                     CHANGEME_YOUR_MILA_USERNAME
```
- **Reduce MFA prompts**: add `ControlMaster auto` and `ControlPersist yes` to the mila entry →
  MFA once per machine boot (Linux/macOS; Windows via WSL). cluv requires ControlMaster.
- `scp file.zip mila:scratch/uploaded.zip`, `rsync -avz mila:path/ local/`,
  `ssh mila squeue -u '$USER'` (single-quote remote variables).
- Login-node host key fingerprints:
  `SHA256:baEGIa311fhnxBWsIZJ/zYhq2WfCttwyHRKzAb8zlp8` (ECDSA),
  `SHA256:Xr0/JqV/+5DNguPfiN5hb8rSG+nBAcfVCJoSyrR0W0o` (RSA),
  `SHA256:gfXZzaPiaYHcrPqzHvBi6v+BWRS/lXOS/zAjOKeoBJg` (ED25519). Mismatch → do not trust.
- Compute-node fingerprints:
  `SHA256:hGH64v72h/c0SfngAWB8WSyMj8WSAf5um3lqVsa7Cfk` (ECDSA),
  `SHA256:4Es56W5ANNMQza2sW2O056ifkl8QBvjjNjfMqpB7/1U` (RSA),
  `SHA256:gUQJw6l1lKjM1cCyennetPoQ6ST0jMhQAs/57LhfakA` (ED25519).
- **SSH to a compute node**: only allowed while you have a job on it; the session is adopted
  by the job and ends with it. Requires your public key in `~/.ssh/authorized_keys` on the
  cluster (`chmod 600 ~/.ssh/authorized_keys; chmod 700 ~/.ssh; chmod go-w ~`) or agent
  forwarding. From laptop: `ssh -J mila USER@cn-xxxx` or with the wildcard rule
  `ssh cn-xxxx.server.mila.quebec`.
- **`ssh mila-cpu`** (created by `mila init`): auto-allocates a CPU on a compute node; multiple
  connections reuse the same job. Good for light work (VSCode editing, building, tests).
- Keys: generate with `ssh-keygen -t ed25519` (or `-t rsa -b 3072`); regenerate a public key
  with `ssh-keygen -y -f ~/.ssh/id_ed25519`. Use a passphrase. **Never share the private key**;
  Mila will never ask for it (phishing).
- Shell: default `/bin/bash`; Zsh is also supported; Mila doesn't debug other shell setups.
  Change shell → support ticket.
- **Login-node reboots**: 1st Tuesday of each month 2:00 PM EST → login-1 & login-2;
  last Tuesday 2:00 PM EST → login-3 & login-4. Unsaved editor work is lost; jobs are not
  affected.
- Brute-force protection: too many failed connections → IP banned, auto-unban after **1 hour**
  ("connection refused").
- How it works: key challenge → MFA prompt (TOTP/email code or push) → access granted.

### Login-node etiquette (enforced)
Login nodes are entry points only. Rule of thumb: **never run anything taking more than a few
seconds** unless it mostly sleeps or moves data.
- ❌ Training scripts, compiling CUDA packages (`pip install` of vllm/flash-attn from source),
  TensorBoard, VSCode remote server, building containers, creating torrent checksums.
- ✅ Editing with vim/nano/emacs, `cp`/`mv`, downloading a dataset, `tmux`, ssh to compute
  nodes, a sleeping launcher script (e.g. submitit HPO driver), Slurm commands.

---

## 5. Slurm on the Mila cluster

### 5.1 Concepts
- **Job** = resource allocation for a time, created by `sbatch` (batch) or `salloc`
  (interactive). **Step** = created by each `srun` inside a job (by default uses all job
  resources). **Task** = one instance of the command launched by a step; rank =
  `$SLURM_PROCID` (0..N-1). A task runs on exactly one node.
- Think about "how many times": once per job, per node (e.g. dataset staging to
  `$SLURM_TMPDIR`), per GPU (DDP process), etc.
  - Stage-in once per node: `srun --ntasks-per-node=1 --ntasks=$SLURM_NNODES tar -xf $SCRATCH/ds.tar -C $SLURM_TMPDIR`
  - DDP: `srun python my_script.py` (1 task/GPU) or `srun --ntasks-per-node=1 torchrun ...`
  - Stage-out once: `srun --ntasks=1 cp -a $SLURM_TMPDIR/results $SCRATCH/results/$SLURM_JOBID`
- `srun` outside a job creates a job (discouraged). Prefer `--ntasks-per-node` over bare
  `--ntasks` for GPU jobs (deterministic placement, local-rank → GPU binding, `--mem` is per node).
- Working directory of a batch job = directory where `sbatch` was run. Directives can be on the
  command line or as `#SBATCH` lines. Default output: `slurm-<JOBID>.out` in the submit dir;
  change with `--output=logs/%x-%j.out --error=logs/%x-%j.err` (%x name, %j job id).
- If no partition is given, GPU jobs go to **`long`** and CPU jobs to **`long-cpu`**
  ("Using default long partition").
- Max **1000 jobs** queued per user (MaxSubmitJobs). Max **2000 tasks/PIDs per job**
  (`fork: retry: Resource temporarily unavailable`).
- Interactive jobs can last up to a week — always set `--time`. Interactive sessions are tied
  to the login node they were started from.

### 5.2 Common options
| Option | Meaning |
|---|---|
| `-n/--ntasks` | number of tasks |
| `--ntasks-per-node`, `--nodes` | task layout |
| `-c/--cpus-per-task` | cores per task |
| `-t/--time` | walltime (e.g. `1:00:00`, `2-00:00:00`) |
| `--mem`, `--mem-per-gpu`, `--mem-per-cpu` | memory (`--mem` is per node) |
| `--gres=gpu:N`, `--gres=gpu:<model|arch|mem>:N` | GPUs, e.g. `gpu:a100l:2`, `gpu:48gb:1`, `gpu:rtx8000:1` |
| `--gpus-per-task=l40s:1`, `--gpus=1`, `--ntasks-per-gpu=N` | alternative GPU forms |
| `--constraint=` | node features: `nvlink`, `"dgx&ampere"`, `ampere`, `80gb`... |
| `-p/--partition` | partition (§6) |
| `--tmp=200G` | local disk |
| `--array=1-100` | job array (`$SLURM_ARRAY_TASK_ID`) |
| `--requeue`, `--signal=B:TERM@300` | requeue + SIGTERM 5 min before time limit |
| `--account=` | required on DRAC/PAICE (§12–13) |

Small requests schedule faster ("small jobs always run first").

### 5.3 Essential commands
```bash
sbatch job.sh [args]            # submit (extra args go to the script, use "$@" inside)
salloc --gres=gpu:1 -c 2 --mem=12000 -t 2:00:00     # interactive shell on a compute node
srun <cmd>                      # run a step inside a job
squeue --me   /  squeue -u $USER / squeue -j <id>
squeue -Ojobid,name,username,partition,state,timeused,nodelist,gres,tres
sacct -j <id> --format=JobID,State,ExitCode,Elapsed,MaxRSS,ReqMem
sacct -u $USER --start=YYYY-MM-DD ; sacct -j <id> -o NodeList
scontrol show job <id>          # details, incl. Restarts= (preemption count)
scontrol show -d job <id> | grep GRES   # which GPU index you got (also $SLURM_JOB_GPUS)
scontrol show node <node> ; scontrol requeue <id>
scancel <id> ; scancel --me ; scancel -u $USER ; scancel -t PD (pending only)
scancel --signal=TERM <id>      # cancel but allow requeue/cleanup (plain scancel → FAILED, not requeueable)
sinfo ; sinfo -N -l ; sinfo -Ogres:27,nodelist,features -tidle,mix,alloc
savail                          # Mila custom: list available GPUs
sattach --pty <jobid>.0         # re-attach a dropped interactive job
srun --preserve-env ...         # pass current env to interactive job
```
Job states: `PD` pending, `R` running, `CG` completing; sacct: `COMPLETED`, `FAILED`,
`TIMEOUT`, `OUT_OF_MEMORY`. Pending reasons: `(Priority)`/`(Resources)` normal;
`(QOSMaxJobsPerUserLimit)` too many jobs; `(ReqNodeNotAvail)` impossible/unavailable
constraint or nodes down/reserved.

### 5.4 Good-practice batch script (Mila)
```bash
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=l40s:1      # or rtx8000:1, or 1 for any GPU
#SBATCH --mem-per-gpu=16G
#SBATCH --time=00:15:00
set -e
mkdir -p $SLURM_TMPDIR/data
cp /network/datasets/cifar10/cifar-10-python.tar.gz $SLURM_TMPDIR/data/   # stage data
srun uv run python main.py          # uv syncs env from uv.lock (use --offline w/o internet, --locked for reproducibility)
cp -r $SLURM_TMPDIR/results $SCRATCH/   # save what you need before job end
```
Reproducibility with `torch.use_deterministic_algorithms(True)`: `export CUBLAS_WORKSPACE_CONFIG=:4096:8` (slower).

Matching `pyproject.toml` (single-GPU example):
```toml
[project]
name = "single-gpu-example"
version = "0.1.0"
requires-python = ">=3.11,<3.14"
dependencies = ["rich>=14.0.0", "torch>=2.7.1", "torchvision>=0.22.1", "tqdm>=4.67.1"]
```

Matching `main.py` skeleton (ResNet18 on CIFAR-10, the base of all Mila minimal examples):
```python
import argparse, os, random, sys
from pathlib import Path
import numpy as np, torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from tqdm import tqdm

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)
    torch.random.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    assert torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = torch.device("cuda", 0)
    model = resnet18(num_classes=10).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    data_path = Path(os.environ.get("SLURM_TMPDIR", ".")) / "data"   # data staged by job.sh
    train_ds, val_ds, _ = make_datasets(str(data_path))
    nw = get_num_workers()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=nw, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=nw, shuffle=False)

    for epoch in range(args.epochs):
        model.train()
        for x, y in tqdm(train_loader, desc=f"epoch {epoch}", disable=not sys.stdout.isatty()):
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        val_loss, val_acc = validation_loop(model, val_loader, device)
        print(f"Epoch {epoch}: val_loss={val_loss:.3f} val_acc={val_acc:.2%}")

@torch.no_grad()
def validation_loop(model, loader, device):
    model.eval(); total, n, correct = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total += F.cross_entropy(logits, y).item(); n += x.shape[0]
        correct += logits.argmax(-1).eq(y).sum().item()
    return total, correct / n

def make_datasets(path, val_split=0.1, seed=42):
    train = CIFAR10(root=path, transform=transforms.ToTensor(), download=True, train=True)
    test = CIFAR10(root=path, transform=transforms.ToTensor(), download=True, train=False)
    n_val = int(val_split * len(train))
    train, val = random_split(train, (len(train) - n_val, n_val), torch.Generator().manual_seed(seed))
    return train, val, test

def get_num_workers():   # DataLoader workers = CPUs allocated to the task
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else torch.multiprocessing.cpu_count()

if __name__ == "__main__":
    main()
```
Full sources of all examples: https://github.com/mila-iqia/mila-docs/tree/master/docs/examples

### 5.5 Preemption
- Priority: `unkillable > main > long`. Preempted jobs are **killed without notice and
  automatically requeued** on the same partition, restarting from scratch (detect via
  `Restarts` > 0 in `scontrol show job`, or `$SLURM_RESTART_COUNT`).
- **`-grace` partitions** (`main-grace`, `long-grace`, `main-cpu-grace`, `long-cpu-grace`):
  same limits, but **120 s grace**: `SIGCONT` + `SIGTERM`, then `SIGKILL`; **not
  automatically requeued** (call `sbatch` yourself in the trap if you want to resubmit).
```bash
exit_script() { echo "Preempted, saving"; trap - SIGTERM; kill -- -$$; }
trap exit_script SIGTERM
python3 my_script
```
- Use **checkpointing** (§10.4) so that requeued jobs resume.

---

## 6. Mila partitions

| Partition | Max resources per job | Max time | Notes |
|---|---|---|---|
| `unkillable` | 6 CPUs, 32G, 1 GPU | 2 days | highest priority, not preemptible |
| `unkillable-cpu` | 2 CPUs, 16G | 2 days | CPU only |
| `short-unkillable` | 1000G, 4 GPUs | **3 hours** | only 4-GPU nodes (cn-g A100-80GB, cn-l L40S) + H100 nodes (cn-n, max 4 of 8 GPUs per job) |
| `main` | 8 CPUs, 48G, 2 GPUs | 5 days | standard priority |
| `main-cpu` | 8 CPUs, 64G | 5 days | CPU only |
| `long` | no resource limit | 7 days | lowest priority, preemptible; **default**; all GPUs except H100 |
| `long-cpu` | no resource limit | 7 days | CPU only; default for CPU jobs |

- **H100 GPUs are ONLY available in `short-unkillable`.**
- Requesting `unkillable`/`main`/`long` for a CPU-only job auto-translates to the `-cpu` variant.
- `-grace` variants exist (§5.5).
- Mila tries to guarantee **at least one GPU per student, at all times, in interactive mode**
  (a floor, obtainable almost immediately). Don't camp on non-preemptible partitions to avoid
  writing checkpointing code (lowers your priority).

---

## 7. Mila cluster hardware (all x86_64)

| Nodes | GPU | GPU mem (GB) | GPUs/node | CPUs | RAM (GB) | Optimal bundle GPU:CPU:RAM | Local disk (TB) | Features |
|---|---|---|---|---|---|---|---|---|
| cn-a[001-011] | RTX8000 | 48 | 8 | 40 | 384 | 1:5:47G | 3.6 | turing,48gb |
| cn-b[001-005] | V100 | 32 | 8 | 40 | 384 | 1:5:47G | 3.6 | volta,nvlink,32gb |
| cn-c[001-040] | RTX8000 | 48 | 8 | 64 | 384 | 1:8:47G | 3 | turing,48gb |
| cn-g[001-029] | A100 | 80 | 4 | 64 | 1024 | 1:16:248G | 7 | ampere,nvlink,80gb |
| cn-i001 | A100 | 80 | 4 | 64 | 1024 | 1:16:250G | 3.6 | ampere,80gb |
| cn-j001 | A6000 | 48 | 8 | 64 | 1024 | 1:16:125G | 3.6 | ampere,48gb |
| cn-k[001-004] | A100 | 40 | 4 | 48 | 512 | 1:12:125G | 3.6 | ampere,nvlink,40gb |
| cn-l[001-091] | L40S | 48 | 4 | 48 | 1024 | 1:12:250G | 7 | lovelace,48gb |
| cn-n[001-002] | H100 | 80 | 8 | 192 | 2048 | 1:24:268G | 35 | hopper,nvlink,80gb |
| cn-d[001-002] (DGX) | A100 | 40 | 8 | 128 | 1024 | 1:16:124G | 14 | ampere,nvlink,dgx,40gb |
| cn-d[003-004] (DGX) | A100 | 80 | 8 | 128 | 2048 | 1:16:250G | 28 | ampere,nvlink,dgx,80gb |
| cn-e[002-003] (DGX) | V100 | 32 | 8 | 40 | 512 | 1:5:62G | 7 | volta,nvlink,dgx,32gb |
| cn-f[001-004] | CPU | – | – | 32 | 256 | 0:1:7G | 10 | rome |
| cn-h[001-004] | CPU | – | – | 64 | 768 | 0:1:11G | 7 | milan |
| cn-m[001-004] | CPU | – | – | 96 | 1024 | 0:1:10G | 7 | sapphire |

- GRES names include `a100` (40GB), `a100l` (80GB), `l40s`, `rtx8000`, `v100`, `h100`, etc.
  DGX A100 40GB: `--gres=gpu:a100:N --constraint="dgx&ampere"`; 80GB: `--gres=gpu:a100l:N --constraint="dgx&ampere"`.
  DGX A100: 6 NVSwitches, 600 GB/s point-to-point, 4.8 TB/s bisection.
- Features usable with `--constraint`: `32gb/40gb/48gb/80gb`, `volta/turing/ampere/lovelace/hopper`,
  `nvlink`, `dgx`.
- All Mila compute nodes have **unrestricted Internet access** (unlike most DRAC/PAICE clusters).
- "Optimal bundle" = proportional CPU/RAM per GPU; staying within it avoids wasting a node.

---

## 8. Mila storage

| Path | Perf | Use | Quota (space/files) | Backup | Auto-cleanup |
|---|---|---|---|---|---|
| `/network/datasets/` | High | curated raw datasets, read-only | – | – | – |
| `/network/weights/` | High | curated model weights, read-only | – | – | – |
| `$HOME` = `/home/mila/<u>/<username>/` | Low | code, libraries, binaries, small important results | 100 GiB / 1M files | daily | no |
| `$SCRATCH` = `/network/scratch/<u>/<username>/` | High | processed datasets, temp results, checkpoints; optimized for small files | 5 TiB / unlimited | **no** | files unused **90 days** deleted (daily purge; shorter if FS > 90% full) |
| `$SLURM_TMPDIR` | Highest | node-local disk for the job | – | no | **deleted at job end** |
| `/network/projects/<group>/` | Fair | shared long-term project storage (request via IT) | 1 TiB / 1M files per group (can request more) | daily | no |
| `$ARCHIVE` = `/network/archive/<u>/<username>/` | Low | long-term personal storage (logs, samples, paper data) | 5 TB soft / 5.1 TB hard, 7-day grace | **no** | no |

(`<u>` = first letter of username.)
- Check quota: `disk-quota` (HOME/SCRATCH), `df -h $ARCHIVE`.
- `$HOME` restore: IT ticket with path and date.
- `$ARCHIVE` is only mounted on **login and CPU-only nodes**; it's tuned for large files →
  archive directories: `tar cJf $ARCHIVE/exp.tar.xz --xattrs exp`.
- `uv cache prune` / `uv cache clean` free space in `~/.cache/uv` when `$HOME` is full.
- **Datasets**: request additions via the datasets form (https://forms.gle/1VMFrJTwd291HJdt8);
  browse at https://datasets.server.mila.quebec/. `datasets/restricted/` needs a support ticket
  with the access group (e.g. `scannet_users`, find it with `ls /network/datasets/restricted`),
  your username and the group owner's approval. Datasets following DRAC good practices are
  mirrored on DRAC in `~/projects/rrg-bengioy-ad/data/curated/`; list with
  `ssh <cluster> -C "projects/rrg-bengioy-ad/data/curated/list_datasets_cc.sh"`.
- **Weights**: weights form (https://forms.gle/HLeBkJBozjC3jG2D9); `weights/restricted/` same
  procedure as restricted datasets.
- Golden rule: copy data to `$SLURM_TMPDIR` at job start (once per node), work there, copy
  results to `$SCRATCH` (or `$HOME`/projects/archive for long-term) before the job ends.
  Keep crucial data (paper/thesis) in more than one place.

### Sharing data with ACLs (e.g. in `$SCRATCH`)
```bash
setfacl -Rdm user:${USER}:rwx  $SCRATCH/X/Y/Z      # you: future files (default ACL)
setfacl -Rdm user:${USER2}:rwx $SCRATCH/X/Y/Z      # collaborator: future files
setfacl -Rm  user:${USER2}:rwx $SCRATCH/X/Y/Z      # collaborator: existing files (do after the default, avoids race)
setfacl -m user:${USER2}:x $SCRATCH/X/Y ; setfacl -m user:${USER2}:x $SCRATCH/X ; setfacl -m user:${USER2}:x $SCRATCH   # traverse parents (rx to list)
setfacl -x user:${USER2} DIR   # remove one entry ;  setfacl -b DIR  # remove all
getfacl DIR                     # inspect
```
`/`, `/network`, `/network/scratch` are already traversable; `$SCRATCH`, `$HOME` and
subfolders need explicit grants.

### Data transfer
- `rsync` preferred; limit bandwidth: `rsync --bwlimit=10mb ...`.
- Globus: Mila has no license, but you can set up **Globus Connect Personal** on the Mila
  cluster to transfer to/from a Globus endpoint such as DRAC.

### Contributing / sharing datasets publicly
- Add to `/network/datasets` via the form. Public sharing via **Academic Torrent** (Mila seeds;
  put data in `/network/scratch/.transit_datasets`, create the torrent with
  `torrent create --tracker https://academictorrents.com/announce.php NAME` — not on a login
  node — and send the URL to the helpdesk) or **Mila Google Drive** (staff uploads; provide
  archive in `.transit_datasets`, name, license .txt (Creative Commons, preferably with
  Attribution), MD5, arXiv/GitHub URLs, unpack instructions). Download with `gdown` (shared
  quota) or `rclone` with your own client id. Get a **DOI** via figshare ("Metadata record
  only"). Naming: no "dataset" in the name, no capitals/special chars/spaces.
- Alternatives: Hugging Face Hub, GitHub/GitLab for small data.

---

## 9. Software environments

### 9.1 uv (recommended)
- Not a module; install per cluster: `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- Project workflow:
```bash
uv init my-project --python=3.12   # pyproject.toml, .python-version
uv add torch "numpy>=2.0"          # resolves full graph, aborts on conflicts (unlike pip)
uv add --dev pytest ; uv remove torch
uv sync                            # recreate exact env from uv.lock (commit uv.lock!)
uv run python train.py             # syncs then runs; no activation needed
uv python pin 3.11 / uv sync --python 3.11
uv tool install wandb ; uv tool list ; uv tool upgrade --all ; uv tool uninstall wandb
uv add --script exp.py numpy ; uv run --script exp.py   # PEP 723 inline-deps scripts
uvx <tool>                         # run a tool ad hoc (e.g. uvx milalib monitor, uvx tensorboard)
```
- In Slurm: `srun uv run python train.py`. Use `uv run --offline` on clusters without compute-node
  internet (DRAC/PAICE: run `uv sync` on a login node first). On Mila, install GPU frameworks from
  a GPU compute node (`salloc --gres=gpu:1 ...; uv sync`).

### 9.2 Modules / pip / conda
- `module avail`, `module load <m>`, `module spider <term>`. Mila module sections: Core,
  Compiler, Cuda (toolkits, cudnn), Pytorch/Tensorflow builds for Mila GPUs. Nested modules
  auto-load deps (e.g. `cudnn/7.6/cuda/9.0/tensorrt/7.0`). User site-packages take precedence;
  use `python -s` to skip them; check `package.__file__`.
- Classic venv: `module load python/3.10; python -m venv $HOME/env; source $HOME/env/bin/activate; pip install torch`.
- Conda on nodes without GPU fails driver detection → `CONDA_OVERRIDE_CUDA=11.8 conda create ...`.
- Portability principles: record versions, isolate projects, use containers as last resort.
- CPU vs GPU runs diverge numerically (float non-associativity, different reduction order).
  Not a bug; float64 helps on CPU but is very slow on most GPUs; compare aggregate metrics.

### 9.3 Frameworks (minimal examples)
- PyTorch/JAX setups via `pyproject.toml` (e.g. `torch>=2.7.1`, `jax>=0.7`, Python `>=3.11,<3.14`).
- **Flash Attention**: prefer a pre-built wheel matching CUDA/torch/Python (e.g.
  `flash_attn-2.8.3.post1+cu126torch2.7cxx11abiFALSE-cp312-...whl`) referenced in
  `[tool.uv.sources]` (`url=` or `path=`); can be installed on login nodes. Otherwise **build
  from source in a compute job** (not login node) with `MAX_JOBS=4`,
  `FLASH_ATTENTION_SKIP_CUDA_BUILD=0`, `TORCH_CUDA_ARCH_LIST="9.0"` (H100; list several with `;`).
  Find cached wheel: `find ~/.cache/uv -name "flash_attn*.whl"`.

### 9.4 Containers (no Docker on clusters)
- **Podman** (tech preview on Mila, rootless, docker-compatible CLI). Storage is per-job and
  lost at job end → bind-mount: `podman run --mount type=bind,source=$SCRATCH/exp,destination=/data/exp ...`.
  GPUs: `--device nvidia.com/gpu=all` or `--device nvidia.com/gpu=N`. Harmless warnings
  about subuid/shared mount can be ignored.
- **Singularity/Apptainer**: pull from DockerHub (`singularity pull docker://pytorch/pytorch:...`),
  sandbox (`singularity build --sandbox dir img.sif`, `singularity shell --writable`), or
  recipes (need sudo → build on a Linux laptop; on Mila you can build on a CPU compute node
  with `SINGULARITY_CACHEDIR=$SLURM_TMPDIR`). **Never build on login nodes**; don't rebuild per
  job. Store images in `$SCRATCH`, copy to `$SLURM_TMPDIR`, run with
  `singularity exec --nv -H $HOME:/home -B $SLURM_TMPDIR:/dataset -B $SCRATCH:/final_log img.sif python code.py`.
  On DRAC: `module load singularity`. Not supported on macOS.

---

## 10. Development tools & workflows

### 10.1 milatools (`mila`)
- `mila init` (SSH setup, keys), `mila code PATH --alloc <salloc args>` (VSCode/Cursor on a
  compute node; checks `$HOME` quota first), `mila code PATH --job <JOBID>` (attach to existing
  job), `ssh mila-cpu`. Also helps create DRAC SSH configs.

### 10.2 VSCode
- Never on login nodes. Use `mila-cpu` in Remote-SSH dropdown or `mila code`.
- Select interpreter: Ctrl+Shift+P → "Python: Select interpreter" (`which python` for path).
- "Cannot reconnect" → `"remote.SSH.lockfilesInTmp": true`. Debugger timeouts →
  `export DEBUGPY_PROCESS_SPAWN_TIMEOUT=500`. VSCode auto-forwards ports (e.g. TensorBoard 6006).

### 10.3 JupyterHub
- Mila: https://jupyterhub.server.mila.quebec (Google OAuth, @mila.quebec). DRAC has its own.
- Starts JupyterLab as a Slurm job. **Stop it** via Hub → Control Panel → Stop my server
  (closing the tab keeps the job running).
- Access paths outside `$HOME` via symlinks: `ln -s /network/datasets $HOME`,
  `ln -sf $SLURM_TMPDIR $HOME` (each session).

### 10.4 Checkpointing (strongly recommended)
- Save model + optimizer + epoch + best metric + all RNG states (Python/NumPy/torch) each epoch
  to e.g. `$SCRATCH/<exp>/<SLURM_JOB_ID>/checkpoints/checkpoint.pth`; load it at start if present
  (job ID is preserved across requeues).
- `#SBATCH --requeue` (requeue with the same job ID) and `#SBATCH --signal=B:TERM@300`
  (SIGTERM 5 min before the time limit); the job script can log
  `echo "Job has been preempted $SLURM_RESTART_COUNT times."`.
- `scancel --signal=TERM <id>` to stop while allowing requeue; `scontrol requeue <id>` (only for
  running, pending or recently completed jobs). Plain `scancel` → `FAILED`, cannot be requeued.
- Copy valuable checkpoints out of `$SCRATCH` (no backup, purge). Watch disk usage of many checkpoints.
- Pattern from the Mila example (`main.py`):
```python
import os, signal, shutil, uuid, torch
from pathlib import Path
from typing import TypedDict

class RunState(TypedDict):   # everything needed for an exact resume
    epoch: int; best_acc: float
    model_state: dict; optimizer_state: dict
    random_state: tuple; numpy_random_state: dict
    torch_random_state: torch.Tensor; torch_cuda_random_state: list

CKPT_DIR = Path(os.environ["SCRATCH"]) / "checkpointing_example" / os.environ["SLURM_JOB_ID"] / "checkpoints"

def signal_handler(signum, frame):
    print(f"Job received a {signal.Signals(signum).name} signal!")
    # keep it quick; don't checkpoint mid-epoch here. With wandb: wandb.mark_preempting()
signal.signal(signal.SIGTERM, signal_handler)

def load_checkpoint(ckpt_dir, **kw):
    restarts = int(os.environ.get("SLURM_RESTART_COUNT", 0))
    f = ckpt_dir / "checkpoint.pth"
    if not f.exists():
        if restarts:   # requeued but nothing saved → checkpointing code isn't running before preemption
            print(f"Warning: restarted {restarts} times but no checkpoint found")
        return None
    kw.setdefault("weights_only", False)
    return torch.load(f, **kw)

def save_checkpoint(ckpt_dir, is_best, state: RunState):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    f = ckpt_dir / "checkpoint.pth"
    tmp = f.with_name(f"{f.name}.temp{uuid.uuid1()}")
    torch.save(state, tmp)
    os.replace(tmp, f)                     # atomic: never leaves a half-written checkpoint
    if is_best:
        shutil.copyfile(f, f.with_name("model_best.pth"))

# at start: state = load_checkpoint(CKPT_DIR, map_location=device)
# if state: restore model/optimizer/RNG states, start_epoch = state["epoch"] + 1
# end of each epoch: save_checkpoint(CKPT_DIR, val_acc > best_acc, RunState(...))
```
  At most one epoch of progress is lost on interruption.

### 10.5 Launching many jobs
- Pass args through: `srun uv run python main.py "$@"` then `sbatch job.sh --learning-rate 0.1`.
- **Job arrays**: `sbatch --array=1-5 job.sh` → `$SLURM_ARRAY_TASK_ID`.
- **Many tasks per GPU** (packing): `--gres=gpu:1 --ntasks-per-gpu=N` + `srun`, use
  `$SLURM_PROCID`; combine: `sbatch --array=1-10 --ntasks-per-gpu=5 --gpus=1 --cpus-per-task=1 job.sh`.
- Packing with `srun --multi-prog conf` (`0 python s.py a`, `1 python s.py b`, or `0-1 python s.py %t`),
  or several `srun --gres=gpu:1 --ntasks=1 --exclusive ... &` + `wait` in a whole-node job.
- Keep job counts reasonable (low hundreds), each job ≥ several minutes.
- **HPO with Oríon** (Mila's framework): job array + `orion hunt`, pickleddb database on
  `$SCRATCH`. Others: wandb sweeps, Hydra, Ray Tune. Job-script part of the Mila example:
```bash
export ORION_CONFIG=$SLURM_TMPDIR/orion-config.yml
cat > $ORION_CONFIG <<- EOM
    experiment:
        name: orion-example
        algorithms:
            tpe:
                seed: null
                n_initial_points: 5
        max_broken: 10
        max_trials: 10
    storage:
        database:
            host: $SCRATCH/orion.pkl
            type: pickleddb
EOM
srun uv run orion hunt --config $ORION_CONFIG python main.py --learning-rate~'loguniform(1e-5, 1.0)'
```
  In `main.py`: `from orion.client import report_objective; report_objective(1 - val_accuracy)`
  (Oríon minimizes). Add `orion` to dependencies. Launch: `sbatch --array=1-10 --ntasks-per-gpu=5 --gpus=1 --cpus-per-task=1 job.sh`
  (10 jobs × 5 trials sharing each GPU). Inspect: `uv run orion info -n orion-example`,
  `uv run orion plot regret -n orion-example`. Move the Oríon DB out of scratch when done.
  Other search-space syntax: `"lr": "orion~loguniform(1e-5, 1.0)"`, `"epochs": "orion~fidelity(1, 100, base=2)"` (hyperband).

### 10.6 Multi-task / distributed training
- Env vars per task: `SLURM_PROCID` (rank), `SLURM_LOCALID` (local rank), `SLURM_NTASKS`
  (world size), `SLURM_NODEID`; job-wide: `SLURM_JOB_NODELIST`, `SLURM_JOB_NUM_NODES`, `SLURM_TMPDIR`.
- Rendezvous in job script:
```bash
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)  # "127.0.0.1" if single node
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$SLURM_NTASKS
srun --gres-flags=allow-task-sharing uv run python main.py   # 1 task per GPU
```
- **`--gres-flags=allow-task-sharing` is required** with `--gpus-per-task`: otherwise cgroups
  isolate each task's GPU and NCCL fails with `ncclUnhandledCudaError: Call to CUDA function failed`
  (Slurm bug https://support.schedmd.com/show_bug.cgi?id=17875).
- **Multi-GPU (single node) example**: `--ntasks=2 --ntasks-per-node=2 --cpus-per-task=4 --gpus-per-task=l40s:1 --mem-per-gpu=16G`,
  `MASTER_ADDR="127.0.0.1"`. Code changes vs single GPU:
```python
import torch.distributed as dist
from torch.distributed import ReduceOp
from torch.utils.data.distributed import DistributedSampler
rank, world_size = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"])
os.environ["RANK"], os.environ["WORLD_SIZE"] = str(rank), str(world_size)
dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
is_master = rank == 0
device = torch.device("cuda", 0)          # Slurm sets CUDA_VISIBLE_DEVICES per task → each sees 1 GPU
model = torch.nn.parallel.DistributedDataParallel(model.to(device), device_ids=[0], output_device=0)
train_sampler = DistributedSampler(train_ds, shuffle=True)   # also for val/test with shuffle=False
train_loader = DataLoader(train_ds, batch_size=bs, sampler=train_sampler, shuffle=False, num_workers=nw)
for epoch in range(epochs):
    train_sampler.set_epoch(epoch)        # required, else same shuffle order every epoch
    ...
    dist.reduce(n_correct, dst=0, op=ReduceOp.SUM)   # aggregate metrics on rank 0 (all_reduce in validation)
```
  Datasets: only the master downloads; others wait on `dist.barrier()`, then the master joins the
  barrier and everyone reads from disk.
- **Multi-node example**: `--ntasks=4 --ntasks-per-node=2` (2 nodes), stage data with one task per
  node (`srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -c 'mkdir -p $SLURM_TMPDIR/data && cp ... $SLURM_TMPDIR/data/'`),
  `MASTER_ADDR` = first node hostname. Use **TCP init** (the `env://` method uses file locks that
  can deadlock on the cluster's distributed filesystem) and a short timeout:
  `dist.init_process_group(backend="nccl", init_method=f"tcp://{MASTER_ADDR}:{MASTER_PORT}", timeout=timedelta(seconds=60), world_size=world_size, rank=rank)`
  (default timeout is 30 min). Use `local_rank = int(os.environ["SLURM_LOCALID"])`; the
  `local_rank == 0` process of each node downloads/prepares data for that node.
- torchrun alternative: one task per node, `srun torchrun --nproc_per_node=$SLURM_GPUS_PER_NODE --nnodes=$SLURM_JOB_NUM_NODES --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$RDV_ADDR train.py`, `--wait-all-nodes=1`.
- Only rank 0 saves checkpoints; barrier-sync weights. Data parallelism is easiest; model
  parallelism needs tightly-connected GPUs (NVLink/InfiniBand).
- Advanced ImageNet example (multi-node, `safe_sbatch` submitting at a git commit for
  reproducibility, `prepare_data.py` once per node, `--partition=short-unkillable`).

### 10.7 cluv (multi-cluster submission)
- `uv tool install cluster-uv` (or `uv add cluster-uv`); needs Python ≥ 3.11, a GitHub-hosted
  project, working SSH with ControlMaster (WSL2 on Windows).
- `cluv init` (adds `[tool.cluv]` to pyproject, `scripts/` templates, `logs` symlink),
  `cluv login [cluster]`, `cluv sync [cluster]` (push commits, clone/checkout + `uv sync` on
  each cluster, rsync results back from `results_path`; optional dataset sync with
  `data_source`/`datasets_path`, skip with `--no-sync-datasets`),
  `cluv submit <cluster> scripts/job.sh --time=00:10:00 -- python main.py --lr 0.01`,
  `cluv submit first ...` (submit everywhere, keep the first to start, cancel others),
  several `sbatch_args` accounts per cluster (e.g. `rrg-bengioy-ad` and `def-bengioy`) race the
  same way, `cluv status`.

### 10.8 Other
- **Git LFS**: download `git-lfs-linux-amd64-v<VER>.tar.gz` from GitHub releases, extract, add
  to `PATH` in `~/.bashrc`, `git lfs version`.
- **Research Project Template**: Hydra + PyTorch Lightning/JAX + WandB + pytest.
- Reproducibility: pin env (uv.lock), Git with tagged commits, share datasets, release code
  with papers.

---

## 11. Monitoring, experiment tracking and compute efficiency

### 11.1 Experiment tracking
- **WandB (Mila org)**: request access at it-support@mila.quebec (students of Core profs). Sign
  in at wandb.ai with your @mila.quebec email → SSO (mila.quebec IdP). Existing account: first
  add @mila.quebec as **primary email**, log out, then SSO (else duplicate account). New account:
  choose **Professional** (Mila pays).
  - On cluster: `uv tool install --upgrade wandb; wandb login` (key saved in `~/.netrc`, once per cluster).
  - `wandb.init(project=..., name=os.environ.get("SLURM_JOB_ID"), id=os.environ.get("SLURM_JOB_ID"), resume="allow", group=os.environ.get("SLURM_ARRAY_JOB_ID"), tags=[...], config=vars(args) | {f"env/{k}": v for k, v in os.environ.items() if k.startswith("SLURM")})`;
    `wandb.log({"train/loss": ...})`; prefixes `train/`, `val/`, `perf/`; call `wandb.finish()`.
  - No internet on compute nodes: `export WANDB_MODE=offline` (or `wandb off`) then
    `wandb sync --sync-all` from a login node. On DRAC, `module load httpproxy` is an alternative
    but doesn't support artifacts and can hang at the end → offline mode recommended.
  - WandB System tab measures whole-node CPU/RAM (inaccurate) and only the main process unless
    `settings=wandb.Settings(x_stats_track_process_tree=True)`.
- **Comet (Mila org)**: request via it-support; log in at https://comet.mila.quebec/ (account linked if it uses @mila.quebec).
- **TensorBoard + PyTorch profiler**: `torch.profiler.profile(schedule=..., on_trace_ready=torch.profiler.tensorboard_trace_handler(logdir))`,
  logs to `$SCRATCH/logs/$SLURM_JOB_ID`; run `uvx tensorboard --logdir ...` **on a compute node**,
  tunnel `ssh -L 6006:localhost:6006 <node>.server.mila.quebec`, open http://127.0.0.1:6006.

### 11.2 Resource monitoring tools
| Question | Tool |
|---|---|
| Job state, memory/time used | `squeue`, `sacct` (compare `MaxRSS` vs `ReqMem`) |
| Efficiency of past jobs | Compute Utilization Dashboard (sarc.mila.quebec) |
| GPU busy now? I/O-bound? | `uvx milalib monitor -i 1 -m gpu_util -m mem_util -m power -m sm_occupancy`, `nvidia-smi`, WandB System tab, NVSMI log at end of `slurm-<id>.out` |
| Hot spots in training loop | PyTorch profiler + TensorBoard |
| Node state | Netdata at `<node>.server.mila.quebec:19999` (Mila wifi or `ssh -L 19999:<node>.server.mila.quebec:19999 mila`), Grafana dashboard |
- milalib in WandB: `from milalib.wandb import monitor; with monitor(interval=5): train()`.
- DRAC portals: Nibi https://portal.nibi.sharcnet.ca/, Rorqual https://metrix.rorqual.calculquebec.ca/,
  Trillium https://my.scinet.utoronto.ca/, TamIA https://portail.tamia.ecpia.ca/,
  Vulcan https://portal.vulcan.alliancecan.ca/.

### 11.3 Compute utilization policy ("Safety net")
- Researchers must request the right amount of compute and use it efficiently. IDT monitors
  **hardware telemetry only** (never code/private data).
- Tools: dashboard + **monthly Slack reports**; **waste notification** if your **SM occupancy
  < 15% for 2 consecutive weeks**; 1-on-1 IDT coaching. Co-designed with the Mila Research
  Compute Committee.
- **RGU** (Reference GPU Unit, DRAC metric): A100-40GB = 4 RGU, H100-80GB = 12.15 RGU.
  Compute = RGU × #GPUs × time (RGU·week). Cost ≈ $1100 per RGU·year → ≈ $21/RGU·week,
  $0.125/RGU·hour.
- Metrics: `gpu_utilization` (fraction of time any kernel runs — can be 100% while wasteful) vs
  **SM occupancy** (fraction of warps active vs max; used by the dashboard).
  SMO ≥ 50% excellent · 30–50% good · 15–30% good with room · 5–15% poor · < 5% critical.
- Dashboard: filter bar (dates, period, cluster, job states; "Update"), "RGU allocated vs
  efficiency levels" (Used / Acceptable Idle / Critical Waste <15% / No data), "RGU by cluster",
  job table; click a bar to focus.
- Diagnosis flow: SMO ≥ 30% → fine; else VRAM < 20% → bigger batch, smaller GPU, or pack jobs;
  else check data loader (I/O or CPU preprocessing bottleneck; `num_workers` 2–4 per GPU,
  `pin_memory=True`, stage data to `$SLURM_TMPDIR`).
- Do: profile before scaling, checkpoint, right-size (RTX8000/V100 or MIG slices for small work),
  request minimal blocks. Don't: hoard GPUs idle in interactive sessions, avoid preemptible
  partitions, over-request CPUs (e.g. 40 CPUs for 1 GPU), add GPUs to fix I/O bottlenecks.
- Never `torch.autograd.set_detect_anomaly(True)` in normal jobs: it stats every source file per
  tensor → extreme IOPS on `$HOME`; you will be contacted.

### 11.4 Environmental impact
Mila cluster ~115–150 kW, Quebec hydro at 0.6 kg CO2/MWh → < 1 t CO2/year (~$20/year offset).
Hardware manufacturing impact not estimated.

---

## 12. DRAC — Digital Research Alliance of Canada clusters

### 12.1 Overview
- National clusters shared across Canada, for larger experiments (many jobs, multi-node,
  multi-GPU). Access via **allocations** (resources a group can target over ~1 year). An
  allocation is a **scheduler weighting (fair-share), not a cap** on simultaneous usage.
- Every Canadian professor has a default allocation (`def-<prof>-gpu`, `def-<prof>-cpu`);
  managing it is the professor's responsibility (Mila can't support it). Core-professor
  students should get the **Mila global allocation `rrg-bengioy-ad`**. Ask your supervisor
  which allocations apply to you.

### 12.2 Account creation and access
1. Create an account at https://ccdb.alliancecan.ca (use your **@mila.quebec** email).
2. Apply for a **role** at https://ccdb.alliancecan.ca/me/add_role with your sponsor's
   **CCRI** (Compute Canada Role Identifier); wait for sponsor approval. You can hold roles
   under several sponsors/allocations (e.g. your supervisor's `def-<prof>` and the Mila global
   allocation `rrg-bengioy-ad`).
3. Request access to all systems in the **HPC** tab under *Resources > Access Systems*
   (and *Artificial Intelligence* tab for PAICE).
4. Mila global allocation: email **ccdb-accounts@mila.quebec** with **your** CCRI.
5. SSH: add your public key in CCDB *My Account > SSH Keys* (`mila init` can help write
   the SSH config). Set up **MFA** in *My Account > Multifactor Authentication* (DRAC MFA is
   separate from Mila's).
6. **Annual renewal** required for all accounts (sponsor and sponsored); see DRAC renewal FAQ.

### 12.3 Clusters in the Mila global allocation (`rrg-bengioy-ad`, April 7, 2026 → Spring 2027)
| Cluster | CPUs | RGUs | ≈ GPU equiv | GPU | Compute-node internet |
|---|---|---|---|---|---|
| **Fir** (successor of Cedar, keeps its FS) | 0 | 2090 | 171 | H100-80G | Yes |
| **Nibi** (successor of Graham, keeps its FS) | 0 | 363 | 30 | H100-80G | Yes |
| **Rorqual** (successor of Béluga) | 263 | 1172 | 96 | H100-80G | No |
| **Trillium** (successor of Niagara; mostly CPU) | 768 | 375 | 31 | H100-80G | No |

- **Trillium** specifics: two sub-clusters sharing a filesystem — CPU (`trillium.alliancecan.ca`,
  whole-node allocation only) and GPU (`trillium-gpu.alliancecan.ca`, whole node or single GPU =
  1/4 node). **Jobs must be submitted from `$SCRATCH`.** Read the Trillium Quickstart.
- **Narval** (not in the global allocation; may be accessible via your supervisor): oldest,
  A100-40GB GPUs, no compute-node internet; fine for jobs that can't use a full H100.
- Historical: the older mega-allocation was on Narval, Béluga and Cedar.

### 12.4 Running jobs on DRAC
```bash
sbatch --time=1:00:00 --account=rrg-bengioy-ad job.sh                 # CPU
sbatch --time=1:00:00 --account=rrg-bengioy-ad --gres=gpu:1 job.sh    # GPU
salloc --time=1:00:00 --account=rrg-bengioy-ad --gres=gpu:1           # interactive
```
(Account differs with affiliation, e.g. `def-<prof>`.) Short interactive jobs usually get
quicker access (good for testing). Full docs: https://docs.alliancecan.ca/wiki/Running_jobs.

Example (Narval-style, venv built inside the job):
```bash
#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=124G
#SBATCH --time=03:00:00
#SBATCH -o /scratch/<user>/slurm-%j.out
module load StdEnv/2023 python/3.12
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index torch torchvision          # DRAC pre-built wheels (avail_wheels <name>)
unzip $SCRATCH/DATASET.zip -d $SLURM_TMPDIR       # or tar -xf ... -C $SLURM_TMPDIR
python main.py --path $SLURM_TMPDIR --data_path $SLURM_TMPDIR
cp $SLURM_TMPDIR/RESULTS $SCRATCH
```
- Modules: `module avail / load / spider`; `module load python/3.12`; see DRAC Python, PyTorch,
  TensorFlow wiki pages. uv also works (install it yourself; `uv sync` on login node,
  `uv run --offline` in jobs without internet).

### 12.5 DRAC storage
| Storage | Path | Use |
|---|---|---|
| `$HOME` | `/home/<user>/` | code, libraries (Lustre); **no experiment results** |
| `$HOME/projects` | `/project/rrg-bengioy-ad` | **compressed raw** datasets only (size & file-count limits); make new datasets group-readable: `chgrp -R rpp-bengioy <dataset>` |
| `$SCRATCH` | `/scratch/<user>` | processed datasets, results, logs; large quota; **files unused for 3 months purged** (email warning first) |
| `$SLURM_TMPDIR` | node-local | copy data in at job start, intermediate checkpoints; cleared after job |
Transfer results back to Mila when experiments are done. Curated Mila datasets mirror:
`~/projects/rrg-bengioy-ad/data/curated/`.

### 12.6 Internet / tracking on DRAC
- Most compute nodes lack internet: use WandB **offline** + `wandb sync`, or `module load httpproxy`
  (whitelists WandB/Comet servers; no artifacts, may hang at the end → offline preferred).
- Globus available for transfers.

### 12.7 DRAC FAQ
- `ImportError: /lib64/libm.so.6: version GLIBC_2.23 not found` → non-standard FS layout; see
  DRAC "Installing binary packages".
- "Disk quota exceeded" on `/project` → wrong group permissions on files (DRAC FAQ; may be obsolete).

---

## 13. PAICE — Pan-Canadian AI Compute Environment

- DRAC-hosted clusters dedicated to AI research, reserved for **CIFAR AI Chairs**' groups.
  Use **AIP allocations**: `--account=aip-${PI_NAME}` (your supervising professor). Regular DRAC
  and Mila global allocations do **not** work.
- Accounts on CCDB (same steps as DRAC; request systems in the *Artificial Intelligence* tab);
  your professor must add you to their AIP allocation (share your CCRI). Same SSH/MFA/renewal
  process as DRAC.
- Tiered access: by default tier 1 + tier 2 get 85% of each cluster, tier 3 gets 10% (ask your
  supervisor which tier applies).

| Cluster (Mila default tier) | CPUs | RGUs | ≈ GPUs | GPU | Internet | Notes |
|---|---|---|---|---|---|---|
| **TamIA** (tier1+tier2) | 435 | 1738 | 143 | H100-80G, H200 | No | Mila + Calcul Québec, Université Laval; **per-node allocation** |
| **Killarney** (tier3) | 0 | 794 | 75 | H100-80G, L40S | Yes | Vector + SciNet, U. Toronto |
| **Vulcan** (tier3) | 0 | 850 | 82 | L40S | No | U. Alberta + Amii |
(Period April 7, 2026 → Spring 2027.) Storage like DRAC (`/project/<project>`); same
`httpproxy`/offline WandB advice. Slack: `#tamia-cluster`. (The cheat-sheet errata only says that
the cluster that was going to be called "STACC" will be called "Lynx". The docs give no other
details about it.)

### Mila vs DRAC/PAICE
Same Slurm + similar filesystems, but Mila has more control and **unrestricted internet on all
compute nodes**. Advice: use a healthy mix of Mila and DRAC/PAICE clusters (DRAC often has more
free capacity); tools like milatools and cluv make switching easy.

---

## 14. AI assistants for Mila questions

- **Condensed docs file** (`all_docs_condensed.md`, ~1000 lines) to upload into Claude/ChatGPT/
  Gemini with a context message; include "Mila"/"cluster" in questions; verify flags with
  `man`/`--help`, limits with `sinfo`, storage policies in the docs.
- **Mila Docs MCP server** (Algolia DocSearch, no auth):
  `claude mcp add --transport http mila-docs https://558773LESW.algolia.net/mcp/1/3_N8rKyLQhaxUEpt0FMx6A/mcp`
  (or add `{"mcpServers":{"mila-docs":{"type":"http","url":"..."}}}` to `~/.claude.json`;
  Cursor: `~/.cursor/mcp.json` with `"url"`).
- **Claude Code**: `curl -fsSL https://claude.ai/install.sh | bash` (Pro/Team/Enterprise
  subscription; WSL2 on Windows), or VS Code extension. Mila skills:
  `/plugin marketplace add mila-iqia/skills` then `/plugin install mila-tools@mila-skills`,
  `/reload-plugins`. Skills: `/mila-account-setup`, `/mila-local-setup`,
  `/mila-connect-cluster`, `/mila-run-jobs`.
- The docs site has an "Ask AI" chatbot (bottom-right).

---

## 15. FAQ (Mila cluster)

- **connection refused on login** → IP banned after too many failures; wait 1 h, else ticket.
- **Change shell** / **reset cluster password** / **invalid account or account/partition
  combination** → support ticket. Note: cluster password = Google Workspace password; a reset
  changes it for all Google services.
- **Interactive shell** → `salloc [options]` (bound to the login node it was started from).
- **`srun: error: --mem and --mem-per-cpu are mutually exclusive`** → ignore (salloc default mem).
- **Where are my jobs?** → `squeue -u $USER`, `scontrol show job <id>`.
- **Cancel** → `scancel <id>`, `scancel -u $USER`, `scancel -t PD`.
- **SSH into my job's node / `Permission denied (publickey)`** → key in `~/.ssh/authorized_keys`
  (e.g. `ssh-keygen; cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys`, fix permissions).
- **Where to put data during a job** → `$SLURM_TMPDIR`; save outputs to shared FS before the job
  ends (especially if preemptible).
- **`oom-kill event(s)`** → increase `--mem`/`--mem-per-cpu` or fix a leak.
- **`fork: retry: Resource temporarily unavailable`** → >2000 processes in the job.
- **PyTorch 1.10 `INTERNAL ASSERT FAILED ... MapAllocator.cpp`** → upgrade to ≥1.11 (or LD_PRELOAD hack).
- **MIG: `torch.cuda.device_count()==0`** → PyTorch ≥ 2.0 (or `unset CUDA_VISIBLE_DEVICES` on 1.13).
- **Told my job abuses IOPS** → disable `set_detect_anomaly`.
- **Conda "CUDA driver not available"** → `CONDA_OVERRIDE_CUDA=<ver>`.
- **Job pending for long** → normal priority queueing; request less time/resources, try another
  GPU type/cluster.
- **Job restarted from scratch** → preemption; add checkpointing.

---

## 16. General theory (short)
- Cluster = login nodes (entry), compute nodes (heterogeneous, GPU classes), storage nodes; never
  overload login nodes (may lead to administrative action).
- Workload manager queues jobs; order depends on priority/fair-share, wait time, size.
  Partitions group nodes with policies. Soft limits enforced by **preemption** (with or without
  grace); hard limits by refusing jobs; exceeding requested RAM kills the job, GPUs beyond the
  request are invisible.
- Filesystems differ by performance/retention/backup; using the wrong one can make a job
  hundreds of times slower.

---

## 17. Glossary

- **cluster**: computers working together that run your jobs. **login node**: where you connect
  and submit jobs (no heavy work). **compute node**: runs your jobs, where GPUs are.
- **GPU**: hardware that speeds up training. **VRAM**: GPU memory. **SM occupancy**: fraction of
  the GPU's compute units in use. **MIG**: Multi-Instance GPU (a GPU split into isolated slices).
- **Slurm**: the job scheduler. **job**: a unit of work with resources and a time limit.
  **job step**: a command run inside a job (one `srun`). **task**: one copy of a step's command
  (one process, with a rank). **batch job**: submitted with `sbatch`, runs without you connected.
  **interactive job**: `salloc`/`mila code`, gives you a shell. **partition**: group of nodes with
  limits/priority. **preemption**: killing a lower-priority job to free resources. **job array**:
  many copies of one job (`--array`).
- **MFA**: multi-factor authentication. **OTP**: one-time password. **TOTP**: time-based OTP
  (6-digit code from an app). **SSH**: Secure SHell. **WSL**: Windows Subsystem for Linux.
  **ControlMaster**: a reusable SSH connection (fewer MFA prompts).
- **DRAC / the Alliance**: Digital Research Alliance of Canada. **PAICE**: Pan-Canadian AI
  Compute Environment. **CCDB**: DRAC account portal. **CCRI**: Compute Canada Role Identifier.
  **RGU**: Reference GPU Unit. **AIP**: Artificial Intelligence Project (PAICE allocation).
  **IDT**: Mila's Innovation, Development and Technology team.
- **DDP**: PyTorch DistributedDataParallel. **HPO**: hyperparameter optimization.
  **ACL**: Access Control List. **uv.lock**: uv's exact-version lockfile.

---

## 18. How to get help (escalation order)
1. Docs + FAQ (and the site's "Ask AI" chatbot / an LLM with the condensed docs).
2. Slack `#mila-cluster` (also `#compute-canada` for DRAC, `#tamia-cluster`).
3. IT support portal https://it-support.mila.quebec / helpdesk portal 5 (accounts, SSH, access,
   quotas, restricted datasets, project dirs, restores).
4. IDT Office Hours — Lab A, Tue 3–5pm & Wed 2–4pm (also online) — job debugging and optimization.
5. DRAC-specific account/allocation issues: your supervisor, ccdb-accounts@mila.quebec (global
   allocation), support@tech.alliancecan.ca.
