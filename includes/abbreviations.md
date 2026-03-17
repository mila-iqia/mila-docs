<!-- Cluster and infrastructure -->
*[cluster]:         A group of computers that work together; you connect to run jobs on them.
*[login node]:      The machine you first connect to on the cluster; used to prepare and submit jobs.
*[compute node]:    A machine on the cluster that runs your jobs; you get one when you request resources (e.g. GPUs).
*[GPU]:             Graphics Processing Unit; hardware that speeds up heavy computation, used for training models.
<!-- Jobs and scheduling -->
*[Slurm]:           The job scheduler on the cluster; it assigns your jobs to compute nodes.
*[Slurm job]:       A unit of work you submit to Slurm; it gets resources (e.g. a compute node) and runs until it finishes or hits its time limit.
*[Slurm job step]:  A single command or program run inside a Slurm job; one job can run one or more steps in sequence.
*[program]:         The executable or script that runs inside a job step (e.g. `uv run python main.py` or your training script).
*[batch Slurm job]: A Slurm job that runs on the cluster without you staying connected; you submit it and check results later.
<!-- Access and environment -->
*[MFA]:             Multi-Factor Authentication; signing in with more than one check (e.g. password plus a code).
*[SSH]:             Secure SHell; a secure way to connect from your computer to a remote machine.
*[TOTP]:            Time-based One-Time Password; a short-lived code from an app (e.g. Google Authenticator).
*[WSL]:             Windows Subsystem for Linux; lets you run a Linux environment on Windows so you can use the same commands as in this guide.
