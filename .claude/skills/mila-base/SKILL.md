---
type: shared-policy
version: 1.0.0
---

# Mila Base: Shared Policies

This file is loaded by other mila-* skills at the start of each response.
Do not invoke it directly.

## Command execution

Whenever a terminal command is presented to the user, display it in a code
block as usual. If the command can be run directly in the current shell
(without SSH-ing into another machine), also ask:

> Would you like me to run this command for you?

If the user says yes, execute it using the Bash tool. If the user says no
or does not respond, continue guiding them to run it themselves.

Do NOT offer to run commands that require an active SSH connection to another
machine (e.g., commands shown as running on the cluster, inside the VSCode
remote terminal, or via `srun`/`sbatch`).
