---
type: shared-policy
version: 1.0.0
---

# Mila Base: Shared Policies

This skill is loaded by other mila-* skills via the Skill tool. Apply its
policies silently (no preamble to the user) when invoked this way.
Do not invoke it in response to a direct user request.

## Command execution

Whenever a terminal command is presented to the user, display it in a code
block as usual. If the command can be run directly in the current shell —
without SSH-ing into another machine **and** without pausing for interactive
input — also ask:

> Would you like me to run this command for you?

If the user says yes, execute it using the Bash tool. If the user says no
or does not respond, continue guiding them to run it themselves.

Do NOT offer to run:
- Commands that require an active SSH connection to another machine (e.g.,
  commands shown as running on the cluster, inside the VSCode remote
  terminal, or via `srun`/`sbatch`).
- Interactive commands that pause for user input (e.g., `mila init`,
  `wsl --install`, or any command that opens a wizard or prompt). These
  cannot run non-interactively via the Bash tool.
- If it is unclear whether a command is interactive, do not offer to run it.
