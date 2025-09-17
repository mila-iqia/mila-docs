# Debugging & Profiling

## Multi-node debugging with VSCode

```bash
srun uv run debugpy --listen 0.0.0.0:21404 --wait-for-client main.py
```
