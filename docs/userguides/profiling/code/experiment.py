import os
from pathlib import Path
import torch
# Import Pytorch profiler
import torch.profiler


# Linear regression training example
x = torch.arange(-5, 5, 0.1).view(-1, 1)
y = -5 * x + 0.1 * torch.randn(x.size())

model = torch.nn.Linear(1, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

# Define in which folder we want the results to be stored
SCRATCH = Path(os.environ.get("SCRATCH", "fake_scratch"))
SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", "0")
logs_dir = SCRATCH / "logs" / SLURM_JOB_ID
logs_dir.mkdir(parents=True, exists_ok=True)


profiler = torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(logs_dir),
    record_shapes=True,
    with_stack=True,
)

# Start the profiler
profiler.start()

# While the model is training
def train_model(iter):
    for epoch in range(iter):
        y1 = model(x)
        loss = criterion(y1, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Write the metrics while training the model
        profiler.step()

# Train the model
train_model(10)

# Stop the profiler when you do not need it anymore
profiler.stop()
