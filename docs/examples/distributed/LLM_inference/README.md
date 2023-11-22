# FastAPI + HuggingFace + SLURM

Proof-of-concept for an API that performs inference with a Large Language Model (LLM) on the Mila cluster.

![LLM_api](https://user-images.githubusercontent.com/13387299/184188304-3ce82a7f-29a6-49ed-86ba-4842db4e207e.png)

## The goal:

- One ML researcher/student can submit this as a job on a SLURM cluster, and other users can use a single shared model instance via HTTP or a simple python client.

## Installation:

To run the server locally:

```console
> conda env create -n llm python=3.10
> conda activate llm
> pip install git+https://www.github.com/lebrice/LLM_api.git
```

(WIP) To connect to a running LLM server:

(Requires python >= 3.7)
```console
> pip install git+https://www.github.com/lebrice/LLM_api.git
```


## Usage:

Available options:
```console
$ python app/server.py --help
usage: server.py [-h] [--model str] [--hf_cache_dir Path] [--port int]
                 [--reload bool] [--offload_folder Path] [--use_public_ip bool]

 API for querying a large language model.

options:
  -h, --help            show this help message and exit

Settings ['settings']:
  Configuration settings for the API.

  --model str           HuggingFace model to use. Examples: facebook/opt-13b,
                        facebook/opt-30b, facebook/opt-66b, bigscience/bloom,
                        etc. (default: facebook/opt-13b)
  --hf_cache_dir Path   (default: $SCRATCH/cache/huggingface)
  --port int            The port to run the server on. (default: 12345)
  --reload bool         Whether to restart the server (and reload the model) when
                        the source code changes. (default: False)
  --offload_folder Path
                        Folder where the model weights will be offloaded if the
                        entire model doesn't fit in memory. (default:
                        $SLURM_TMPDIR)
  --use_public_ip bool  Set to True to make the server available on the node's
                        public IP, rather than localhost. Setting this to False
                        is useful when using VSCode to debug the server, since
                        the port forwarding is done automatically for you.
                        Setting this to True makes it so many users on the
                        cluster can share the same server. However, at the
                        moment, you would still need to do the port forwarding
                        setup yourself, if you want to access the server from
                        outside the cluster. (default: False)
```

Spinning up the server:
```console
> python app/server.py
HF_HOME='/home/mila/n/normandf/scratch/cache/huggingface'
TRANSFORMERS_CACHE='/home/mila/n/normandf/scratch/cache/huggingface/transformers'
Running the server with the following settings: {"model_capacity": "13b", "hf_cache_dir": "~/scratch/cache/huggingface", "port": 12345, "reload": false, "offload_folder": "/Tmp/slurm.1968686.0"}
INFO:     Started server process [25042]
INFO:     Waiting for application startup.
Writing address_string='cn-b003:8000' to server.txt
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:12345 (Press CTRL+C to quit)
```

(WIP) Run as a slurm job:

```console
> sbatch run_server.sh
```

(WIP) Using the python client to Connect to a running server:

```python
import time
from app.client import server_is_up, get_completion_text
while not server_is_up():
    print("Waiting for the server to be online...")
    time.sleep(10)
print("server is up!")
rest_of_story = get_completion_text("Once upon a time, there lived a great wizard.")
```
