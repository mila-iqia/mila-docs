""" API for querying a large language model. """
from __future__ import annotations

import functools
import logging
import os
import socket
from dataclasses import asdict, dataclass
from logging import getLogger as get_logger
from pathlib import Path

import torch
import uvicorn
from fastapi import Depends, FastAPI, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseSettings
from simple_parsing import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, BloomForCausalLM
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from transformers.models.opt.modeling_opt import OPTForCausalLM

# TODO: Setup logging correctly with FastAPI.
logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)

# SCRATCH = Path(os.environ["SCRATCH"])
SCRATCH = Path("/data/fake_scratch")
# SLURM_TMPDIR = Path(os.environ.get("SLURM_TMPDIR", f"/Tmp/slurm.{os.environ['SLURM_JOB_ID']}.0"))
SLURM_TMPDIR = None


@dataclass(init=False)
class Settings(BaseSettings):
    """Configuration settings for the API."""

    model: str = "meta-llama/Llama-2-7b-chat-hf"
    """ HuggingFace model to use.
    Examples: facebook/opt-13b, facebook/opt-30b, facebook/opt-66b, bigscience/bloom, etc.
    """

    hf_cache_dir: Path = SCRATCH / "cache" / "huggingface"

    port: int = 12345
    """ The port to run the server on."""

    reload: bool = False
    """ Whether to restart the server (and reload the model) when the source code changes. """

    offload_folder: Path = Path(SLURM_TMPDIR or "model_offload")
    """
    Folder where the model weights will be offloaded if the entire model doesn't fit in memory.
    """

    use_public_ip: bool = False
    """ Set to True to make the server available on the node's public IP, rather than localhost.

    Setting this to False is useful when using VSCode to debug the server, since the port
    forwarding is done automatically for you.

    Setting this to True makes it so many users on the cluster can share the same server. However,
    at the moment, you would still need to do the port forwarding setup yourself, if you want to
    access the server from outside the cluster.
    """


def write_server_address_to_file(port: int = 12345):
    node_hostname = socket.gethostname()
    with open("server.txt", "w") as f:
        address_string = f"{node_hostname}:{port}"
        print(f"Writing {address_string=} to server.txt")
        f.write(address_string)


app = FastAPI(
    on_startup=[
        write_server_address_to_file,
    ],
    title="SLURM + FastAPI + HuggingFace",
    dependencies=[],
)


@functools.cache
def get_settings() -> Settings:
    # Creates a Settings object from the environment variables.
    return Settings()


@app.get("/")
def root(request: Request):
    return RedirectResponse(url=f"{request.base_url}docs")


@dataclass
class CompletionResponse:
    prompt: str
    response: str
    model: str


def preload_components(settings: Settings = Depends(get_settings)):
    print(f"Preloading components: {settings=}")
    load_completion_model(capacity=settings.model, offload_folder=settings.offload_folder)
    load_tokenizer(capacity=settings.model)


@app.get("/complete/")
async def get_completion(
    prompt: str,
    max_response_length: int = 30,
    settings: Settings = Depends(get_settings),
) -> CompletionResponse:
    """Returns the completion of the given prompt by a language model with the given capacity."""
    model_name = settings.model
    offload_folder = settings.offload_folder
    print(f"Completion request: {prompt=}, model: {model_name}")

    model = load_completion_model(model=model_name, offload_folder=offload_folder)
    tokenizer = load_tokenizer(model=model_name)

    response_text = get_response_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_response_length=max_response_length,
    )

    print(f"Completion response: {response_text}")
    return CompletionResponse(
        prompt=prompt,
        response=response_text,
        model=model_name,
    )


@functools.cache
def load_completion_model(model: str, offload_folder: Path) -> OPTForCausalLM | BloomForCausalLM:
    print(f"Loading model: {model}...")
    extra_kwargs = {}
    if model.startswith("bigscience/bloom"):
        extra_kwargs.update(load_in_8bit=True)
    pretrained_causal_lm_model = AutoModelForCausalLM.from_pretrained(
        model,
        device_map="auto",
        torch_dtype=torch.float16,
        offload_folder=offload_folder,
        use_auth_token=True,
        **extra_kwargs,
    )
    print("Done.")
    return pretrained_causal_lm_model


@functools.cache
def load_tokenizer(model: str) -> GPT2Tokenizer:
    print(f"Loading Tokenizer for model {model}...")
    # NOTE: See https://github.com/huggingface/tokenizers/pull/1005
    kwargs = {}
    if model.startswith("facebook/opt"):
        kwargs.update(use_fast=False)
    pretrained_tokenizer = AutoTokenizer.from_pretrained(
        model,
        device_map="auto",
        torch_dtype=torch.float16,
        **kwargs,
    )
    return pretrained_tokenizer


@torch.no_grad()
def get_response_text(
    model: OPTForCausalLM | BloomForCausalLM,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    max_response_length: int = 30,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    print(f"Generating based on {prompt=}...")
    generate_ids = model.generate(
        inputs.input_ids.to(model.device), max_length=max_response_length
    )
    prompt_and_response = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    assert isinstance(prompt_and_response, str)
    model_response = prompt_and_response.replace(prompt, "").lstrip()
    return model_response


# TODOs:
# - Check with students what kind of functionality they want, e.g. extracting representations:
# @torch.no_grad()
# def get_hidden_state(prompt: str, capacity: Capacity = DEFAULT_CAPACITY) -> Tensor:
#     inputs = tokenize(prompt)
#     model = load_embedding_model()
#     outputs = model(**inputs.to(model.device))

#     last_hidden_states = outputs.last_hidden_state
#     return last_hidden_states
# - Add a training example!
# - Create a slurm sbatch script to run this.


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_arguments(Settings, "settings", default=Settings())
    args = parser.parse_args()
    settings: Settings = args.settings

    HF_HOME = os.environ.setdefault("HF_HOME", str(settings.hf_cache_dir))
    TRANSFORMERS_CACHE = os.environ.setdefault(
        "TRANSFORMERS_CACHE", str(settings.hf_cache_dir / "transformers")
    )
    print(f"{HF_HOME=}")
    print(f"{TRANSFORMERS_CACHE=}")

    print(f"Running the server with the following settings: {settings.json()}")

    # NOTE: Can't use `reload` or `workers` when passing the app by value.
    if not settings.reload:
        app.dependency_overrides[get_settings] = lambda: settings
    else:
        # NOTE: If we we want to use `reload=True`, we set the environment variables, so they are
        # used when that module gets imported.
        for k, v in asdict(settings).items():
            os.environ[k.upper()] = str(v)

    write_server_address_to_file(port=settings.port)

    uvicorn.run(
        (app if not settings.reload else "app.server:app"),  # type: ignore
        port=settings.port,
        # Using the public IP makes the server publicly available, but a bit harder to debug (no
        # automatic port forwarding in VSCode for example).
        host=socket.gethostname() if settings.use_public_ip else "127.0.0.1",
        log_level="debug",
        reload=settings.reload,
    )


if __name__ == "__main__":
    main()
