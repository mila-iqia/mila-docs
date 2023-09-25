""" TODO: Client-side code to communicate with the server that is running somewhere on the cluster.

IDEAS:
- Could look for slurm jobs that have a given name, like `deploy.sh` and extract the port from the
  job's command-line ags!
"""
from pathlib import Path
import requests
import time


def get_server_url_and_port() -> tuple[str, int]:
    with open("server.txt") as f:
        server_url_and_port = f.read().strip()
    server_url, _, port = server_url_and_port.partition(":")
    return server_url, int(port)


def debug():
    # WIP: Not working yet.
    while not Path("server.txt").exists():
        time.sleep(1)
        print(f"Waiting for server to start...")
    server_url, port = get_server_url_and_port()
    print(f"Found server at {server_url}:{port}")
    response = requests.get(
        f"http://{server_url}:{port}/complete/",
        params={
            "prompt": "Hello, my name is Bob. I love fishing, hunting, and my favorite food is",
        },
    )
    print(response)


if __name__ == "__main__":
    debug()
