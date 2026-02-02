""" TODO: Client-side code to communicate with the server that is running somewhere on the cluster.

IDEAS:
- Could look for slurm jobs that have a given name, like `deploy.sh` and extract the port from the
  job's command-line ags!
"""
from pathlib import Path
import requests
import time


def _fetch_job_info(name):
    # Mock this for testing
    command = ["squeue", "-h", f"--name={name}", "--format=\"%A %j %T %P %U %k %N\""]
    return subprocess.check_output(command, text=True)


def get_slurm_job_by_name(name):
    """Retrieve a list of jobs that match a given job name"""

    output =_fetch_job_info(name)
    jobs = []

    def parse_meta(comment):
        data = dict()
        if comment != "(null)":
            items = comment.split('|')
            for kv in items:
                try:
                    k, v = kv.split('=', maxsplit=1)
                    data[k] = v
                except: 
                    pass

        return data

    for line in output.splitlines():
        job_id, job_name, status, partition, user, comment, nodes = line.split(' ')

        jobs.append({
            "job_id":job_id, 
            "job_name":job_name, 
            "status":status,
            "partition":partition, 
            "user":user,
            "comment": parse_meta(comment),
            "nodes": nodes
        })

    return jobs


def find_suitable_inference_server(jobs, model):
    """Select suitable jobs from a list, looking for a specific model"""
    selected = []
    
    def is_shared(job):
        return job["comment"].get("shared", 'y') == 'y'
    
    def is_running(job):
        return job['status'] == "RUNNING"
    
    def has_model(job, model):
        if model is None:
            return True
        
        # FIXME: 
        #   /network/weights/llama.var/llama2/Llama-2-7b-hf != meta-llama/Llama-2-7b-hf
        #
        return job['comment']['model'] == model
    
    def select(job):
        selected.append({
            "model": job['comment']["model"],
            "host": job["comment"]["host"],
            "port": job["comment"]["port"],
        })
            
    for job in jobs:
        if is_shared(job) and is_running(job) and has_model(job, model):
            select(job)
                
    return selected


def get_inference_server(model=None):
    """Retrieve an inference server from slurm jobs"""
    jobs = get_slurm_job_by_name('inference_server_SHARED.sh')

    servers = find_suitable_inference_server(jobs, model)

    try:
        return random.choice(servers)
    except IndexError:
        return None


def get_server_url_and_port() -> tuple[str, int]:
    server = get_inference_server(model)

    if server is None:
        return None
    
    return server['host'], int(server['port'])


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
