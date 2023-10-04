import openai


#
# Parse the server info from the job comment
#
def parse_meta(comment):
    data = dict()
    if comment != "(null)":
        items = comment.split("|")
        for kv in items:
            try:
                k, v = kv.split("=", maxsplit=1)
                data[k] = v
            except:
                pass

    return data


def get_job_comment(name="inference_server.sh"):
    command = ["squeue", "-h", f"--name={name}", '--format="%k"']

    return subprocess.check_output(command, text=True).replace('"', "")


server = parse_meta(get_job_comment())

# Override OpenAPI API URL with out custom server
openai.api_key = "EMPTY"
openai.api_base = f"http://{server['host']}:{server['port']}/v1"


# profit
completion = openai.Completion.create(
    model=server['model'], 
    prompt=args.prompt
)

print(completion)