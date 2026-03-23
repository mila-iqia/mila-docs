from jinja2 import Environment, FileSystemLoader


def on_pre_build(config, **kwargs):
    """
    # Create markdown file for the Office hours from the configuration data
    # This is done before `mkdocs build` or `mkdocs serve` in order
    # to avoid duplicating the Office Hours times
    """

    env = Environment(loader=FileSystemLoader('hooks/templates'))
    template = env.get_template('office_hours_template.md')
    output = template.render(office_hours_data = config["extra"]["office_hours"])
    with open("docs/generated_from_config/office_hours.md", "w") as outfile:
        outfile.write(output)