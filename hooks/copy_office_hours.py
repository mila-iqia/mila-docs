from jinja2 import Environment, FileSystemLoader


def on_pre_build(config, **kwargs):
    """
    # Create markdown file for the Office hours from the configuration data
    # This is done before `mkdocs build` or `mkdocs serve` in order
    # to avoid duplicating the Office Hours times
    """

    env = Environment(loader=FileSystemLoader('hooks/templates'))
    office_hours_data = config["extra"]["office_hours"]
    
    for (template_file,output_file) in [
            ("office_hours_template.md", "office_hours.md"), # Long file (for pages)
            ("office_hours_short_template.md", "office_hours_short.md") # Short file (for cards)
        ]:
        template = env.get_template(template_file)
        output = template.render(office_hours_data=office_hours_data)

        with open(f"docs/generated_from_config/{output_file}", "w") as outfile:
            outfile.write(output)
