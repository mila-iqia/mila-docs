def on_page_content(html, page, **kwargs):
    skills = page.meta.get("skills", [])
    if not skills:
        return html
    identifiers = " ".join(skills)
    return html + f'\n<span class="skill-identifiers">{identifiers}</span>'
