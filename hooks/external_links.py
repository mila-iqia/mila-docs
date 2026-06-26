import re

_EXTERNAL_A_RE = re.compile(
    r'<a(?=\s)([^>]*)href=["\']https?://[^"\']*["\']([^>]*)>',
    re.IGNORECASE,
)


def on_page_content(html, page, config, **kwargs):
    def add_new_tab(m):
        tag = m.group(0)
        if "target=" not in tag:
            tag = tag[:-1] + ' target="_blank" rel="noopener noreferrer">'
        return tag

    return _EXTERNAL_A_RE.sub(add_new_tab, html)
