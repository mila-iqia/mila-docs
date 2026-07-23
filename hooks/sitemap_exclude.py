"""
Hook: sitemap_exclude.py

MkDocs Material's `search: exclude: true` frontmatter suppresses a page from the
search index, but it does not remove the page from sitemap.xml. Without this
hook, search-excluded pages would still appear in the sitemap and could be
discovered and indexed by crawlers.

This hook reuses the same `search: exclude: true` signal to also remove those
pages from the sitemap, so no new frontmatter key is required.

Build flow
----------
The hook runs across three MkDocs lifecycle phases within a single build:

1. on_pre_build — clears the shared URL set so each build starts clean.
2. on_page_context — inspects each rendered page; if it is marked ``search:
   exclude: true``, its canonical URL is added to the set.
3. on_post_build — edits sitemap.xml on disk to remove the collected URLs, then
   regenerates sitemap.xml.gz.

MkDocs Material produces two sitemap files: sitemap.xml (plain XML) and
sitemap.xml.gz (the same content compressed with gzip). Both must be updated
together — see on_post_build for details.

Why post-build editing?
MkDocs writes sitemap.xml during the build, before on_post_build fires. There is
no earlier hook phase that can intercept sitemap generation, so patching the
file on disk after the build completes is the only option.
"""

import gzip
import logging
import os
import shutil
import xml.etree.ElementTree as ET

log = logging.getLogger("mkdocs.hooks.sitemap_exclude")

# Module-level set so all three hook functions share state within one build.
_excluded_urls: set[str] = set()


def on_pre_build(config, **kwargs):
    """Reset the excluded URL set at the start of every build.

    ``mkdocs serve`` keeps the module loaded between incremental rebuilds, so
    module-level state persists across builds. Resetting here ensures that URLs
    collected during a previous build do not carry over into the next one.
    """
    global _excluded_urls
    _excluded_urls = set()


def on_page_context(context, page, config, nav, **kwargs):
    """Collect canonical URLs for pages that opt out of search (and sitemap).

    ``page.canonical_url`` is used because it is the exact URL MkDocs writes
    into the sitemap's ``<loc>`` elements, making the lookup in on_post_build
    reliable regardless of trailing-slash or base-URL configuration.
    """
    if page.meta.get("search", {}).get("exclude"):
        if page.canonical_url:
            _excluded_urls.add(page.canonical_url)
            log.debug("sitemap_exclude: queued for removal: %s", page.canonical_url)
    return context


def on_post_build(config, **kwargs):
    """Remove collected URLs from sitemap.xml and regenerate sitemap.xml.gz.

    Inline comments below explain the two non-obvious implementation choices:
    why the XML declaration is written manually, and why sitemap.xml.gz must be
    regenerated after patching sitemap.xml.
    """
    if not _excluded_urls:
        log.debug("sitemap_exclude: no pages to exclude, sitemap unchanged.")
        return

    site_dir = config["site_dir"]
    sitemap_path = os.path.join(site_dir, "sitemap.xml")
    sitemap_gz_path = sitemap_path + ".gz"

    if not os.path.isfile(sitemap_path):
        log.warning(
            "sitemap_exclude: sitemap.xml not found at %s — skipping.", sitemap_path
        )
        return

    # Register the sitemap namespace before parsing so ElementTree serialises it
    # without an ugly "ns0:" prefix on output.
    SITEMAP_NS = "http://www.sitemaps.org/schemas/sitemap/0.9"
    ET.register_namespace("", SITEMAP_NS)

    try:
        tree = ET.parse(sitemap_path)
    except ET.ParseError as exc:
        log.error("sitemap_exclude: failed to parse sitemap.xml: %s", exc)
        return

    root = tree.getroot()

    # Detect Clark-notation namespace prefix from root tag (e.g.
    # "{http://...}").
    ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""

    removed = 0
    for url_elem in root.findall(f"{ns}url"):
        loc_elem = url_elem.find(f"{ns}loc")
        if loc_elem is not None and loc_elem.text in _excluded_urls:
            root.remove(url_elem)
            removed += 1
            log.info("sitemap_exclude: removed %s", loc_elem.text)

    # Prepend the XML declaration manually instead of using
    # ET.write(xml_declaration=True). That flag would automatically add the
    # <?xml ...?> processing instruction, but Python's ElementTree always
    # formats it with single-quoted attributes:
    #   <?xml version='1.0' encoding='utf-8'?>
    # MkDocs Material writes the declaration with double-quoted attributes and
    # uppercase encoding:
    #   <?xml version="1.0" encoding="UTF-8"?>
    # Overwriting the file with the single-quoted form would introduce a
    # persistent, cosmetic diff on every build. Writing the declaration as a
    # byte literal keeps the output byte-for-byte identical to what Material
    # originally produced.
    xml_bytes = b'<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(
        root, encoding="unicode"
    ).encode("utf-8")

    with open(sitemap_path, "wb") as fh:
        fh.write(xml_bytes)

    log.info("sitemap_exclude: wrote cleaned sitemap.xml (%d URL(s) removed).", removed)

    # Regenerate sitemap.xml.gz to keep it in sync with the patched sitemap.xml.
    # MkDocs Material produces two sitemap files: sitemap.xml is plain XML, and
    # sitemap.xml.gz is the same content compressed with gzip. Many crawlers,
    # including Googlebot, fetch the compressed version directly to save
    # bandwidth. If only sitemap.xml were patched, the two files would diverge
    # and different crawlers would see different content. The .gz file is only
    # regenerated if Material produced it during this build.
    if os.path.isfile(sitemap_gz_path):
        with (
            open(sitemap_path, "rb") as f_in,
            gzip.open(sitemap_gz_path, "wb", compresslevel=9) as f_out,
        ):
            shutil.copyfileobj(f_in, f_out)
        log.info("sitemap_exclude: regenerated sitemap.xml.gz.")
