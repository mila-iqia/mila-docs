# -*- coding: utf-8 -*-
# test
from __future__ import division, print_function, unicode_literals

from datetime import datetime

from recommonmark.parser import CommonMarkParser

import sphinx_theme

extensions = [
            'sphinx-prompt',
            'sphinx_copybutton',
            'recommonmark',
            'sphinx.ext.autosectionlabel',
            'sphinx.ext.todo']

templates_path = ['templates', '_templates', '.templates']
source_suffix = ['.rst', '.md']
source_parsers = {
            '.md': CommonMarkParser,
        }
master_doc = 'index'
project = u'MILA Technical Documentation'
copyright = str(datetime.now().year)
version = 'latest'
release = 'latest'

htmlhelp_basename = 'mila-docs'
file_insertion_enabled = False
latex_documents = [
  ('index', 'mila-docs.tex', u'Mila technical Documentation',
   u'', 'manual'),
]

exclude_patterns = ['_build',
                    'Userguide_*',
                    'Theory_cluster_*',
                    'Information_*',
                    'Purpose_*',
                    'Extra_compute_*',
                    'IDT_*',
                    'singularity/*']
# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    'logo_only': True
    }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_js_files = ['documentation_options.js', 'documentation_options_fix.js']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}
#---sphinx-themes-----
html_theme = 'neo_rtd_theme'
html_theme_path = [sphinx_theme.get_html_theme_path()]
html_logo = '_static/image.png'
html_context = {
    # Enable the "Edit in GitHub link within the header of each page.
    'display_github': True,
    # Set the following variables to generate the resulting github URL for each page.
    # Format Template: https://{{ github_host|default("github.com") }}/{{ github_user }}/{{ github_repo }}/blob/{{ github_version }}{{ conf_py_path }}{{ pagename }}{{ suffix }}
    'github_user': 'mila-iqia',
    'github_repo': 'mila-docs',
    'github_version': 'master/',
    'conf_py_path': 'docs/'
}

# Include CNAME file so GitHub Pages can set Custom Domain name
html_extra_path = ['CNAME']

def setup(app):
    app.add_css_file('custom.css')
