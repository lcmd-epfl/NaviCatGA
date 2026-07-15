# Configuration file for the Sphinx documentation builder.
import os
import sys

# Make the package importable for autoapi
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -------------------------------------------------------
project = "NaviCatGA"
copyright = "2021, rlaplaza, lcmd-epfl"
author = "rlaplaza, lcmd-epfl"
release = "0.1"

# -- General configuration -----------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "autoapi.extension",
    "sphinx_copybutton",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# autoapi: point at the package directly (no src/ layout here)
autoapi_dirs = ["../navicatGA"]
autoapi_type = "python"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_python_class_content = "both"
autoapi_member_order = "groupwise"
autoapi_add_toctree_entry = False  # added manually via api.md
autoapi_ignore = [
    "*/test/*",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

suppress_warnings = [
    "myst.xref_missing",
    "autoapi.python_import_resolution",
    "toc.not_readable",
]

# -- Options for HTML output ---------------------------------------------------
html_theme = "furo"
html_title = "NaviCatGA"
html_logo = "../images/navicatga_logo.png"

html_static_path = ["_static"]

html_theme_options = {
    "sidebar_hide_name": True,
    "source_repository": "https://github.com/lcmd-epfl/NaviCatGA",
    "source_branch": "master",
    "source_directory": "docs/",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}
