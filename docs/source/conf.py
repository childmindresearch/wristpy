"""configuration file for Sphinx documentation."""

import pathlib
import sys

src_path = pathlib.Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))


project = "wristpy"
copyright = "2025, Child Mind Institute"
author = """ Adam Santorelli, Florian Rupprecht, Freymon Perez, Greg Kiar, 
Reinder Vos de Wael"""

release = "0.2.0"


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "sphinx_copybutton",
]
autosummary_generate = True
autosummary_imported_members = True

exclude_patterns = []

source_suffix = {
    ".rst": None,
    ".md": None,
}

html_theme = "furo"


html_static_path = ["_static"]


html_css_files = [
    "custom.css",
]


html_theme_options = {
    "sidebar_hide_name": False,
    "light_logo": "images/wristpy_logo.png",
    "dark_logo": "images/wristpy_logo.png",
}


napoleon_google_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = True


autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}


myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_image",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
}
