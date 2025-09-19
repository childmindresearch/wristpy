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
    "myst_nb",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "sphinx_copybutton",
]
autosummary_generate = True
autosummary_imported_members = True

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

html_theme = "furo"


html_static_path = ["_static"]


html_css_files = [
    "custom.css",
]

html_domain_indices = ["py-modindex"]
modindex_common_prefix = ["wristpy."]

html_theme_options = {
    "sidebar_hide_name": False,
    "light_logo": "images/wristpy_logo_light.png",
    "dark_logo": "images/wristpy_logo_dark.png",
}


napoleon_google_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = True


autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

nb_execution_mode = "force"
nb_execution_timeout = 60
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

#
