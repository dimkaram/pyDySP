# Configuration file for the Sphinx documentation builder.
#
# For the full list of options see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys

# Add the package src/ directory to sys.path so autodoc can import pydysp
sys.path.insert(0, os.path.abspath("../../src"))

from pydysp import __version__

# -- Project information -----------------------------------------------------

project = "pyDySP"
copyright = "2024, Dr Dimitris Karamitros"
author = "Dr Dimitris Karamitros"

# The full version, including alpha/beta/rc tags
release = __version__

# -- General configuration ---------------------------------------------------

# Sphinx extensions
extensions = [
    "sphinx.ext.autodoc",  # API docs from docstrings
    "sphinx.ext.napoleon",  # NumPy/Google style docstrings
    "sphinx.ext.viewcode",  # Add [source] links
]

# Templates path (relative to this directory)
templates_path = ["_templates"]

# Patterns to ignore when searching for source files
# (also affects html_static_path and html_extra_path)
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Autodoc configuration
autodoc_typehints = "description"  # show type hints in the description
autodoc_member_order = "bysource"  # keep members in source order

# Napoleon configuration (we use NumPy-style docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Options for HTML output -------------------------------------------------

# Theme for HTML pages
html_theme = "sphinx_rtd_theme"

# Paths that contain custom static files (such as style sheets).
# None are used; add "_static" here (and create the folder) if needed.
html_static_path = []
