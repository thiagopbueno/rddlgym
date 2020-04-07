# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import rddlgym

import sphinx_rtd_theme


# -- Project information -----------------------------------------------------

project = "rddlgym"
copyright = "2018-2020, Thiago P. Bueno"
author = "Thiago P. Bueno"

# The full version, including alpha/beta/rc tags
# release = rddlgym.__release__
# version = rddlgym.__version__
version = "0.5.14"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinxcontrib.napoleon",
    "sphinx_rtd_theme",
]

# -- Extension configuration -------------------------------------------------
# napoleon_include_private_with_doc = True
# napoleon_include_special_with_doc = True


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

import sphinx_rtd_theme

html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Sort members by type
autodoc_member_order = "bysource"

autodoc_default_options = {
    # "member-order": "bysource",
    "special-members": "special-members",
    "private-members": "private-members",
}
