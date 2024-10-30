# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "PyMAB"
copyright = "2024, Daniela Lopes"
author = "Daniela Lopes"
release = "0.1.0-alpha.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",        # API documentation generation
    "sphinx.ext.doctest",        # Test code snippets in documentation
    "sphinx.ext.viewcode",       # Add links to highlighted source code
    "sphinx.ext.intersphinx",    # Link to other projects' documentation
    "sphinx.ext.napoleon",       # Support for Google and NumPy style docstrings
    "sphinx.ext.coverage",       # Documentation coverage checking
    "sphinx.ext.githubpages",    # Generate .nojekyll file for GitHub Pages
    "sphinx.ext.mathjax",        # Render math via MathJax
    "sphinx.ext.todo",           # Support for TODO items
    "sphinx.ext.autosummary",    # Generate summary tables
    "piccolo_theme",             # Theme
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
    "inherited-members": True,
}
autodoc_typehints = "description"
autodoc_class_signature = "separated"
autodoc_typehints_format = "short"

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
source_encoding = "utf-8-sig"

templates_path = ["_templates"]
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "piccolo_theme"
html_theme_options = {
    "globaltoc_collapse": False,
    "globaltoc_maxdepth": 4,
    "navigation_with_keys": True,
    "show_nav_level": 2,
    "show_toc_level": 2,
}
html_static_path = ["_static"]

add_module_names = False
python_use_unqualified_type_names = True
pygments_style = "sphinx"

todo_include_todos = True

autosummary_generate = True
autosummary_imported_members = True

nitpicky = True
