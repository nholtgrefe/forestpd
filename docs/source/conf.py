"""Sphinx configuration for the forestpd documentation."""

from __future__ import annotations

import importlib.metadata

try:
    release = importlib.metadata.version("forestpd")
except importlib.metadata.PackageNotFoundError:
    release = "0.1.0-dev.1"

project = "forestpd"
author = "Niels Holtgrefe"
copyright = "2026, Niels Holtgrefe"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "matplotlib.sphinxext.plot_directive",
]
try:
    import sphinx_copybutton  # noqa: F401

    extensions.append("sphinx_copybutton")
except ImportError:
    pass

# Docstrings are NumPy style and already state every type, so the signatures stay short.
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_ivar = True  # "Attributes" sections as fields of the class, not separate entries
autodoc_typehints = "none"
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True}
toc_object_entries_show_parents = "hide"  # "optimize()", not "MaxMPDInstance.optimize()"
# PhyloZoo is optional; forestpd.io.pz is documented whether or not it is installed.
autodoc_mock_imports = ["phylozoo"]

# Figures of the examples are drawn when the docs are built, from the code shown.
plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = [("png", 150)]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

html_theme = "pydata_sphinx_theme"
html_title = "ForestPD"
html_favicon = "_static/forestpd-icon.svg"
html_static_path = ["_static"]
html_theme_options = {
    "logo": {
        "image_light": "_static/forestpd-logo-compact.svg",
        "image_dark": "_static/forestpd-logo-compact-dark.svg",
        "alt_text": "ForestPD",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/nholtgrefe/forestpd",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
    "navbar_align": "left",
    "show_toc_level": 2,
}
html_context = {"default_mode": "auto"}
# The guide pages have no subpages, so no left sidebar; the API pages keep theirs.
html_sidebars = {page: [] for page in ("index", "installation", "problem", "manual", "examples")}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# Docstrings refer to objects by the module that defines them (``forestpd.instance.MaxMPDInstance``),
# the pages document them where users import them from (``forestpd.MaxMPDInstance``).
PUBLIC_MODULES = {
    "forestpd.instance": "forestpd",
    "forestpd.api": "forestpd",
    "forestpd.core.taxa": "forestpd",
    "forestpd.core.tree": "forestpd",
    "forestpd.core": "forestpd",
    "forestpd.exceptions": "forestpd",
    "forestpd.maf.base": "forestpd.maf",
    "forestpd.maf.heuristic": "forestpd.maf",
    "forestpd.maf.maffe": "forestpd.maf",
    "forestpd.maf.tiered": "forestpd.maf",
    "forestpd.io.newick": "forestpd.io",
    "forestpd.io.splits": "forestpd.io",
}


def resolve_public_name(app, env, node, contnode):
    """
    Resolve a Python reference the domain could not: a defining module's path by the object's
    public name, and a bare name (``split_system``) by the one documented object of that name.
    """
    if node.get("refdomain") != "py":
        return None
    domain = env.get_domain("py")
    target = node["reftarget"]
    candidate = None
    for private, public in sorted(PUBLIC_MODULES.items(), key=lambda item: -len(item[0])):
        if target.startswith(private + "."):
            candidate = public + target[len(private) :]
            break
    if candidate is None or candidate not in domain.objects:
        matches = [name for name in domain.objects if name.endswith("." + target)]
        candidate = matches[0] if len(matches) == 1 else None
    if candidate is None:
        return None
    return domain.resolve_xref(
        env, node["refdoc"], app.builder, node["reftype"], candidate, node, contnode
    )


def setup(app):
    app.connect("missing-reference", resolve_public_name)
