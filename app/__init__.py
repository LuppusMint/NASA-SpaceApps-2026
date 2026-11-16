"""Fire Relay South America web application package.

The package exposes :func:`app.server.create_app`, a dependency-free WSGI
application that reads the harmonized Parquet artefacts produced by the
preprocessing and modelling pipelines and serves the offline bilingual
dashboard in ``app/static``.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "3.0.0"
