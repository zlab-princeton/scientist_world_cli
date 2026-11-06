"""sciworld: run an AI scientist inside a research environment built from a paper.

The ``sw`` command line is the usual entry point. The same operations are
available as functions::

    import sciworld

    sciworld.init(data_dir="~/sciworld-data", backend="venv")
    run = sciworld.research(id="2502.13759", agent="claude-code", mode="innovate")
    sciworld.evaluate(id="2502.13759", run=run["run_name"])
"""

__version__ = "0.3.0"

from sciworld._api import (  # noqa: E402
    config,
    evaluate,
    info,
    init,
    list_environments,
    research,
    select,
)

__all__ = [
    "__version__",
    "config",
    "evaluate",
    "info",
    "init",
    "list_environments",
    "research",
    "select",
]
