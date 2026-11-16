"""DuckDB connection helper with explicit, portable resource limits.

Environment variables
---------------------
``FIRE_RELAY_DUCKDB_MEM``
    DuckDB ``memory_limit`` (default ``6GB``).
``FIRE_RELAY_THREADS``
    Worker thread count (default ``4``).
``FIRE_RELAY_DUCKDB_TMP``
    Optional spill directory.  Defaults to ``<cwd>/.duckdb_tmp`` so that a
    developer checkout never writes multi-GB spill files to ``/tmp``.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb

DEFAULT_MEMORY = os.environ.get("FIRE_RELAY_DUCKDB_MEM", "6GB")
DEFAULT_THREADS = int(os.environ.get("FIRE_RELAY_THREADS", "4"))
DEFAULT_MAX_TEMP = "200GB"


def _default_scratch() -> Path:
    configured = os.environ.get("FIRE_RELAY_DUCKDB_TMP")
    if configured:
        return Path(configured).expanduser()
    return Path.cwd() / ".duckdb_tmp"


def configure(
    con: duckdb.DuckDBPyConnection,
    *,
    memory_limit: str | None = None,
    threads: int | None = None,
    temp_dir: str | Path | None = None,
    max_temp_directory_size: str = DEFAULT_MAX_TEMP,
) -> duckdb.DuckDBPyConnection:
    """Apply memory, thread and spill settings to an open connection."""
    resolved_memory = memory_limit or DEFAULT_MEMORY
    resolved_threads = int(threads if threads is not None else DEFAULT_THREADS)
    scratch = Path(temp_dir or _default_scratch()).expanduser()
    scratch.mkdir(parents=True, exist_ok=True)
    safe_scratch = scratch.as_posix().replace("'", "''")
    safe_memory = str(resolved_memory).replace("'", "''")
    con.execute(f"SET memory_limit='{safe_memory}'")
    con.execute(f"SET threads={max(1, resolved_threads)}")
    con.execute(f"SET temp_directory='{safe_scratch}'")
    con.execute(f"SET max_temp_directory_size='{max_temp_directory_size}'")
    con.execute("SET preserve_insertion_order=false")
    return con


def connect(
    path: str | Path = ":memory:",
    *,
    memory_limit: str | None = None,
    threads: int | None = None,
    temp_dir: str | Path | None = None,
    max_temp_directory_size: str = DEFAULT_MAX_TEMP,
    **_: object,
) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection and immediately apply the standard settings."""
    con = duckdb.connect(str(path))
    return configure(
        con,
        memory_limit=memory_limit,
        threads=threads,
        temp_dir=temp_dir,
        max_temp_directory_size=max_temp_directory_size,
    )


def settings(con: duckdb.DuckDBPyConnection) -> dict[str, str]:
    """Return the effective DuckDB settings relevant to this project."""
    rows = con.execute(
        """
        SELECT name, value
        FROM duckdb_settings()
        WHERE name IN (
            'memory_limit', 'threads', 'temp_directory',
            'max_temp_directory_size', 'preserve_insertion_order'
        )
        """
    ).fetchall()
    return {str(name): str(value) for name, value in rows}


__all__ = ["DEFAULT_MEMORY", "DEFAULT_THREADS", "configure", "connect", "settings"]
