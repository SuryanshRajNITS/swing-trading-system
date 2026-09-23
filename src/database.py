"""Minimal SQLite connection and health-check helpers."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from src.configuration import DatabaseSettings


def connect(settings: DatabaseSettings) -> sqlite3.Connection:
    """Open the configured local database, creating its parent directory."""
    settings.path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(settings.path)


def initialize_database(settings: DatabaseSettings) -> None:
    """Initialize an empty SQLite database and verify it accepts a query."""
    with closing(connect(settings)) as connection:
        connection.execute("SELECT 1").fetchone()


def database_is_healthy(settings: DatabaseSettings) -> bool:
    """Return whether the configured SQLite database can answer a basic query."""
    try:
        with closing(connect(settings)) as connection:
            return connection.execute("SELECT 1").fetchone() == (1,)
    except sqlite3.Error:
        return False
