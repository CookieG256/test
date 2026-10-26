import csv
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


class EventStore:
    CSV_COLUMNS = ("Nombre", "Cargo", "Acción", "Fecha", "Hora")

    def __init__(self, database_path: Path, csv_path: Path) -> None:
        self.database_path = database_path
        self.csv_path = csv_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
        self._initialize_csv()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize_database(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    action TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    image_path TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS person_state (
                    name TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    last_event_at TEXT NOT NULL
                )
                """
            )

    def _initialize_csv(self) -> None:
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            with self.csv_path.open("w", newline="", encoding="utf-8") as csv_file:
                csv.writer(csv_file).writerow(self.CSV_COLUMNS)

    def get_person_state(self, name: str) -> sqlite3.Row | None:
        with self._connection() as connection:
            return connection.execute(
                "SELECT status, last_event_at FROM person_state WHERE name = ?",
                (name,),
            ).fetchone()

    def record_known_event(
        self, name: str, role: str, action: str, timestamp: datetime
    ) -> None:
        status = "Dentro" if action == "ENTRADA" else "Fuera"
        timestamp_text = timestamp.isoformat(timespec="seconds")
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO events (name, role, action, created_at) VALUES (?, ?, ?, ?)",
                (name, role, action, timestamp_text),
            )
            connection.execute(
                """
                INSERT INTO person_state (name, status, last_event_at)
                VALUES (?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    status = excluded.status,
                    last_event_at = excluded.last_event_at
                """,
                (name, status, timestamp_text),
            )
        self._append_csv(name, role, action, timestamp)

    def record_unknown_event(self, timestamp: datetime, image_path: Path | None) -> None:
        image_path_text = str(image_path) if image_path else None
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO events (name, role, action, created_at, image_path)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    "Desconocido",
                    "Desconocido",
                    "INTENTO_ACCESO",
                    timestamp.isoformat(timespec="seconds"),
                    image_path_text,
                ),
            )
        self._append_csv("Desconocido", "Desconocido", "INTENTO_ACCESO", timestamp)

    def _append_csv(
        self,
        name: str,
        role: str,
        action: str,
        timestamp: datetime,
    ) -> None:
        with self.csv_path.open("a", newline="", encoding="utf-8") as csv_file:
            csv.writer(csv_file).writerow(
                (
                    name,
                    role,
                    action,
                    timestamp.strftime("%d/%m/%Y"),
                    timestamp.strftime("%H:%M:%S"),
                )
            )