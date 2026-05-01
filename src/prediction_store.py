from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def init_prediction_store(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path))
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                source TEXT NOT NULL,
                prediction_mode TEXT NOT NULL,
                predicted_price REAL NOT NULL,
                price_range_low REAL NOT NULL,
                price_range_high REAL NOT NULL,
                input_json TEXT NOT NULL,
                output_json TEXT NOT NULL,
                github_status TEXT,
                github_detail TEXT
            )
            """
        )
        connection.commit()
    finally:
        connection.close()


def log_prediction(
    path: str | Path,
    *,
    source: str,
    input_payload: dict,
    output_payload: dict,
    github_status: str | None = None,
    github_detail: str | None = None,
) -> int:
    init_prediction_store(path)
    connection = sqlite3.connect(str(path))
    try:
        cursor = connection.execute(
            """
            INSERT INTO predictions (
                created_at,
                source,
                prediction_mode,
                predicted_price,
                price_range_low,
                price_range_high,
                input_json,
                output_json,
                github_status,
                github_detail
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                source,
                str(output_payload.get("prediction_mode", "unknown")),
                float(output_payload.get("predicted_price", 0)),
                float(output_payload.get("price_range_low", 0)),
                float(output_payload.get("price_range_high", 0)),
                json.dumps(input_payload, ensure_ascii=True),
                json.dumps(output_payload, ensure_ascii=True),
                github_status,
                github_detail,
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)
    finally:
        connection.close()


def recent_predictions(path: str | Path, limit: int = 20) -> list[dict]:
    init_prediction_store(path)
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT id, created_at, source, prediction_mode, predicted_price, price_range_low, price_range_high, github_status, github_detail
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def update_github_status(path: str | Path, prediction_id: int, github_status: str, github_detail: str | None = None) -> None:
    init_prediction_store(path)
    connection = sqlite3.connect(str(path))
    try:
        connection.execute(
            """
            UPDATE predictions
            SET github_status = ?, github_detail = ?
            WHERE id = ?
            """,
            (github_status, github_detail, prediction_id),
        )
        connection.commit()
    finally:
        connection.close()
