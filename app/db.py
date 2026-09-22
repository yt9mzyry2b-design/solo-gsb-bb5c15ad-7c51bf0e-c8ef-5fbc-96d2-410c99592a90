"""SQLite 持久化层：参考字与幂等提交记录。

使用单连接 + 锁（WAL 模式），FastAPI 同步端点在线程池中调用时保证线程安全。
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reference_glyphs (
    character   TEXT PRIMARY KEY,
    strokes     TEXT NOT NULL,          -- JSON: [[[x,y],...],...]
    source      TEXT,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS submissions (
    submission_id TEXT PRIMARY KEY,
    request_hash  TEXT NOT NULL,        -- 请求负载的 SHA-256
    response      TEXT NOT NULL,        -- JSON: ValidateResponse
    created_at    TEXT NOT NULL
);
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: str):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------ 参考字

    def upsert_reference(
        self, character: str, strokes: list, source: Optional[str]
    ) -> dict:
        now = _utcnow()
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO reference_glyphs (character, strokes, source, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(character) DO UPDATE SET
                    strokes = excluded.strokes,
                    source = excluded.source,
                    updated_at = excluded.updated_at
                """,
                (character, json.dumps(strokes), source, now),
            )
        return {
            "character": character,
            "stroke_count": len(strokes),
            "strokes": strokes,
            "source": source,
            "updated_at": now,
        }

    def get_reference(self, character: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM reference_glyphs WHERE character = ?",
                (character,),
            ).fetchone()
        if row is None:
            return None
        strokes = json.loads(row["strokes"])
        return {
            "character": row["character"],
            "stroke_count": len(strokes),
            "strokes": strokes,
            "source": row["source"],
            "updated_at": row["updated_at"],
        }

    def list_references(self) -> list[dict]:
        """列出全部已注册参考字，按字符排序（保证识别结果稳定）。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM reference_glyphs ORDER BY character ASC"
            ).fetchall()
        records = []
        for row in rows:
            strokes = json.loads(row["strokes"])
            records.append(
                {
                    "character": row["character"],
                    "stroke_count": len(strokes),
                    "strokes": strokes,
                    "source": row["source"],
                    "updated_at": row["updated_at"],
                }
            )
        return records

    # ------------------------------------------------------------ 幂等提交

    def get_submission(self, submission_id: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM submissions WHERE submission_id = ?",
                (submission_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "submission_id": row["submission_id"],
            "request_hash": row["request_hash"],
            "response": json.loads(row["response"]),
            "created_at": row["created_at"],
        }

    def snapshot_submissions(self) -> list[dict]:
        """在单个一致性快照中读取全部提交记录（一次加锁 + 单一查询）。

        返回结果按创建时间、幂等键升序；调用方可在快照之外安全地做过滤与
        聚合，统计过程中即便有新提交落库也不会混入本次结果。
        """
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT submission_id, request_hash, response, created_at
                FROM submissions
                ORDER BY created_at ASC, submission_id ASC
                """
            ).fetchall()
        return [
            {
                "submission_id": row["submission_id"],
                "request_hash": row["request_hash"],
                "response": json.loads(row["response"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def save_submission(
        self, submission_id: str, request_hash: str, response: dict
    ) -> tuple[dict, bool]:
        """写入提交记录。返回 (记录, 是否由本次调用插入)。

        并发下同键写入由主键约束兜底，以首个落库结果为准。
        """
        now = _utcnow()
        inserted = False
        with self._lock, self._conn:
            try:
                self._conn.execute(
                    """
                    INSERT INTO submissions (submission_id, request_hash, response, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (submission_id, request_hash, json.dumps(response), now),
                )
                inserted = True
            except sqlite3.IntegrityError:
                pass  # 并发重复：以首个写入为准
            row = self._conn.execute(
                "SELECT * FROM submissions WHERE submission_id = ?",
                (submission_id,),
            ).fetchone()
        record = {
            "submission_id": row["submission_id"],
            "request_hash": row["request_hash"],
            "response": json.loads(row["response"]),
            "created_at": row["created_at"],
        }
        return record, inserted


_db: Optional[Database] = None


def init_db(path: str) -> Database:
    global _db
    _db = Database(path)
    return _db


def get_db() -> Database:
    if _db is None:
        raise RuntimeError("数据库未初始化")
    return _db
