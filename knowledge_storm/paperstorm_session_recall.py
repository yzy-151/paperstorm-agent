"""Fast, auditable cross-session recall backed by SQLite FTS5."""

import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


class SessionRecallStore:
    """Store full chat transcripts separately from durable user facts."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(str(self.path), timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self):
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS session_messages (
                    message_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    sequence_no INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_session_messages_scope
                    ON session_messages(user_id, chat_id, sequence_no);
                CREATE VIRTUAL TABLE IF NOT EXISTS session_messages_fts USING fts5(
                    content,
                    content='session_messages',
                    content_rowid='rowid',
                    tokenize='unicode61'
                );
                CREATE TRIGGER IF NOT EXISTS session_messages_ai AFTER INSERT ON session_messages BEGIN
                    INSERT INTO session_messages_fts(rowid, content) VALUES (new.rowid, new.content);
                END;
                CREATE TRIGGER IF NOT EXISTS session_messages_ad AFTER DELETE ON session_messages BEGIN
                    INSERT INTO session_messages_fts(session_messages_fts, rowid, content)
                    VALUES ('delete', old.rowid, old.content);
                END;
                CREATE TRIGGER IF NOT EXISTS session_messages_au AFTER UPDATE ON session_messages BEGIN
                    INSERT INTO session_messages_fts(session_messages_fts, rowid, content)
                    VALUES ('delete', old.rowid, old.content);
                    INSERT INTO session_messages_fts(rowid, content) VALUES (new.rowid, new.content);
                END;
                """
            )

    def append_message(
        self,
        user_id,
        chat_id,
        message_id,
        role,
        content,
        metadata=None,
        created_at=None,
    ):
        content = str(content or "").strip()
        if not content:
            return {"status": "skipped", "reason": "empty content"}
        with self._connect() as connection:
            sequence = connection.execute(
                "SELECT COALESCE(MAX(sequence_no), 0) + 1 FROM session_messages WHERE chat_id=?",
                (str(chat_id),),
            ).fetchone()[0]
            connection.execute(
                """INSERT OR IGNORE INTO session_messages
                   (message_id, user_id, chat_id, sequence_no, role, content,
                    metadata_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(message_id),
                    str(user_id),
                    str(chat_id),
                    int(sequence),
                    str(role),
                    content,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    created_at or _now(),
                ),
            )
        return {"status": "stored", "message_id": str(message_id)}

    def search(self, user_id, query, top_k=5, context_radius=2):
        query = str(query or "").strip()
        if not query:
            return _empty_result(query)
        match_query = _fts_query(query)
        if not match_query:
            return _empty_result(query)
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT m.*, bm25(session_messages_fts) AS rank
                   FROM session_messages_fts
                   JOIN session_messages m ON m.rowid=session_messages_fts.rowid
                   WHERE session_messages_fts MATCH ? AND m.user_id=?
                   ORDER BY rank ASC, m.created_at DESC LIMIT ?""",
                (match_query, str(user_id), max(1, int(top_k))),
            ).fetchall()
            if not rows:
                rows = self._search_cjk_fallback(
                    connection,
                    user_id=str(user_id),
                    query=query,
                    top_k=max(1, int(top_k)),
                )
            results = []
            for row in rows:
                neighbors = connection.execute(
                    """SELECT role, content FROM session_messages
                       WHERE user_id=? AND chat_id=? AND sequence_no BETWEEN ? AND ?
                       ORDER BY sequence_no""",
                    (
                        str(user_id),
                        row["chat_id"],
                        max(1, int(row["sequence_no"]) - int(context_radius)),
                        int(row["sequence_no"]) + int(context_radius),
                    ),
                ).fetchall()
                results.append(
                    {
                        "message_id": row["message_id"],
                        "chat_id": row["chat_id"],
                        "role": row["role"],
                        "content": row["content"],
                        "created_at": row["created_at"],
                        "score": round(-float(row["rank"]), 6),
                        "context": [item["content"] for item in neighbors],
                    }
                )
        return {
            "query": query,
            "results": results,
            "retrieval": "sqlite_fts5_bm25",
            "candidate_count": len(results),
        }

    @staticmethod
    def _search_cjk_fallback(connection, user_id, query, top_k):
        """Recover Chinese substring matches that unicode61 cannot segment."""
        grams = _cjk_ngrams(query, size=4)
        if not grams:
            return []
        clauses = " OR ".join("content LIKE ?" for _ in grams)
        score = " + ".join("CASE WHEN content LIKE ? THEN 1 ELSE 0 END" for _ in grams)
        patterns = ["%{0}%".format(gram) for gram in grams]
        return connection.execute(
            """SELECT *, -CAST(({score}) AS REAL) AS rank
               FROM session_messages
               WHERE user_id=? AND ({clauses})
               ORDER BY rank ASC, created_at DESC LIMIT ?""".format(
                score=score,
                clauses=clauses,
            ),
            tuple(patterns + [user_id] + patterns + [top_k]),
        ).fetchall()


def _fts_query(query):
    terms = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}", str(query))
    unique = []
    for term in terms:
        normalized = term.lower()
        if normalized not in unique:
            unique.append(normalized)
    return " OR ".join('"{0}"'.format(term.replace('"', '""')) for term in unique)


def _cjk_ngrams(text, size=4):
    sequences = re.findall(r"[\u4e00-\u9fff]{2,}", str(text or ""))
    grams = []
    for sequence in sequences:
        width = min(max(2, int(size)), len(sequence))
        grams.extend(sequence[index:index + width] for index in range(len(sequence) - width + 1))
    return list(dict.fromkeys(grams))[:24]


def _empty_result(query):
    return {
        "query": query,
        "results": [],
        "retrieval": "sqlite_fts5_bm25",
        "candidate_count": 0,
    }


def _now():
    return datetime.now(timezone.utc).isoformat()
