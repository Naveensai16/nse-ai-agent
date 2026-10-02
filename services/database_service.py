"""SQLite-backed conversation and message persistence service."""

import logging
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Optional, Union

logger = logging.getLogger(__name__)

DEFAULT_DB_FILENAME = "conversations.sqlite3"


def get_default_db_path() -> Path:
    """Resolve the default SQLite database path from environment or project structure."""
    data_dir = os.getenv("DATA_DIR", "./data")
    path = Path(data_dir) / DEFAULT_DB_FILENAME
    return path.resolve()


@contextmanager
def get_db_connection(
    db_path: Optional[Union[str, Path]] = None,
) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for SQLite connections with foreign keys enabled.

    Args:
        db_path: Target SQLite database file path. Defaults to get_default_db_path().

    Yields:
        Configured sqlite3.Connection.
    """
    resolved_path = Path(db_path) if db_path is not None else get_default_db_path()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(resolved_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize_database(db_path: Optional[Union[str, Path]] = None) -> None:
    """Create conversation and message tables if they do not exist.

    Args:
        db_path: Target SQLite database file path.
    """
    with get_db_connection(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conversation_id 
                ON messages (conversation_id);
            CREATE INDEX IF NOT EXISTS idx_messages_created_at 
                ON messages (created_at);
            CREATE INDEX IF NOT EXISTS idx_conversations_updated_at 
                ON conversations (updated_at);
            """
        )
    logger.debug("Database initialized successfully at %s", db_path)


def create_conversation(
    title: str = "New Conversation",
    conversation_id: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    """Create and persist a new conversation session.

    Args:
        title: Human-readable title for the conversation.
        conversation_id: Optional pre-defined UUID string.
        db_path: Target database path.

    Returns:
        Dictionary representing the created conversation record.
    """
    initialize_database(db_path)
    cid = conversation_id or str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    with get_db_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO conversations (conversation_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (cid, title, now, now),
        )

    return {
        "conversation_id": cid,
        "title": title,
        "created_at": now,
        "updated_at": now,
    }


def list_conversations(
    limit: int = 50,
    db_path: Optional[Union[str, Path]] = None,
) -> list[dict[str, Any]]:
    """List recent conversations ordered by last update time descending.

    Args:
        limit: Maximum number of conversations to retrieve.
        db_path: Target database path.

    Returns:
        List of conversation records as dictionaries.
    """
    initialize_database(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT conversation_id, title, created_at, updated_at
            FROM conversations
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]


def get_conversation(
    conversation_id: str,
    db_path: Optional[Union[str, Path]] = None,
) -> Optional[dict[str, Any]]:
    """Retrieve metadata for a single conversation by ID.

    Args:
        conversation_id: UUID of conversation.
        db_path: Target database path.

    Returns:
        Conversation dictionary if found, else None.
    """
    initialize_database(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT conversation_id, title, created_at, updated_at
            FROM conversations
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def add_message(
    conversation_id: str,
    role: str,
    content: str,
    message_id: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    """Append a new message to an existing conversation and update conversation timestamp.

    Args:
        conversation_id: UUID of the target conversation.
        role: Role of the message sender ('user', 'assistant', 'system').
        content: Text content of the message.
        message_id: Optional custom message UUID string.
        db_path: Target database path.

    Returns:
        Dictionary representing the created message record.

    Raises:
        sqlite3.IntegrityError: If conversation_id does not exist.
    """
    initialize_database(db_path)
    mid = message_id or str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    with get_db_connection(db_path) as conn:
        # Insert message (will fail with IntegrityError if conversation_id does not exist)
        conn.execute(
            """
            INSERT INTO messages (message_id, conversation_id, role, content, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (mid, conversation_id, role, content, now),
        )
        # Update conversation updated_at
        conn.execute(
            """
            UPDATE conversations
            SET updated_at = ?
            WHERE conversation_id = ?
            """,
            (now, conversation_id),
        )

    return {
        "message_id": mid,
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
        "created_at": now,
    }


def get_messages(
    conversation_id: str,
    db_path: Optional[Union[str, Path]] = None,
) -> list[dict[str, Any]]:
    """Retrieve all messages for a conversation ordered chronologically.

    Args:
        conversation_id: UUID of the target conversation.
        db_path: Target database path.

    Returns:
        List of message dictionaries ordered from oldest to newest.
    """
    initialize_database(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT message_id, conversation_id, role, content, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at ASC
            """,
            (conversation_id,),
        )
        return [dict(row) for row in cursor.fetchall()]


def update_conversation_title(
    conversation_id: str,
    new_title: str,
    db_path: Optional[Union[str, Path]] = None,
) -> bool:
    """Update conversation title and its updated_at timestamp.

    Args:
        conversation_id: UUID of conversation.
        new_title: Replacement title text.
        db_path: Target database path.

    Returns:
        True if conversation exists and was updated, False otherwise.
    """
    initialize_database(db_path)
    now = datetime.now(timezone.utc).isoformat()
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            """
            UPDATE conversations
            SET title = ?, updated_at = ?
            WHERE conversation_id = ?
            """,
            (new_title, now, conversation_id),
        )
        return cursor.rowcount > 0


def delete_conversation(
    conversation_id: str,
    db_path: Optional[Union[str, Path]] = None,
) -> bool:
    """Delete conversation and cascade deletion to all its messages.

    Args:
        conversation_id: UUID of conversation to delete.
        db_path: Target database path.

    Returns:
        True if conversation existed and was deleted, False otherwise.
    """
    initialize_database(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.execute(
            """
            DELETE FROM conversations
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        )
        return cursor.rowcount > 0
