"""Unit tests for the SQLite conversation and message database service."""

import sqlite3
import time
from pathlib import Path

import pytest

from services.database_service import (
    add_message,
    create_conversation,
    delete_conversation,
    get_conversation,
    get_messages,
    initialize_database,
    list_conversations,
    update_conversation_title,
)


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """Provide an isolated temporary SQLite database path for each test."""
    db_file = tmp_path / "test_chat.sqlite3"
    return db_file


class TestDatabaseService:
    """Test suite for conversation and message storage using SQLite."""

    def test_database_initialization(self, temp_db: Path) -> None:
        """Verify database tables and indexes are created properly and idempotently."""
        initialize_database(temp_db)

        # Inspect sqlite_master for created tables
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('conversations', 'messages')"
        )
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert "conversations" in tables
        assert "messages" in tables

        # Re-initialization should be safe and idempotent
        initialize_database(temp_db)

    def test_create_conversation(self, temp_db: Path) -> None:
        """Test creating a conversation generates UUID and valid timestamps."""
        conv = create_conversation(title="NIFTY 50 Analysis", db_path=temp_db)

        assert isinstance(conv["conversation_id"], str)
        assert len(conv["conversation_id"]) == 36  # Standard UUID length
        assert conv["title"] == "NIFTY 50 Analysis"
        assert conv["created_at"] is not None
        assert conv["updated_at"] is not None

        # Verify retrieval matches
        fetched = get_conversation(conv["conversation_id"], db_path=temp_db)
        assert fetched is not None
        assert fetched["conversation_id"] == conv["conversation_id"]
        assert fetched["title"] == "NIFTY 50 Analysis"

    def test_list_conversations(self, temp_db: Path) -> None:
        """Test listing conversations respects limits and orders by updated_at descending."""
        conv1 = create_conversation(title="Conv 1", db_path=temp_db)
        time.sleep(0.01)
        conv2 = create_conversation(title="Conv 2", db_path=temp_db)
        time.sleep(0.01)
        conv3 = create_conversation(title="Conv 3", db_path=temp_db)

        # Limit 2
        recent_two = list_conversations(limit=2, db_path=temp_db)
        assert len(recent_two) == 2
        assert recent_two[0]["conversation_id"] == conv3["conversation_id"]
        assert recent_two[1]["conversation_id"] == conv2["conversation_id"]

        # All 3
        all_convs = list_conversations(limit=10, db_path=temp_db)
        assert len(all_convs) == 3

    def test_add_user_message(self, temp_db: Path) -> None:
        """Test appending a user message to a conversation."""
        conv = create_conversation(title="Chat", db_path=temp_db)
        msg = add_message(
            conversation_id=conv["conversation_id"],
            role="user",
            content="What is the current trend for TCS?",
            db_path=temp_db,
        )

        assert isinstance(msg["message_id"], str)
        assert msg["conversation_id"] == conv["conversation_id"]
        assert msg["role"] == "user"
        assert msg["content"] == "What is the current trend for TCS?"
        assert msg["created_at"] is not None

    def test_add_assistant_message(self, temp_db: Path) -> None:
        """Test appending an assistant response message to a conversation."""
        conv = create_conversation(title="Chat", db_path=temp_db)
        add_message(
            conversation_id=conv["conversation_id"],
            role="user",
            content="Hello",
            db_path=temp_db,
        )
        asst_msg = add_message(
            conversation_id=conv["conversation_id"],
            role="assistant",
            content="TCS is currently showing bullish momentum above its 20 DMA.",
            db_path=temp_db,
        )

        assert asst_msg["role"] == "assistant"
        assert "bullish" in asst_msg["content"]

    def test_retrieve_ordered_messages(self, temp_db: Path) -> None:
        """Test messages are returned in chronological order (oldest to newest)."""
        conv = create_conversation(title="Multi-turn", db_path=temp_db)
        cid = conv["conversation_id"]

        add_message(cid, "user", "Msg 1", db_path=temp_db)
        time.sleep(0.01)
        add_message(cid, "assistant", "Msg 2", db_path=temp_db)
        time.sleep(0.01)
        add_message(cid, "user", "Msg 3", db_path=temp_db)

        messages = get_messages(cid, db_path=temp_db)
        assert len(messages) == 3
        assert [m["content"] for m in messages] == ["Msg 1", "Msg 2", "Msg 3"]
        assert [m["role"] for m in messages] == ["user", "assistant", "user"]

    def test_conversation_updated_at(self, temp_db: Path) -> None:
        """Test conversation updated_at changes when messages are added or title is updated."""
        conv = create_conversation(title="Initial Title", db_path=temp_db)
        initial_updated_at = conv["updated_at"]

        # 1. Update on new message
        time.sleep(0.02)
        add_message(conv["conversation_id"], "user", "Ping", db_path=temp_db)

        after_msg = get_conversation(conv["conversation_id"], db_path=temp_db)
        assert after_msg is not None
        assert after_msg["updated_at"] > initial_updated_at

        # 2. Update on title rename
        time.sleep(0.02)
        updated = update_conversation_title(
            conv["conversation_id"], "Renamed Title", db_path=temp_db
        )
        assert updated is True

        after_rename = get_conversation(conv["conversation_id"], db_path=temp_db)
        assert after_rename is not None
        assert after_rename["title"] == "Renamed Title"
        assert after_rename["updated_at"] > after_msg["updated_at"]

    def test_delete_conversation(self, temp_db: Path) -> None:
        """Test deleting a conversation removes it and returns True."""
        conv = create_conversation(title="To Delete", db_path=temp_db)
        cid = conv["conversation_id"]

        deleted = delete_conversation(cid, db_path=temp_db)
        assert deleted is True

        # Should no longer exist
        assert get_conversation(cid, db_path=temp_db) is None

        # Deleting non-existent conversation returns False
        assert delete_conversation("non-existent-id", db_path=temp_db) is False

    def test_foreign_key_behavior(self, temp_db: Path) -> None:
        """Test foreign key enforcement: cascading message deletion and rejection of invalid parent IDs."""
        conv = create_conversation(title="Parent", db_path=temp_db)
        cid = conv["conversation_id"]

        # Add messages under conversation
        add_message(cid, "user", "Child 1", db_path=temp_db)
        add_message(cid, "assistant", "Child 2", db_path=temp_db)
        assert len(get_messages(cid, db_path=temp_db)) == 2

        # 1. Cascade delete: deleting parent conversation cascades to messages
        delete_conversation(cid, db_path=temp_db)

        # Directly verify messages table is empty
        conn = sqlite3.connect(temp_db)
        conn.execute("PRAGMA foreign_keys = ON;")
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (cid,))
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 0

        # 2. Foreign key rejection: inserting message with invalid conversation_id raises IntegrityError
        with pytest.raises(sqlite3.IntegrityError):
            add_message(
                conversation_id="non-existent-uuid-1234",
                role="user",
                content="Orphan message",
                db_path=temp_db,
            )
