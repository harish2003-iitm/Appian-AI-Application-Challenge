"""SQLite database operations for document and citation storage"""

import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Optional, Any

class SQLiteDB:
    def __init__(self, db_path: str = "data/knowledge_base.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize database tables"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Documents table - stores original documents
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                doc_type TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Chunks table - stores document chunks with provenance
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                page_number INTEGER,
                paragraph_number INTEGER,
                char_start INTEGER,
                char_end INTEGER,
                metadata TEXT,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )
        ''')

        # Search history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_context TEXT NOT NULL,
                query TEXT NOT NULL,
                results_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Citations table - tracks which citations were used
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS citations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                search_id INTEGER,
                chunk_id INTEGER NOT NULL,
                relevance_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (search_id) REFERENCES search_history(id),
                FOREIGN KEY (chunk_id) REFERENCES chunks(id)
            )
        ''')

        # Conversation threads table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_threads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                case_type TEXT,
                state TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Messages table - stores conversation messages per thread
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (thread_id) REFERENCES conversation_threads(id)
            )
        ''')

        conn.commit()
        conn.close()

    def add_document(self, title: str, doc_type: str, content: str, metadata: str = None) -> int:
        """Add a new document to the database"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO documents (title, doc_type, content, metadata) VALUES (?, ?, ?, ?)",
            (title, doc_type, content, metadata)
        )
        doc_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return doc_id

    def add_chunk(self, document_id: int, chunk_index: int, content: str,
                  page_number: int = None, paragraph_number: int = None,
                  char_start: int = None, char_end: int = None, metadata: str = None) -> int:
        """Add a document chunk with provenance information"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO chunks (document_id, chunk_index, content, page_number,
                              paragraph_number, char_start, char_end, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (document_id, chunk_index, content, page_number, paragraph_number,
              char_start, char_end, metadata))
        chunk_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return chunk_id

    def get_document(self, doc_id: int) -> Optional[Dict]:
        """Get a document by ID"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def get_chunk(self, chunk_id: int) -> Optional[Dict]:
        """Get a chunk by ID with document info"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT c.*, d.title as doc_title, d.doc_type
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.id = ?
        ''', (chunk_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def get_all_documents(self) -> List[Dict]:
        """Get all documents"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_chunks_by_document(self, document_id: int) -> List[Dict]:
        """Get all chunks for a document"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index",
            (document_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def log_search(self, case_context: str, query: str, results_count: int) -> int:
        """Log a search query"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO search_history (case_context, query, results_count) VALUES (?, ?, ?)",
            (case_context, query, results_count)
        )
        search_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return search_id

    def add_citation(self, search_id: int, chunk_id: int, relevance_score: float) -> int:
        """Add a citation record"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO citations (search_id, chunk_id, relevance_score) VALUES (?, ?, ?)",
            (search_id, chunk_id, relevance_score)
        )
        citation_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return citation_id

    def get_search_history(self, limit: int = 10) -> List[Dict]:
        """Get recent search history"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM search_history ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def clear_all_data(self):
        """Clear all data from the database (for testing)"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM citations")
        cursor.execute("DELETE FROM search_history")
        cursor.execute("DELETE FROM chunks")
        cursor.execute("DELETE FROM documents")
        conn.commit()
        conn.close()

    # ============ Conversation Thread Methods ============

    def create_thread(self, title: str, case_type: str = None, state: str = None) -> int:
        """Create a new conversation thread"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO conversation_threads (title, case_type, state) VALUES (?, ?, ?)",
            (title, case_type, state)
        )
        thread_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return thread_id

    def get_all_threads(self, limit: int = 20) -> List[Dict]:
        """Get all conversation threads, most recent first"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT t.*,
                   (SELECT COUNT(*) FROM messages WHERE thread_id = t.id) as message_count,
                   (SELECT content FROM messages WHERE thread_id = t.id AND role = 'user' ORDER BY created_at LIMIT 1) as first_query
            FROM conversation_threads t
            ORDER BY t.updated_at DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_thread(self, thread_id: int) -> Optional[Dict]:
        """Get a thread by ID"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM conversation_threads WHERE id = ?", (thread_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def update_thread(self, thread_id: int, title: str = None, case_type: str = None, state: str = None):
        """Update thread details"""
        conn = self._get_connection()
        cursor = conn.cursor()
        updates = ["updated_at = CURRENT_TIMESTAMP"]
        params = []

        if title:
            updates.append("title = ?")
            params.append(title)
        if case_type:
            updates.append("case_type = ?")
            params.append(case_type)
        if state:
            updates.append("state = ?")
            params.append(state)

        params.append(thread_id)
        cursor.execute(f"UPDATE conversation_threads SET {', '.join(updates)} WHERE id = ?", params)
        conn.commit()
        conn.close()

    def delete_thread(self, thread_id: int):
        """Delete a thread and all its messages"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))
        cursor.execute("DELETE FROM conversation_threads WHERE id = ?", (thread_id,))
        conn.commit()
        conn.close()

    def add_message(self, thread_id: int, role: str, content: str, sources: str = None) -> int:
        """Add a message to a thread"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (thread_id, role, content, sources) VALUES (?, ?, ?, ?)",
            (thread_id, role, content, sources)
        )
        message_id = cursor.lastrowid
        # Update thread's updated_at
        cursor.execute("UPDATE conversation_threads SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (thread_id,))
        conn.commit()
        conn.close()
        return message_id

    def get_thread_messages(self, thread_id: int) -> List[Dict]:
        """Get all messages for a thread"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM messages WHERE thread_id = ? ORDER BY created_at ASC",
            (thread_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
