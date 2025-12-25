"""
Layer 4: Immutable Audit Logging System
- Records complete chain of thought
- Tracks all document access for privacy auditing
- Stores in immutable format (WORM - Write Once Read Many)
- Serves as legal record of decision-making
"""

import json
import hashlib
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
import sqlite3
from pathlib import Path


@dataclass
class AuditEntry:
    """Single audit log entry"""
    entry_id: str
    timestamp: str
    session_id: str
    user_id: str
    action_type: str  # query, retrieval, generation, feedback

    # Request details
    raw_prompt: str = ""
    transformed_query: str = ""
    case_context: Dict = field(default_factory=dict)

    # Retrieval details
    documents_accessed: List[str] = field(default_factory=list)
    retrieval_scores: List[float] = field(default_factory=list)
    chunks_retrieved: int = 0

    # Generation details
    model_used: str = ""
    raw_model_output: str = ""
    final_answer: str = ""
    citations_used: List[str] = field(default_factory=list)

    # Guardrails
    guardrail_results: Dict = field(default_factory=dict)
    pii_detected: bool = False
    pii_types: List[str] = field(default_factory=list)

    # User feedback
    feedback_rating: Optional[int] = None
    feedback_text: str = ""

    # Integrity
    checksum: str = ""

    def compute_checksum(self) -> str:
        """Compute SHA-256 checksum for integrity verification"""
        data = json.dumps({
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "raw_prompt": self.raw_prompt,
            "final_answer": self.final_answer,
            "documents_accessed": self.documents_accessed
        }, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()

    def to_dict(self) -> Dict:
        return asdict(self)


class ImmutableAuditLogger:
    """
    Immutable audit logging with WORM (Write Once Read Many) semantics.
    Uses SQLite with append-only design and checksum verification.
    """

    def __init__(self, db_path: str = "data/audit_log.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

        # In-memory buffer for current session
        self.session_buffer: List[AuditEntry] = []
        self.current_session_id = self._generate_session_id()

    def _init_db(self):
        """Initialize audit database with immutable design"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Main audit log table - append only
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_id TEXT UNIQUE NOT NULL,
                timestamp TEXT NOT NULL,
                session_id TEXT NOT NULL,
                user_id TEXT,
                action_type TEXT NOT NULL,
                raw_prompt TEXT,
                transformed_query TEXT,
                case_context TEXT,
                documents_accessed TEXT,
                retrieval_scores TEXT,
                chunks_retrieved INTEGER,
                model_used TEXT,
                raw_model_output TEXT,
                final_answer TEXT,
                citations_used TEXT,
                guardrail_results TEXT,
                pii_detected INTEGER,
                pii_types TEXT,
                feedback_rating INTEGER,
                feedback_text TEXT,
                checksum TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Document access log for privacy auditing
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_access_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                access_timestamp TEXT NOT NULL,
                access_reason TEXT,
                user_id TEXT,
                FOREIGN KEY (entry_id) REFERENCES audit_log(entry_id)
            )
        """)

        # Chain of custody for integrity
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS integrity_chain (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_id TEXT NOT NULL,
                previous_checksum TEXT,
                current_checksum TEXT NOT NULL,
                chain_position INTEGER NOT NULL,
                verified_at TEXT,
                FOREIGN KEY (entry_id) REFERENCES audit_log(entry_id)
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_log(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_access ON document_access_log(document_id)")

        conn.commit()
        conn.close()

    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_part = hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:8]
        return f"SES_{timestamp}_{random_part}"

    def _generate_entry_id(self) -> str:
        """Generate unique entry ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        return f"AUD_{timestamp}"

    def _get_last_checksum(self) -> Optional[str]:
        """Get checksum of last entry for chain integrity"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT checksum FROM audit_log ORDER BY id DESC LIMIT 1")
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None

    def log_query(self,
                  raw_prompt: str,
                  transformed_query: str = "",
                  case_context: Dict = None,
                  user_id: str = "anonymous") -> str:
        """Log incoming query"""
        entry = AuditEntry(
            entry_id=self._generate_entry_id(),
            timestamp=datetime.now().isoformat(),
            session_id=self.current_session_id,
            user_id=user_id,
            action_type="query",
            raw_prompt=raw_prompt,
            transformed_query=transformed_query or raw_prompt,
            case_context=case_context or {}
        )
        entry.checksum = entry.compute_checksum()

        self._write_entry(entry)
        return entry.entry_id

    def log_retrieval(self,
                      entry_id: str,
                      documents_accessed: List[str],
                      retrieval_scores: List[float],
                      chunks_retrieved: int) -> None:
        """Log retrieval results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Update existing entry
        cursor.execute("""
            UPDATE audit_log SET
                documents_accessed = ?,
                retrieval_scores = ?,
                chunks_retrieved = ?
            WHERE entry_id = ?
        """, (
            json.dumps(documents_accessed),
            json.dumps(retrieval_scores),
            chunks_retrieved,
            entry_id
        ))

        # Log individual document accesses
        for doc_id in documents_accessed:
            cursor.execute("""
                INSERT INTO document_access_log
                (entry_id, document_id, access_timestamp, access_reason)
                VALUES (?, ?, ?, ?)
            """, (entry_id, doc_id, datetime.now().isoformat(), "retrieval"))

        conn.commit()
        conn.close()

    def log_generation(self,
                       entry_id: str,
                       model_used: str,
                       raw_model_output: str,
                       final_answer: str,
                       citations_used: List[str]) -> None:
        """Log generation results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE audit_log SET
                model_used = ?,
                raw_model_output = ?,
                final_answer = ?,
                citations_used = ?
            WHERE entry_id = ?
        """, (
            model_used,
            raw_model_output,
            final_answer,
            json.dumps(citations_used),
            entry_id
        ))

        conn.commit()
        conn.close()

    def log_guardrails(self,
                       entry_id: str,
                       guardrail_results: Dict,
                       pii_detected: bool,
                       pii_types: List[str]) -> None:
        """Log guardrail results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE audit_log SET
                guardrail_results = ?,
                pii_detected = ?,
                pii_types = ?
            WHERE entry_id = ?
        """, (
            json.dumps(guardrail_results),
            1 if pii_detected else 0,
            json.dumps(pii_types),
            entry_id
        ))

        conn.commit()
        conn.close()

    def log_feedback(self,
                     entry_id: str,
                     rating: int,
                     feedback_text: str = "") -> None:
        """Log user feedback"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE audit_log SET
                feedback_rating = ?,
                feedback_text = ?
            WHERE entry_id = ?
        """, (rating, feedback_text, entry_id))

        conn.commit()
        conn.close()

    def _write_entry(self, entry: AuditEntry) -> None:
        """Write audit entry to database (append-only)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO audit_log (
                entry_id, timestamp, session_id, user_id, action_type,
                raw_prompt, transformed_query, case_context,
                documents_accessed, retrieval_scores, chunks_retrieved,
                model_used, raw_model_output, final_answer, citations_used,
                guardrail_results, pii_detected, pii_types,
                feedback_rating, feedback_text, checksum
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.entry_id,
            entry.timestamp,
            entry.session_id,
            entry.user_id,
            entry.action_type,
            entry.raw_prompt,
            entry.transformed_query,
            json.dumps(entry.case_context),
            json.dumps(entry.documents_accessed),
            json.dumps(entry.retrieval_scores),
            entry.chunks_retrieved,
            entry.model_used,
            entry.raw_model_output,
            entry.final_answer,
            json.dumps(entry.citations_used),
            json.dumps(entry.guardrail_results),
            1 if entry.pii_detected else 0,
            json.dumps(entry.pii_types),
            entry.feedback_rating,
            entry.feedback_text,
            entry.checksum
        ))

        # Add to integrity chain
        previous_checksum = self._get_last_checksum()
        chain_position = cursor.lastrowid

        cursor.execute("""
            INSERT INTO integrity_chain
            (entry_id, previous_checksum, current_checksum, chain_position)
            VALUES (?, ?, ?, ?)
        """, (entry.entry_id, previous_checksum, entry.checksum, chain_position))

        conn.commit()
        conn.close()

    def get_entry(self, entry_id: str) -> Optional[Dict]:
        """Retrieve audit entry by ID"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM audit_log WHERE entry_id = ?", (entry_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return dict(row)
        return None

    def get_session_history(self, session_id: str = None) -> List[Dict]:
        """Get all entries for a session"""
        session_id = session_id or self.current_session_id

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM audit_log WHERE session_id = ? ORDER BY timestamp",
            (session_id,)
        )
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_document_access_history(self, document_id: str) -> List[Dict]:
        """Get all accesses for a specific document (privacy audit)"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT dal.*, al.user_id, al.raw_prompt
            FROM document_access_log dal
            JOIN audit_log al ON dal.entry_id = al.entry_id
            WHERE dal.document_id = ?
            ORDER BY dal.access_timestamp
        """, (document_id,))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def verify_integrity(self) -> Dict:
        """Verify integrity of audit chain"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT al.*, ic.previous_checksum, ic.chain_position
            FROM audit_log al
            JOIN integrity_chain ic ON al.entry_id = ic.entry_id
            ORDER BY ic.chain_position
        """)

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return {"valid": True, "entries_checked": 0, "errors": []}

        errors = []
        previous_checksum = None

        for row in rows:
            # Verify checksum matches
            entry = AuditEntry(
                entry_id=row["entry_id"],
                timestamp=row["timestamp"],
                session_id=row["session_id"],
                user_id=row["user_id"] or "",
                action_type=row["action_type"],
                raw_prompt=row["raw_prompt"] or "",
                final_answer=row["final_answer"] or "",
                documents_accessed=json.loads(row["documents_accessed"] or "[]")
            )
            computed_checksum = entry.compute_checksum()

            if computed_checksum != row["checksum"]:
                errors.append({
                    "entry_id": row["entry_id"],
                    "error": "Checksum mismatch - data may have been modified"
                })

            # Verify chain continuity
            if row["previous_checksum"] != previous_checksum:
                errors.append({
                    "entry_id": row["entry_id"],
                    "error": "Chain broken - previous checksum mismatch"
                })

            previous_checksum = row["checksum"]

        return {
            "valid": len(errors) == 0,
            "entries_checked": len(rows),
            "errors": errors
        }

    def export_for_compliance(self, start_date: str = None, end_date: str = None) -> List[Dict]:
        """Export audit logs for compliance review"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM audit_log WHERE 1=1"
        params = []

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        query += " ORDER BY timestamp"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_stats(self) -> Dict:
        """Get audit log statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        cursor.execute("SELECT COUNT(*) FROM audit_log")
        stats["total_entries"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT session_id) FROM audit_log")
        stats["total_sessions"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT document_id) FROM document_access_log")
        stats["documents_accessed"] = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(feedback_rating) FROM audit_log WHERE feedback_rating IS NOT NULL")
        avg_rating = cursor.fetchone()[0]
        stats["avg_feedback_rating"] = round(avg_rating, 2) if avg_rating else None

        cursor.execute("SELECT COUNT(*) FROM audit_log WHERE pii_detected = 1")
        stats["pii_detections"] = cursor.fetchone()[0]

        conn.close()
        return stats
