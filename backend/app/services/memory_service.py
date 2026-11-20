import os
import re
import time
import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional
from groq import AsyncGroq
from google import genai
from app.core.config import get_settings

# ---------------------------------------------------------------------------
# Memory Extraction Prompt (for background post-session deep synthesis)
# ---------------------------------------------------------------------------
MEMORY_EXTRACTION_SYSTEM_PROMPT = """
You are an expert learning memory extractor for Alex, an AI English-speaking friend.
Your task is to analyze a conversation transcript and extract persistent long-term knowledge about the user.

CATEGORIES TO EXTRACT:
1. User Profile:
   - educational_background: e.g. "Second-year CSE student", "High school graduate"
   - learning_goals: e.g. "Enter the AI field", "Prepare for job interviews", "Speak fluently"
   - skill_level: e.g. "Beginner in Python", "Intermediate English speaker"
   - preferences: e.g. "Prefers brief explanations", "Likes practical code examples"
2. Key Facts:
   - Specific personal facts the user explicitly stated (e.g. college major, target job, hobbies, location).
3. Learning Progress:
   - topics_practised: specific concepts or subjects discussed in this session.
   - concepts_understood: concepts the user showed comfort or competence with.
   - concepts_struggled: concepts or topics the user found difficult.
   - practice_recommendations: areas needing future practice.

RULES:
- Only extract information the user EXPLICITLY stated or demonstrated.
- Do NOT assume, invent, or extrapolate facts.
- If a field is unknown, leave it empty or null.
- Ignore transient chatter ("I am eating an apple", "The weather is sunny").

OUTPUT FORMAT — respond strictly with valid JSON only:
{
  "educational_background": "string or null",
  "learning_goals": "string or null",
  "skill_level": "string or null",
  "preferences": "string or null",
  "key_facts": [
    {"fact_key": "educational_background | learning_subject | career_goal | interest", "fact_value": "extracted value"}
  ],
  "topics_practised": ["string"],
  "concepts_understood": ["string"],
  "concepts_struggled": ["string"],
  "practice_recommendations": ["string"]
}
"""

class MemoryService:
    """
    Learning Memory Vault for Alex.
    Provides SQLite-backed, persistent multi-session memory across 4 core categories:
      A. User Profile (education, goals, skill level, preferences)
      B. Long-Term Grammar Memory (repeated errors, patterns, improvements)
      C. Learning Progress (topics practised, mastered, and needing practice)
      D. Session History (structured summaries across sessions)
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path:
            self.db_path = Path(db_path)
        else:
            # Default to project_root/data/memory.db
            base_dir = Path(__file__).resolve().parent.parent.parent
            self.db_path = base_dir / "data" / "memory.db"

        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a thread-safe connection configured with WAL mode for performance."""
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _init_db(self):
        """Initialises the persistent SQLite schema for all memory categories."""
        try:
            with self._get_connection() as conn:
                # 1. User Profile Table
                conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profile (
                    user_id TEXT PRIMARY KEY,
                    name TEXT,
                    educational_background TEXT,
                    learning_goals TEXT,
                    skill_level TEXT,
                    preferences TEXT,
                    created_at REAL,
                    updated_at REAL
                );
                """)

                # 2. Key-Value User Facts Table (for rapid, pinpoint semantic recall)
                conn.execute("""
                CREATE TABLE IF NOT EXISTS user_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    fact_key TEXT NOT NULL,
                    fact_value TEXT NOT NULL,
                    category TEXT DEFAULT 'general',
                    source_session_id TEXT,
                    created_at REAL,
                    updated_at REAL,
                    UNIQUE(user_id, fact_key)
                );
                """)

                # 3. Long-Term Grammar Memory Table (genuine errors only)
                conn.execute("""
                CREATE TABLE IF NOT EXISTS grammar_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    pattern_description TEXT,
                    error_count INTEGER DEFAULT 1,
                    last_example TEXT,
                    has_improved INTEGER DEFAULT 0,
                    first_seen_at REAL,
                    last_seen_at REAL,
                    UNIQUE(user_id, category)
                );
                """)

                # 4. Learning Progress Table
                conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_progress (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    status TEXT DEFAULT 'practised',
                    details TEXT,
                    created_at REAL,
                    updated_at REAL,
                    UNIQUE(user_id, topic)
                );
                """)

                # 5. Session History Table (compact summaries, avoiding full transcript bloat)
                conn.execute("""
                CREATE TABLE IF NOT EXISTS session_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    user_id TEXT NOT NULL,
                    topics_discussed TEXT,
                    learning_progress_summary TEXT,
                    unresolved_difficulties TEXT,
                    user_decisions_or_preferences TEXT,
                    turn_count INTEGER DEFAULT 0,
                    created_at REAL
                );
                """)

                # 6. Verbatim Conversation Messages Archive (Exact multi-session turn storage)
                conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL, -- 'user' or 'assistant'
                    content TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    created_at REAL NOT NULL
                );
                """)

                # Indices for fast lookups
                conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_user ON user_facts(user_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_grammar_user ON grammar_memory(user_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_progress_user ON learning_progress(user_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_session_user ON session_history(user_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_conv_user ON conversation_messages(user_id, created_at DESC);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_conv_session ON conversation_messages(session_id, turn_index ASC);")
                conn.commit()
        except Exception as e:
            print(f"[MemoryService] Database init error: {e}")

    # ---------------------------------------------------------------------------
    # Retrieval Methods (For Context Injection & Exact Recall)
    # ---------------------------------------------------------------------------

    def detect_exact_recall_request(self, user_text: str) -> bool:
        """
        Determines whether the user is asking to recall past questions, dialogue turns, or past answers.
        """
        if not user_text:
            return False
        t = user_text.lower().strip()
        patterns = [
            r"\blast (?:two|three|four|five|\d+)?\s*(?:questions?|messages?|things?)\b",
            r"\bprevious (?:questions?|messages?|answers?)\b",
            r"\bwhat (?:did|have) i (?:ask|say|talked?)\b",
            r"\bwhat was my (?:last|previous|first) question\b",
            r"\bshow (?:me )?(?:my )?(?:last|previous) (?:questions?|messages?)\b",
            r"\bwhat (?:questions?|answers?) did (?:i|you)\b",
            r"\bwhat answer did you (?:give|say)\b",
            r"\bwhat was (?:your|the) answer\b",
            r"\brepeat what (?:i|you) said\b",
            r"\bremind me (?:what|of)\b",
            r"\bearlier question\b",
            r"\byesterday\b.*\b(?:ask|say|question)\b"
        ]
        return any(re.search(p, t) for p in patterns)

    def get_exact_recent_user_questions(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieves exact verbatim user questions in chronological order."""
        try:
            with self._get_connection() as conn:
                rows = conn.execute(
                    "SELECT id, session_id, content, turn_index, created_at FROM conversation_messages "
                    "WHERE user_id = ? AND role = 'user' ORDER BY created_at DESC LIMIT ?",
                    (user_id, limit)
                ).fetchall()
                results = [dict(r) for r in rows]
                results.reverse()
                return results
        except Exception as e:
            print(f"[MemoryService] get_exact_recent_user_questions error: {e}")
            return []

    def get_exact_recent_messages(self, user_id: str, limit: int = 12) -> List[Dict[str, Any]]:
        """Retrieves exact verbatim recent messages (user and assistant) in chronological order."""
        try:
            with self._get_connection() as conn:
                rows = conn.execute(
                    "SELECT id, session_id, role, content, turn_index, created_at FROM conversation_messages "
                    "WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                    (user_id, limit)
                ).fetchall()
                results = [dict(r) for r in rows]
                results.reverse()
                return results
        except Exception as e:
            print(f"[MemoryService] get_exact_recent_messages error: {e}")
            return []

    def record_message(self, session_id: str, user_id: str, role: str, content: str, turn_index: int = 0):
        """Persists a verbatim message to the permanent SQLite conversation archive."""
        if not user_id or not content or not content.strip():
            return
        now = time.time()
        try:
            with self._get_connection() as conn:
                conn.execute("""
                INSERT INTO conversation_messages (session_id, user_id, role, content, turn_index, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (session_id, user_id, role, content.strip(), turn_index, now))
                conn.commit()
        except Exception as e:
            print(f"[MemoryService] record_message error: {e}")

    def get_user_memory_context(self, user_id: str, current_query: str = "") -> str:
        """
        Builds a compact, high-signal memory context block formatted for LLM system prompt injection.
        Provides Alex with instant recall of user profile, facts, grammar focus, learning journey,
        and verbatim conversation retrieval when asked about past questions or answers.
        """
        if not user_id:
            return ""

        try:
            profile = self.get_user_profile(user_id)
            facts = self.get_user_facts(user_id)
            grammar_items = self.get_active_grammar_weaknesses(user_id, limit=3)
            progress_items = self.get_learning_progress(user_id, limit=5)
            last_session = self.get_last_session_summary(user_id)
            is_recall = self.detect_exact_recall_request(current_query)

            has_data = any([
                profile.get("educational_background"),
                profile.get("learning_goals"),
                profile.get("skill_level"),
                facts,
                grammar_items,
                progress_items,
                last_session,
                is_recall
            ])

            if not has_data:
                return ""

            lines = ["## RETRIEVED USER MEMORY (PERSISTENT LEARNING VAULT)"]
            lines.append("You remember the following verified facts and learning history about this user from previous interactions:")

            # 1. Profile line
            prof_parts = []
            if profile.get("educational_background"):
                prof_parts.append(f"Background: {profile['educational_background']}")
            if profile.get("learning_goals"):
                prof_parts.append(f"Goals: {profile['learning_goals']}")
            if profile.get("skill_level"):
                prof_parts.append(f"Level: {profile['skill_level']}")
            if profile.get("preferences"):
                prof_parts.append(f"Preferences: {profile['preferences']}")
            if prof_parts:
                lines.append(f"- User Profile: {' | '.join(prof_parts)}")

            # 2. Key Facts
            if facts:
                lines.append("- Known Facts:")
                for f in facts[:6]:
                    k_clean = f["fact_key"].replace("_", " ").title()
                    lines.append(f"  * {k_clean}: {f['fact_value']}")

            # 3. Long-term Grammar Focus
            if grammar_items:
                lines.append("- Long-Term Grammar Focus (errors user frequently makes):")
                for g in grammar_items:
                    imp_note = " (Showing improvement)" if g["has_improved"] else ""
                    lines.append(f"  * {g['category']} (occurred {g['error_count']}x){imp_note}: {g['pattern_description'] or ''}")

            # 4. Learning Progress
            if progress_items:
                lines.append("- Learning Progress & Topics:")
                for p in progress_items:
                    lines.append(f"  * {p['topic']} [{p['status'].replace('_', ' ').title()}]: {p.get('details', '')}")

            # 5. Last Session Context
            if last_session:
                summary_text = last_session.get("topics_discussed") or last_session.get("learning_progress_summary") or ""
                if summary_text:
                    lines.append(f"- Last Session Summary: {summary_text}")

            # 6. Exact Verbatim Conversation Log (for Exact Historical Question/Answer Retrieval)
            if is_recall:
                recent_user_qs = self.get_exact_recent_user_questions(user_id, limit=8)
                recent_all_msgs = self.get_exact_recent_messages(user_id, limit=10)
                lines.append("")
                lines.append("## EXACT VERBATIM CONVERSATION ARCHIVE (RETRIEVED FROM DATABASE):")
                lines.append("The user is asking to recall past questions or messages. Quote or list their exact words below without guessing:")
                if recent_user_qs:
                    lines.append("Verbatim User Questions (Chronological Order):")
                    for idx, q in enumerate(recent_user_qs, 1):
                        lines.append(f"  {idx}. \"{q['content']}\"")
                if recent_all_msgs:
                    lines.append("Recent Exact Dialogue Flow:")
                    for m in recent_all_msgs:
                        speaker = "User" if m["role"] == "user" else "Alex"
                        lines.append(f"  * {speaker}: \"{m['content']}\"")
                lines.append("EXACT RECALL RULE: Quote the user's exact words from this archive. Never paraphrase or fabricate questions!")

            lines.append("")
            lines.append("MEMORY BEHAVIOUR RULES:")
            lines.append("1. If the user asks what they are learning, what you remember about them, or their background, answer accurately using these remembered facts.")
            lines.append("2. Adapt your tone and explanation depth to their recorded skill level and background without making every response a lecture.")
            lines.append("3. NEVER invent or hallucinate unverified personal memories. If information was not stored, honestly say you don't know yet.")
            lines.append("4. Do NOT awkwardly recite memory unprompted in every sentence; weave it in naturally only when relevant.")

            return "\n".join(lines)

        except Exception as e:
            print(f"[MemoryService] get_user_memory_context error: {e}")
            return ""

    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Retrieves user profile data."""
        try:
            with self._get_connection() as conn:
                row = conn.execute("SELECT * FROM user_profile WHERE user_id = ?", (user_id,)).fetchone()
                if row:
                    return dict(row)
        except Exception as e:
            print(f"[MemoryService] get_user_profile error: {e}")
        return {}

    def get_user_facts(self, user_id: str) -> List[Dict[str, Any]]:
        """Retrieves individual stored user facts sorted by most recently updated."""
        try:
            with self._get_connection() as conn:
                rows = conn.execute(
                    "SELECT fact_key, fact_value, category, updated_at FROM user_facts WHERE user_id = ? ORDER BY updated_at DESC",
                    (user_id,)
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            print(f"[MemoryService] get_user_facts error: {e}")
            return []

    def get_active_grammar_weaknesses(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieves top recurring grammar mistakes for coaching focus."""
        try:
            with self._get_connection() as conn:
                rows = conn.execute(
                    "SELECT category, pattern_description, error_count, last_example, has_improved "
                    "FROM grammar_memory WHERE user_id = ? ORDER BY error_count DESC, last_seen_at DESC LIMIT ?",
                    (user_id, limit)
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            print(f"[MemoryService] get_active_grammar_weaknesses error: {e}")
            return []

    def get_learning_progress(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieves learning progress topics."""
        try:
            with self._get_connection() as conn:
                rows = conn.execute(
                    "SELECT topic, status, details, updated_at FROM learning_progress WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
                    (user_id, limit)
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            print(f"[MemoryService] get_learning_progress error: {e}")
            return []

    def get_last_session_summary(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves the most recent session summary."""
        try:
            with self._get_connection() as conn:
                row = conn.execute(
                    "SELECT topics_discussed, learning_progress_summary, unresolved_difficulties, created_at "
                    "FROM session_history WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
                    (user_id,)
                ).fetchone()
                if row:
                    return dict(row)
        except Exception as e:
            print(f"[MemoryService] get_last_session_summary error: {e}")
        return None

    def get_all_user_memory(self, user_id: str) -> Dict[str, Any]:
        """Returns the full memory vault representation for transparency and inspection."""
        return {
            "user_id": user_id,
            "profile": self.get_user_profile(user_id),
            "facts": self.get_user_facts(user_id),
            "grammar_memory": self.get_active_grammar_weaknesses(user_id, limit=20),
            "learning_progress": self.get_learning_progress(user_id, limit=20),
            "recent_sessions": self.get_recent_sessions(user_id, limit=5),
            "recent_messages": self.get_exact_recent_messages(user_id, limit=20)
        }

    def get_recent_sessions(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                rows = conn.execute(
                    "SELECT session_id, topics_discussed, learning_progress_summary, unresolved_difficulties, turn_count, created_at "
                    "FROM session_history WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                    (user_id, limit)
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            print(f"[MemoryService] get_recent_sessions error: {e}")
            return []

    # ---------------------------------------------------------------------------
    # Mutation & Storage Methods (Safe, Idempotent, Thread-Safe)
    # ---------------------------------------------------------------------------

    def upsert_user_fact(self, user_id: str, fact_key: str, fact_value: str, category: str = "general", session_id: str = ""):
        """Stores or updates a specific verified user fact."""
        if not user_id or not fact_key or not fact_value:
            return
        fact_key = fact_key.strip().lower().replace(" ", "_")
        fact_value = fact_value.strip()
        now = time.time()
        try:
            with self._get_connection() as conn:
                conn.execute("""
                INSERT INTO user_facts (user_id, fact_key, fact_value, category, source_session_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, fact_key) DO UPDATE SET
                    fact_value = excluded.fact_value,
                    category = excluded.category,
                    source_session_id = excluded.source_session_id,
                    updated_at = excluded.updated_at;
                """, (user_id, fact_key, fact_value, category, session_id, now, now))
                conn.commit()
        except Exception as e:
            print(f"[MemoryService] upsert_user_fact error: {e}")

    def update_user_profile(self, user_id: str, updates: Dict[str, Any]):
        """Updates core user profile fields."""
        if not user_id or not updates:
            return
        now = time.time()
        try:
            with self._get_connection() as conn:
                # Ensure row exists
                conn.execute(
                    "INSERT OR IGNORE INTO user_profile (user_id, created_at, updated_at) VALUES (?, ?, ?)",
                    (user_id, now, now)
                )

                allowed = ["name", "educational_background", "learning_goals", "skill_level", "preferences"]
                set_clauses = []
                values = []
                for k in allowed:
                    if k in updates and updates[k] is not None:
                        val = str(updates[k]).strip()
                        if val:
                            set_clauses.append(f"{k} = ?")
                            values.append(val)

                if set_clauses:
                    set_clauses.append("updated_at = ?")
                    values.append(now)
                    values.append(user_id)
                    query = f"UPDATE user_profile SET {', '.join(set_clauses)} WHERE user_id = ?"
                    conn.execute(query, tuple(values))
                    conn.commit()
        except Exception as e:
            print(f"[MemoryService] update_user_profile error: {e}")

    def record_grammar_errors(self, user_id: str, corrections: List[Dict[str, Any]]):
        """
        Records genuine grammar errors to long-term grammar memory.
        Excludes stylistic expressions or non-grammar suggestions.
        """
        if not user_id or not corrections:
            return
        now = time.time()
        try:
            with self._get_connection() as conn:
                for c in corrections:
                    # STRICT RULE: only record genuine grammar mistakes
                    corr_type = str(c.get("type", "")).lower()
                    if corr_type != "grammar":
                        continue

                    explanation = str(c.get("explanation", "")).strip()
                    orig = str(c.get("original", "")).strip()
                    corr = str(c.get("corrected", "")).strip()

                    # Derive a clean grammar category
                    category = self._classify_grammar_category(orig, corr, explanation)
                    pattern_desc = explanation or f"Mistake: '{orig}' -> '{corr}'"
                    last_example = f"{orig} -> {corr}"

                    conn.execute("""
                    INSERT INTO grammar_memory (user_id, category, pattern_description, error_count, last_example, has_improved, first_seen_at, last_seen_at)
                    VALUES (?, ?, ?, 1, ?, 0, ?, ?)
                    ON CONFLICT(user_id, category) DO UPDATE SET
                        error_count = error_count + 1,
                        pattern_description = CASE WHEN excluded.pattern_description != '' THEN excluded.pattern_description ELSE pattern_description END,
                        last_example = excluded.last_example,
                        has_improved = 0,
                        last_seen_at = excluded.last_seen_at;
                    """, (user_id, category, pattern_desc, last_example, now, now))

                conn.commit()
        except Exception as e:
            print(f"[MemoryService] record_grammar_errors error: {e}")

    def record_learning_progress(self, user_id: str, topic: str, status: str, details: str = ""):
        """Upserts learning progress on a topic (practised, understood, struggling)."""
        if not user_id or not topic:
            return
        topic = topic.strip().title()
        now = time.time()
        try:
            with self._get_connection() as conn:
                conn.execute("""
                INSERT INTO learning_progress (user_id, topic, status, details, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, topic) DO UPDATE SET
                    status = excluded.status,
                    details = CASE WHEN excluded.details != '' THEN excluded.details ELSE details END,
                    updated_at = excluded.updated_at;
                """, (user_id, topic, status, details.strip(), now, now))
                conn.commit()
        except Exception as e:
            print(f"[MemoryService] record_learning_progress error: {e}")

    def save_session_summary(self, session_id: str, user_id: str, summary_data: Dict[str, Any], turn_count: int = 0):
        """Saves a structured, high-level summary of a completed conversation."""
        if not session_id or not user_id:
            return
        now = time.time()
        topics = summary_data.get("conversational_overview") or summary_data.get("topics_discussed", "")
        progress = summary_data.get("learning_progress_summary") or summary_data.get("fluency_celebration", "")
        unresolved = summary_data.get("unresolved_difficulties") or summary_data.get("practice_advice", "")
        decisions = summary_data.get("user_decisions_or_preferences", "")

        try:
            with self._get_connection() as conn:
                conn.execute("""
                INSERT OR REPLACE INTO session_history
                (session_id, user_id, topics_discussed, learning_progress_summary, unresolved_difficulties, user_decisions_or_preferences, turn_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """, (session_id, user_id, topics, progress, unresolved, decisions, turn_count, now))
                conn.commit()
        except Exception as e:
            print(f"[MemoryService] save_session_summary error: {e}")

    # ---------------------------------------------------------------------------
    # Real-Time & Post-Session Information Extraction
    # ---------------------------------------------------------------------------

    def extract_turn_memories(self, user_id: str, session_id: str, user_text: str, assistant_text: str, corrections: List[Dict[str, Any]]):
        """
        Runs non-blocking in the background after each turn to extract high-confidence user facts.
        Uses instant, robust heuristic pattern matching: zero latency, zero extra LLM token cost.
        """
        if not user_id or not user_text:
            return

        text = user_text.strip()
        text_lower = text.lower()

        # 1. Record genuine grammar errors
        if corrections:
            self.record_grammar_errors(user_id, corrections)

        # 2. Extract Educational Background
        # Matches: "I'm a second-year CSE student", "I am a college student studying CS", "I'm in my final year of BTech"
        edu_match = re.search(
            r"\b(?:i(?:'m| am)|currently)(?: a| an)? ([a-z0-9\-]+(?: (?:year|semester|sem))? [a-z0-9\- ]+?(?:student|graduate|fresher|undergrad|engineering student|b\.?tech|degree))\b",
            text_lower
        )
        if edu_match:
            edu_val = edu_match.group(1).title()
            self.upsert_user_fact(user_id, "educational_background", edu_val, category="profile", session_id=session_id)
            self.update_user_profile(user_id, {"educational_background": edu_val})

        # 3. Extract What the User is Currently Learning / Practising
        # Matches: "I am learning Python", "I'm studying data science", "currently practising SQL"
        learning_match = re.search(
            r"\b(?:i(?:'m| am)|currently) (?:learning|practising|practicing|studying|working with|focusing on) ([a-z0-9\+#\.\- ]+?)(?: because|\.|,|\band\b|$)",
            text_lower
        )
        if learning_match:
            learning_topic = learning_match.group(1).strip().title()
            # Filter trivial noise words
            if len(learning_topic) > 1 and learning_topic.lower() not in ["english", "speaking", "to speak", "a lot"]:
                self.upsert_user_fact(user_id, "currently_learning", learning_topic, category="progress", session_id=session_id)
                self.record_learning_progress(user_id, learning_topic, status="in_progress", details=f"User stated: '{text[:80]}'")

        # 4. Extract Career Goals & Targets
        # Matches: "I want to enter the AI field", "my goal is to become a machine learning engineer", "want to work in data analytics"
        goal_match = re.search(
            r"\b(?:i want to|my goal is to|planning to|aiming to) (?:enter|join|get into|become|work in|work as|switch to) (?:the )?([a-z0-9\- ]+?)(?: field| industry| role|\.|,|\band\b|$)",
            text_lower
        )
        if goal_match:
            goal_target = goal_match.group(1).strip().title()
            if len(goal_target) > 2:
                full_goal = f"Enter the {goal_target} field" if "field" not in goal_target.lower() else goal_target
                self.upsert_user_fact(user_id, "career_goal", full_goal, category="profile", session_id=session_id)
                self.update_user_profile(user_id, {"learning_goals": full_goal})

        # 5. Extract Skill Level Disclosures
        # Matches: "I am a beginner in Python", "new to programming", "advanced in Java"
        level_match = re.search(
            r"\b(?:i(?:'m| am)|i consider myself)(?: a| an)? (beginner|intermediate|advanced|fresher|newbie|new to [a-z0-9\+#\.\- ]+)\b",
            text_lower
        )
        if level_match:
            level_val = level_match.group(1).title()
            self.upsert_user_fact(user_id, "skill_level", level_val, category="profile", session_id=session_id)
            self.update_user_profile(user_id, {"skill_level": level_val})

        # 6. Extract Explanation Preferences
        if any(p in text_lower for p in ["keep it short", "briefly", "short explanations", "concise"]):
            self.upsert_user_fact(user_id, "explanation_preference", "Prefers concise, direct answers", category="preference", session_id=session_id)
            self.update_user_profile(user_id, {"preferences": "Prefers concise, direct answers"})
        elif any(p in text_lower for p in ["in detail", "detailed explanation", "step by step"]):
            self.upsert_user_fact(user_id, "explanation_preference", "Prefers detailed, step-by-step explanations", category="preference", session_id=session_id)
            self.update_user_profile(user_id, {"preferences": "Prefers detailed, step-by-step explanations"})

    async def process_session_completion(
        self,
        user_id: str,
        session_id: str,
        report: Dict[str, Any],
        transcript: List[Dict[str, Any]]
    ):
        """
        Deep background synthesis executed when a conversation ends.
        Saves session summary, updates long-term grammar patterns, tracks improvements,
        and uses the LLM to capture any subtle user facts missed by regex heuristics.
        """
        if not user_id or not session_id:
            return

        try:
            turn_count = len([t for t in transcript if t.get("role") == "user"])

            # 1. Save session summary
            self.save_session_summary(session_id, user_id, report, turn_count=turn_count)

            # 2. Update grammar memory from consolidated recurring mistakes
            recurring = report.get("recurring_mistakes", [])
            if isinstance(recurring, list):
                now = time.time()
                with self._get_connection() as conn:
                    for rm in recurring:
                        cat = rm.get("category", "").strip()
                        if not cat:
                            continue
                        count = int(rm.get("occurrence_count", 2))
                        pattern = rm.get("underlying_pattern", "")
                        examples = rm.get("examples", [])
                        last_ex = examples[0] if examples else ""

                        conn.execute("""
                        INSERT INTO grammar_memory (user_id, category, pattern_description, error_count, last_example, has_improved, first_seen_at, last_seen_at)
                        VALUES (?, ?, ?, ?, ?, 0, ?, ?)
                        ON CONFLICT(user_id, category) DO UPDATE SET
                            error_count = error_count + excluded.error_count,
                            pattern_description = CASE WHEN excluded.pattern_description != '' THEN excluded.pattern_description ELSE pattern_description END,
                            last_example = excluded.last_example,
                            has_improved = 0,
                            last_seen_at = excluded.last_seen_at;
                        """, (user_id, cat, pattern, count, last_ex, now, now))
                    conn.commit()

            # 3. Track topics discussed from overview & advice
            overview = report.get("conversational_overview", "")
            if overview:
                self.record_learning_progress(
                    user_id,
                    topic=f"Session {session_id[:6]} Conversation",
                    status="practised",
                    details=overview[:200]
                )

            # 4. Optional Deep LLM Fact Extraction for transcript (if conversation had 2+ turns)
            if turn_count >= 2:
                await self._extract_deep_profile_facts(user_id, session_id, transcript)

        except Exception as e:
            print(f"[MemoryService] process_session_completion error: {e}")

    async def _extract_deep_profile_facts(self, user_id: str, session_id: str, transcript: List[Dict[str, Any]]):
        """Uses LLM in background to discover any nuanced facts about the user's profile/journey."""
        settings = get_settings()
        if not settings.GROQ_API_KEY and not settings.GEMINI_API_KEY:
            return

        # Prepare user dialogue text
        user_utterances = [t["content"] for t in transcript if t.get("role") == "user"]
        if not user_utterances:
            return
        dialogue_text = "\n".join([f"User: {u}" for u in user_utterances])

        try:
            raw_json = None
            if settings.GROQ_API_KEY:
                client = AsyncGroq(api_key=settings.GROQ_API_KEY.strip())
                resp = await client.chat.completions.create(
                    model=settings.GROQ_LLM_MODEL,
                    messages=[
                        {"role": "system", "content": MEMORY_EXTRACTION_SYSTEM_PROMPT},
                        {"role": "user", "content": f"Extract memory from this conversation transcript:\n\n{dialogue_text}"}
                    ],
                    max_tokens=500,
                    temperature=0.1,
                    response_format={"type": "json_object"} if "openai" not in settings.GROQ_LLM_MODEL.lower() else None
                )
                raw_json = resp.choices[0].message.content.strip()

            if raw_json:
                data = json.loads(raw_json)
                if isinstance(data, dict):
                    # Profile updates
                    profile_updates = {}
                    for field in ["educational_background", "learning_goals", "skill_level", "preferences"]:
                        if data.get(field):
                            profile_updates[field] = data[field]
                    if profile_updates:
                        self.update_user_profile(user_id, profile_updates)

                    # Key facts
                    for kf in data.get("key_facts", []):
                        if isinstance(kf, dict) and kf.get("fact_key") and kf.get("fact_value"):
                            self.upsert_user_fact(user_id, kf["fact_key"], kf["fact_value"], category="extracted", session_id=session_id)

                    # Learning topics
                    for tp in data.get("topics_practised", []):
                        if tp:
                            self.record_learning_progress(user_id, str(tp), status="practised")
                    for tp in data.get("concepts_struggled", []):
                        if tp:
                            self.record_learning_progress(user_id, str(tp), status="needs_practice")
                    for tp in data.get("concepts_understood", []):
                        if tp:
                            self.record_learning_progress(user_id, str(tp), status="understood")

        except Exception as e:
            # Memory extraction in background should never raise or impact the user
            print(f"[MemoryService] Background LLM fact extraction skipped: {e}")

    # ---------------------------------------------------------------------------
    # Internal Helpers
    # ---------------------------------------------------------------------------

    def _classify_grammar_category(self, orig: str, corr: str, explanation: str) -> str:
        """Classifies an error into a clean, reusable grammar category for memory tracking."""
        combined = f"{orig} {corr} {explanation}".lower()
        if any(w in combined for w in ["past", "yesterday", "went", "tense", "did"]):
            return "Past-Tense Verb Usage"
        if any(w in combined for w in ["plural", "singular", "subject-verb", "agree", "is/are", "was/were"]):
            return "Subject-Verb Agreement"
        if any(w in combined for w in ["preposition", "in/on/at", "recommend to", "instead of"]):
            return "Preposition Usage"
        if any(w in combined for w in ["article", "a/an", "the"]):
            return "Article Usage (A/An/The)"
        if any(w in combined for w in ["auxiliary", "word order", "can i", "do you"]):
            return "Question Word Order & Auxiliaries"
        if any(w in combined for w in ["pronoun", "he/she", "him/her"]):
            return "Pronoun Usage"
        return "Sentence Grammar & Phrasing"

# Singleton service instance
memory_service = MemoryService()
