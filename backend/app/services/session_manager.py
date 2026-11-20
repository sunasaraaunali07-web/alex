import time
import uuid
from typing import Optional, List, Dict, Any

class Session:
    def __init__(self, session_id: str, user_id: str = "default_user"):
        self.session_id: str = session_id
        self.user_id: str = user_id
        self.created_at: float = time.time()
        self.updated_at: float = time.time()
        self.status: str = "active"  # "active", "analyzing", "completed", "too_short", "failed"
        
        # COMPLETE full session transcript (never truncated, used for post-call feedback)
        self.full_transcript: List[Dict[str, Any]] = []
        
        # Live conversation dialogue history
        self.messages: List[Dict[str, Any]] = []

class SessionManager:
    """
    In-memory session registry for managing multi-turn dialogue context and post-session analysis.
    Note: Sessions are stored in memory and are ephemeral (lost on backend restart).
    """
    def __init__(self, max_sessions: int = 200, session_ttl_seconds: int = 7200):
        self._sessions: Dict[str, Session] = {}
        self.max_sessions = max_sessions
        self.session_ttl_seconds = session_ttl_seconds

    def cleanup_old_sessions(self):
        """Prunes sessions older than TTL (default 2 hours) or if exceeding capacity."""
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if (now - s.updated_at) > self.session_ttl_seconds
        ]
        for sid in expired:
            del self._sessions[sid]

        if len(self._sessions) > self.max_sessions:
            sorted_sids = sorted(self._sessions.keys(), key=lambda k: self._sessions[k].updated_at)
            for sid in sorted_sids[: len(self._sessions) - self.max_sessions]:
                del self._sessions[sid]

    def get_or_create(self, session_id: Optional[str] = None, user_id: Optional[str] = None) -> Session:
        self.cleanup_old_sessions()
        uid = user_id or "default_user"
        if not session_id or session_id not in self._sessions:
            new_id = session_id or str(uuid.uuid4())
            self._sessions[new_id] = Session(new_id, user_id=uid)
            return self._sessions[new_id]
        
        session = self._sessions[session_id]
        if user_id and session.user_id == "default_user":
            session.user_id = user_id
        return session

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def add_turn(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str,
        asr_metadata: Optional[Dict[str, Any]] = None,
        corrections: Optional[List[Dict[str, Any]]] = None
    ) -> Session:
        """
        Records one full conversational turn to both stores.

        Args:
            user_text:      The user's original transcribed speech (never modified).
            assistant_text: Alex's natural conversational reply (response_text only,
                            NOT including the spoken correction lecture). This keeps
                            LLM context clean and conversational.
            asr_metadata:   Whisper acoustic confidence scores.
            corrections:    Real-time grammar corrections given this turn, stored in
                            full_transcript for final analysis (not in messages).
        """
        session = self.get_or_create(session_id)
        session.updated_at = time.time()

        now = time.time()

        # --- Full transcript entry (complete record for analysis) ---
        user_turn_full = {
            "role": "user",
            "content": user_text,
            "asr_metadata": asr_metadata or {},
            "corrections": corrections or [],   # Real-time corrections stored here
            "timestamp": now
        }
        assistant_turn_full = {
            "role": "assistant",
            "content": assistant_text,
            "timestamp": now
        }

        # --- Live messages entry (clean conversation for LLM context) ---
        # Corrections are intentionally excluded so the history stays conversational.
        user_turn_msg = {
            "role": "user",
            "content": user_text,
            "timestamp": now
        }
        assistant_turn_msg = {
            "role": "assistant",
            "content": assistant_text,  # Only the natural reply, not the correction lecture
            "timestamp": now
        }

        # 1. Append to complete historical transcript (never pruned, used for analysis)
        session.full_transcript.append(user_turn_full)
        session.full_transcript.append(assistant_turn_full)

        # 2. Append to live context window (sliding window for LLM)
        session.messages.append(user_turn_msg)
        session.messages.append(assistant_turn_msg)

        return session

    def get_dialogue_history_for_llm(self, session_id: str, window: int = 30) -> List[Dict[str, str]]:
        """
        Returns dialogue turns formatted for conversational LLM context.
        Uses a generous window of 30 messages (15 full conversational turns)
        to prevent losing foundational user facts early in the chat.
        """
        session = self.get(session_id)
        if not session or not session.messages:
            return []
        
        # Take the last N messages
        recent = session.messages[-window:]
        return [{"role": m["role"], "content": m["content"]} for m in recent]

    def get_complete_transcript(self, session_id: str) -> List[Dict[str, Any]]:
        """Returns the complete, un-truncated transcript of the entire session."""
        session = self.get(session_id)
        if not session:
            return []
        return list(session.full_transcript)


session_manager = SessionManager()
