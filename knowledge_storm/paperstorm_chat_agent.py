import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .paperstorm_memory import PaperStormMemoryStore, compress_context


class PaperStormChatAgent:
    """File-backed conversational layer over ResearchQAAgent."""

    def __init__(self, task_service):
        self.task_service = task_service
        self.chat_dir = Path(task_service.root_dir) / "chat_sessions"
        self.chat_dir.mkdir(parents=True, exist_ok=True)

    def create_session(
        self,
        title: str = "",
        topic: str = "",
        run_mode: str = "fake",
        retriever: str = "arxiv",
        output_language: str = "zh",
        expected_keywords: Optional[List[str]] = None,
        forbidden_keywords: Optional[List[str]] = None,
        context_window_size: int = 6,
        **options,
    ) -> Dict:
        chat_id = uuid.uuid4().hex
        session = {
            "chat_id": chat_id,
            "title": title or topic or "PaperStorm Chat",
            "topic": topic,
            "run_mode": run_mode,
            "retriever": retriever,
            "output_language": output_language,
            "expected_keywords": expected_keywords or [],
            "forbidden_keywords": forbidden_keywords or [],
            "context_window_size": max(2, int(context_window_size or 6)),
            "task_id": options.pop("task_id", "") or "",
            "messages": [],
            "compressed_context": {},
            "memory_context": {},
            "created_at": _now(),
            "updated_at": _now(),
            "options": options,
        }
        self._write_session(session)
        return session

    def get_session(self, chat_id: str) -> Dict:
        return self._read_session(chat_id)

    def send_message(self, chat_id: str, message: str) -> Dict:
        message = str(message or "").strip()
        if not message:
            raise ValueError("message is required")

        session = self._read_session(chat_id)
        previous_summary = (session.get("compressed_context") or {}).get("summary", "")
        user_message = _message("user", message)
        session["messages"].append(user_message)
        context_window = self._context_window(session)
        memory = self._build_memory(session, message)
        context_keywords = _context_keywords(session, message)
        compressed = compress_context(
            _with_previous_summary(context_window, previous_summary),
            expected_keywords=context_keywords,
            forbidden_keywords=session.get("forbidden_keywords") or [],
            max_chars=1200,
        )

        answer = self.task_service.ask_research_agent(
            question=message,
            topic=session.get("topic") or message,
            task_id=session.get("task_id") or None,
            run_mode=session.get("run_mode") or "fake",
            retriever=session.get("retriever") or "arxiv",
            output_language=session.get("output_language") or "zh",
            expected_keywords=session.get("expected_keywords") or [],
            forbidden_keywords=session.get("forbidden_keywords") or [],
        )
        session["task_id"] = answer.get("used_task_id") or session.get("task_id", "")
        assistant_message = _message(
            "assistant",
            answer.get("answer", ""),
            {
                "grounded": answer.get("grounded", False),
                "used_task_id": answer.get("used_task_id", ""),
                "retrieval_triggered": answer.get("retrieval_triggered", False),
                "decision": answer.get("decision", {}),
                "citation_count": len(answer.get("citations") or []),
            },
        )
        session["messages"].append(assistant_message)
        context_window = self._context_window(session)
        compressed = compress_context(
            _with_previous_summary(context_window, previous_summary),
            expected_keywords=context_keywords,
            forbidden_keywords=session.get("forbidden_keywords") or [],
            max_chars=1200,
        )
        memory.append_working(
            "User asked: {0}\nAgent answered: {1}".format(
                message,
                assistant_message["content"],
            ),
            {
                "chat_id": chat_id,
                "task_id": session.get("task_id", ""),
                "retrieval_triggered": answer.get("retrieval_triggered", False),
            },
        )
        memory_context = memory.get_context_bundle(query=message, max_items=5)
        session["compressed_context"] = compressed
        session["memory_context"] = memory_context
        session["updated_at"] = _now()
        self._write_session(session)

        return {
            "mode": "chat",
            "chat_id": chat_id,
            "message": user_message,
            "assistant_message": assistant_message,
            "messages": session["messages"],
            "context_window": context_window,
            "compressed_context": compressed,
            "memory_context": memory_context,
            "retrieval_triggered": answer.get("retrieval_triggered", False),
            "used_task_id": session.get("task_id", ""),
            "research_answer": answer,
        }

    def _context_window(self, session: Dict) -> List[Dict]:
        size = max(2, int(session.get("context_window_size") or 6))
        messages = session.get("messages") or []
        return messages[-size:]

    def _build_memory(self, session: Dict, query: str):
        memory = PaperStormMemoryStore()
        topic = session.get("topic") or query
        if topic:
            memory.remember_semantic(
                "当前聊天主题：{0}".format(topic),
                tags=["chat_topic"],
            )
        for keyword in session.get("expected_keywords") or []:
            memory.remember_semantic(
                "期望领域关键词：{0}".format(keyword),
                tags=["expected_keyword"],
            )
        for keyword in session.get("forbidden_keywords") or []:
            memory.remember_episode(
                "检索和回答时需要警惕跑题关键词：{0}".format(keyword),
                {"type": "forbidden_keyword"},
            )
        for message in (session.get("messages") or [])[-6:]:
            memory.append_working(
                "[{0}] {1}".format(message.get("role", ""), message.get("content", "")),
                {"chat_id": session.get("chat_id", "")},
            )
        return memory

    def _session_path(self, chat_id: str):
        return self.chat_dir / "{0}.json".format(chat_id)

    def _read_session(self, chat_id: str):
        path = self._session_path(chat_id)
        if not path.exists():
            raise KeyError("Unknown chat_id: {0}".format(chat_id))
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_session(self, session: Dict):
        self._session_path(session["chat_id"]).write_text(
            json.dumps(session, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _message(role: str, content: str, metadata: Optional[Dict] = None):
    return {
        "id": uuid.uuid4().hex,
        "role": role,
        "content": str(content or ""),
        "metadata": metadata or {},
        "created_at": _now(),
    }


def _with_previous_summary(messages: List[Dict], previous_summary: str):
    if not previous_summary:
        return messages
    return [
        {
            "role": "system",
            "content": "上一轮压缩上下文：{0}".format(previous_summary),
        }
    ] + list(messages)


def _context_keywords(session: Dict, message: str):
    keywords = list(session.get("expected_keywords") or [])
    topic = "{0} {1}".format(session.get("topic", ""), message)
    lowered = topic.lower()
    if "pim" in lowered and "PIM" not in keywords:
        keywords.append("PIM")
    for token in ["神经网络", "无源互调"]:
        if token in topic and token not in keywords:
            keywords.append(token)
    return keywords


def _now():
    return datetime.now(timezone.utc).isoformat()
