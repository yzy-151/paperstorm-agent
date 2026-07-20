import json
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional


class ResearchQAAgent:
    """Research + grounded QA coordinator for PaperStorm task artifacts."""

    def __init__(self, task_service):
        self.task_service = task_service

    def ask(
        self,
        question: str,
        topic: Optional[str] = None,
        task_id: Optional[str] = None,
        mode: str = "auto",
        top_k: int = 3,
        run_mode: str = "fake",
        retriever: str = "arxiv",
        output_language: str = "zh",
        expected_keywords: Optional[List[str]] = None,
        forbidden_keywords: Optional[List[str]] = None,
        **options,
    ) -> Dict:
        question = str(question or "").strip()
        if not question:
            raise ValueError("question is required")
        trace = []
        trace.append(_event("ask_start", question=question, task_id=task_id or ""))

        if task_id:
            state = self.task_service.get_task(task_id)
            if state.get("status") != "succeeded" and mode == "auto":
                trace.append(
                    _event(
                        "research_run_required",
                        reason="existing task is not succeeded",
                        task_id=task_id,
                        task_status=state.get("status", ""),
                    )
                )
                state = self.task_service.run_task(task_id)
                retrieval_triggered = True
                decision = {
                    "action": "retrieve_then_answer",
                    "reason": "existing task was not finished",
                }
            else:
                retrieval_triggered = False
                decision = {
                    "action": "answer_from_existing_kb",
                    "reason": "finished task_id provided",
                }
            used_task_id = task_id
        else:
            task_topic = topic or question
            trace.append(
                _event(
                    "research_task_submit",
                    reason="no task_id provided",
                    topic=task_topic,
                    run_mode=run_mode,
                )
            )
            state = self.task_service.submit_research_task(
                topic=task_topic,
                retriever=retriever,
                output_language=output_language,
                run_mode=run_mode,
                expected_keywords=expected_keywords or [],
                forbidden_keywords=forbidden_keywords or [],
                **options,
            )
            used_task_id = state["task_id"]
            trace.append(_event("research_task_run", task_id=used_task_id))
            state = self.task_service.run_task(used_task_id)
            retrieval_triggered = True
            decision = {
                "action": "retrieve_then_answer",
                "reason": "no task_id provided",
            }

        trace.append(_event("qa_start", task_id=used_task_id))
        answer = self.task_service.query_knowledge_base(
            used_task_id,
            question=question,
            top_k=top_k,
        )
        evidence = answer.get("evidence") or []
        citations = answer.get("citations") or []
        state_keywords = _keywords_from_state(state)
        sufficiency = evaluate_evidence_sufficiency(
            question=question,
            evidence=evidence,
            citations=citations,
            topic=state.get("topic", ""),
            expected_keywords=expected_keywords or state_keywords["expected"],
            forbidden_keywords=forbidden_keywords or state_keywords["forbidden"],
        )
        if task_id and decision["action"] == "answer_from_existing_kb":
            if not sufficiency["sufficient"]:
                decision = {
                    "action": "reject_low_confidence",
                    "reason": sufficiency["reason"],
                }
                answer_text = "已有调研材料不足以可靠回答该问题，请补充检索或创建新的调研任务。"
                citations = []
                evidence = []
            else:
                answer_text = answer.get("answer", "")
        else:
            answer_text = answer.get("answer", "")
        grounded = bool(answer.get("grounded")) and decision["action"] != "reject_low_confidence"
        trace.append(
            _event(
                "qa_end",
                task_id=used_task_id,
                grounded=grounded,
                evidence_count=len(evidence),
                citation_count=len(citations),
                evidence_sufficiency=sufficiency,
            )
        )

        result = {
            "question": question,
            "answer": answer_text,
            "citations": citations,
            "evidence": evidence,
            "grounded": grounded,
            "memory_context": answer.get("memory_context", {}),
            "used_task_id": used_task_id,
            "task_status": state.get("status", ""),
            "retrieval_triggered": retrieval_triggered,
            "decision": decision,
            "evidence_sufficiency": sufficiency,
            "trace": trace,
        }
        history = _append_qa_history(
            self.task_service.get_task(used_task_id),
            result,
        )
        result["qa_history"] = history[-5:]
        result["qa_history_count"] = len(history)
        return result


def evaluate_evidence_sufficiency(
    question: str,
    evidence: List[Dict],
    citations: List[Dict],
    topic: str = "",
    expected_keywords: Optional[List[str]] = None,
    forbidden_keywords: Optional[List[str]] = None,
) -> Dict:
    expected_keywords = expected_keywords or []
    forbidden_keywords = forbidden_keywords or []
    evidence_text = "\n".join(
        "{0}\n{1}".format(item.get("title", ""), item.get("content", ""))
        for item in evidence
    )
    combined = "{0}\n{1}\n{2}".format(question, topic, evidence_text)
    question_terms = _terms(question)
    evidence_terms = _terms(evidence_text)
    topic_terms = _terms(topic)
    keyword_overlap = sorted(question_terms & evidence_terms)
    topic_overlap = sorted((question_terms | topic_terms) & evidence_terms)
    meaningful_overlap = _meaningful_overlap(question, evidence_text)
    topic_anchor_overlap = _meaningful_overlap(question, topic)
    expected_hits = _keyword_hits(evidence_text, expected_keywords)
    forbidden_hits = _keyword_hits(combined, forbidden_keywords)
    evidence_count = len(evidence)
    citation_count = len(citations)
    has_relevance_signal = bool(meaningful_overlap) or bool(keyword_overlap) or _has_domain_anchor(
        question,
        topic,
        expected_hits,
        expected_keywords,
    )

    score = 0
    score += min(30, evidence_count * 15)
    score += min(20, citation_count * 10)
    score += min(25, len(keyword_overlap) * 8)
    score += min(15, len(topic_overlap) * 5)
    score += min(10, len(expected_hits) * 5)
    if has_relevance_signal and expected_hits:
        score += 20
    if forbidden_hits and not expected_hits:
        score -= 15
    score = max(0, min(100, score))
    is_disambiguation_question = bool(forbidden_hits and expected_hits and "pim" in question.lower())
    # Question-relevance gate: the question must share a meaningful term with
    # the evidence AND (when the session declares a topic) must also connect to
    # that topic. Otherwise the existing knowledge base is about something else
    # and we must not answer from it — escalate to fresh research instead.
    question_relevant = bool(meaningful_overlap) and (
        not topic or bool(topic_anchor_overlap) or is_disambiguation_question
    )
    sufficient = (
        evidence_count > 0
        and citation_count > 0
        and question_relevant
        and (
            (score >= 55 and has_relevance_signal)
            or (score >= 40 and bool(keyword_overlap) and has_relevance_signal)
            or is_disambiguation_question
        )
    )
    if not sufficient:
        reason = "insufficient evidence for the question"
    elif forbidden_hits:
        reason = "sufficient evidence, but forbidden keywords were mentioned for disambiguation"
    else:
        reason = "sufficient evidence from existing knowledge base"
    return {
        "sufficient": sufficient,
        "score": round(score, 2),
        "reason": reason,
        "evidence_count": evidence_count,
        "citation_count": citation_count,
        "keyword_overlap": keyword_overlap,
        "topic_relevance": round(min(1.0, len(topic_overlap) / max(1, len(question_terms))), 4),
        "has_relevance_signal": has_relevance_signal,
        "meaningful_overlap": meaningful_overlap,
        "topic_anchor_overlap": topic_anchor_overlap,
        "expected_keyword_hits": expected_hits,
        "forbidden_keyword_hits": forbidden_hits,
    }


def _event(event: str, **payload):
    return {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
    }


def _keywords_from_state(state: Dict):
    return {
        "expected": state.get("expected_keywords") or [],
        "forbidden": state.get("forbidden_keywords") or [],
    }


def _terms(text: str):
    terms = set(re.findall(r"[a-zA-Z0-9_\-]+|[\u4e00-\u9fff]", str(text).lower()))
    stop_terms = {
        "的",
        "了",
        "和",
        "与",
        "或",
        "是",
        "在",
        "有",
        "什",
        "么",
        "如",
        "何",
        "为",
        "不",
        "能",
        "把",
        "这",
        "里",
        "次",
        "调",
        "研",
        "指",
        "关",
        "系",
    }
    return {term for term in terms if term not in stop_terms}


def _meaningful_overlap(question: str, evidence_text: str):
    """Word / CJK-bigram overlap between question and evidence; single CJK
    characters are noise (e.g. one shared char must not make a PIM KB look
    relevant to a Muon-optimizer question)."""
    from .paperstorm_retrieval_runtime import meaningful_terms

    return sorted(meaningful_terms(question) & meaningful_terms(evidence_text))


def _keyword_hits(text: str, keywords: List[str]):
    lowered = str(text).lower()
    return [keyword for keyword in keywords if keyword and keyword.lower() in lowered]


def _has_domain_anchor(
    question: str,
    topic: str,
    expected_hits: List[str],
    expected_keywords: List[str],
):
    lowered_question = question.lower()
    if _keyword_hits(question, expected_keywords):
        return True
    if "pim" in lowered_question and expected_hits:
        return True
    return False


def _append_qa_history(state: Dict, result: Dict):
    output_dir = state.get("output_dir", "")
    if not output_dir:
        return []
    from pathlib import Path

    path = Path(output_dir) / "qa_history.json"
    if path.exists():
        try:
            history = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            history = []
    else:
        history = []
    history.append(
        {
            "question": result.get("question", ""),
            "answer": result.get("answer", ""),
            "grounded": result.get("grounded", False),
            "decision": result.get("decision", {}),
            "evidence_sufficiency": result.get("evidence_sufficiency", {}),
            "citation_count": len(result.get("citations") or []),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    return history
