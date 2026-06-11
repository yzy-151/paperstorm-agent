# Research Chat Agent v2.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second product mode where PaperStorm works like a conversational research chatbot that remembers prior turns, shows a sliding context window, compresses context, and auto-runs research when existing knowledge is insufficient.

**Architecture:** Keep the existing research task workflow unchanged. Add a small file-backed chat session layer that wraps `ResearchQAAgent`, persists messages under the service root, and returns context/debug payloads for the dashboard. The frontend gains a mode switch between research article workflow and chat QA workflow.

**Tech Stack:** Python file-backed service, FastAPI adapter, vanilla HTML/CSS/JS dashboard, unittest.

---

### Task 1: Chat Session Core

**Files:**
- Create: `knowledge_storm/paperstorm_chat_agent.py`
- Modify: `knowledge_storm/paperstorm_service.py`
- Test: `tests/test_paperstorm_chat_agent.py`

- [ ] Write tests for creating a chat session, asking a first question without a task, and asking a second follow-up.
- [ ] Verify the tests fail because chat session APIs do not exist.
- [ ] Implement `PaperStormChatAgent` with JSON persistence, `create_session`, `get_session`, and `send_message`.
- [ ] Wire service methods: `create_chat_session`, `get_chat_session`, `send_chat_message`.
- [ ] Verify tests pass.

### Task 2: Context Window and Compression

**Files:**
- Modify: `knowledge_storm/paperstorm_chat_agent.py`
- Test: `tests/test_paperstorm_chat_agent.py`

- [ ] Write tests that require `context_window`, `compressed_context`, `memory_context`, and `retrieval_triggered`.
- [ ] Verify the tests fail.
- [ ] Use `compress_context` from `paperstorm_memory.py` over recent chat turns.
- [ ] Persist compressed summaries in the chat session snapshot.
- [ ] Verify tests pass.

### Task 3: FastAPI Chat Routes

**Files:**
- Modify: `examples/storm_examples/paperstorm_service_api.py`
- Test: `tests/test_paperstorm_chat_agent.py`

- [ ] Write FastAPI tests for `POST /chat/sessions`, `GET /chat/sessions/{chat_id}`, and `POST /chat/sessions/{chat_id}/messages`.
- [ ] Verify the tests fail.
- [ ] Add Pydantic request models and routes.
- [ ] Verify tests pass.

### Task 4: Dashboard Dual Mode

**Files:**
- Modify: `frontend/paperstorm_dashboard/index.html`
- Modify: `frontend/paperstorm_dashboard/app.js`
- Modify: `frontend/paperstorm_dashboard/styles.css`
- Test: `tests/test_paperstorm_frontend_docs.py`

- [ ] Write frontend structure tests for `research-mode-panel`, `chat-mode-panel`, `chat-session-id`, `chat-message-list`, `chat-context-window`, and `chat-compressed-context`.
- [ ] Verify tests fail.
- [ ] Add dual mode controls and chat panel markup.
- [ ] Add JS handlers for creating sessions, sending chat messages, and rendering context/memory/debug data.
- [ ] Add CSS for a dark chatbot-style layout.
- [ ] Verify tests pass.

### Task 5: Docs, Verification, and Release

**Files:**
- Modify: `README.md`
- Modify: `docs/VERSION_PLAN.md`
- Modify: `docs/RESUME_INTERVIEW_PLAN.md`

- [ ] Document v2.1 as a conversational research agent layer.
- [ ] Run full PaperStorm service/frontend/research QA test set.
- [ ] Commit as `release v2.1 research chat agent`.
- [ ] Push `main`, `version/v2.1`, and tag `v2.1` to the user's `fork` remote.
