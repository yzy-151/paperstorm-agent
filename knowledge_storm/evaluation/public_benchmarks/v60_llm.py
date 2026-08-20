"""LiteLLM adapters with streaming TTFT, usage and explicit judge telemetry."""

import time


class StreamingReader:
    def __init__(self, model, api_key, api_base=None, max_tokens=512, timeout=120, input_price=0.0, output_price=0.0, completion=None):
        if not api_key:
            raise ValueError("reader API key is required")
        if completion is None:
            import litellm
            completion = litellm.completion
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.max_tokens = int(max_tokens)
        self.timeout = int(timeout)
        self.input_price = float(input_price)
        self.output_price = float(output_price)
        self.completion = completion

    def complete_prompt(self, prompt, profile_tokens=None):
        started = time.perf_counter()
        first_token_at = None
        pieces = []
        usage = {}
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "api_key": self.api_key,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        for chunk in self.completion(**kwargs):
            content = _chunk_content(chunk)
            if content:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                pieces.append(content)
            chunk_usage = _get(chunk, "usage", {}) or {}
            if chunk_usage:
                usage = _usage(chunk_usage)
        ended = time.perf_counter()
        if not usage:
            usage = {"prompt_tokens": max(1, len(prompt) // 4), "completion_tokens": max(1, len("".join(pieces)) // 4)}
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        return {
            "text": "".join(pieces).strip(),
            "usage": usage,
            "ttft_ms": round(((first_token_at or ended) - started) * 1000, 4),
            "latency_ms": round((ended - started) * 1000, 4),
            "cost_usd": _cost(usage, self.input_price, self.output_price),
            "profile_tokens": profile_tokens,
        }

    def __call__(self, question, evidence, mode):
        prompt = (
            "Answer the question using only the supplied chat sessions. Give the shortest "
            "factual answer; use Unanswerable only when evidence is absent. Do not explain.\n\n"
            "Retrieval mode: {0}\nQuestion: {1}\n\nSessions:\n{2}\n\nAnswer:"
        ).format(mode, question, evidence or "No session retrieved.")
        return self.complete_prompt(prompt)


class LongMemEvalJudge:
    """Binary LLM judge compatible with the public LongMemEval evaluation shape."""

    protocol = "longmemeval-official-qa-compatible"

    def __init__(self, model, api_key, api_base=None, timeout=120, input_price=0.0, output_price=0.0, completion=None):
        if not api_key:
            raise ValueError("judge API key is required")
        if completion is None:
            import litellm
            completion = litellm.completion
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.timeout = int(timeout)
        self.input_price = float(input_price)
        self.output_price = float(output_price)
        self.completion = completion

    def __call__(self, question, gold, prediction, question_type):
        prompt = _official_judge_prompt(question, gold, prediction, question_type)
        started = time.perf_counter()
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "api_key": self.api_key,
            "temperature": 0.0,
            "max_tokens": 10,
            "timeout": self.timeout,
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        response = self.completion(**kwargs)
        choice = _get(response, "choices", [])[0]
        message = _get(choice, "message", {}) or {}
        text = str(_get(message, "content", "") or "").strip()
        usage = _usage(_get(response, "usage", {}) or {})
        return {
            "correct": "yes" in text.lower(),
            "explanation": text,
            "usage": usage,
            "latency_ms": round((time.perf_counter() - started) * 1000, 4),
            "cost_usd": _cost(usage, self.input_price, self.output_price),
        }


def _chunk_content(chunk):
    choices = _get(chunk, "choices", []) or []
    if not choices:
        return ""
    delta = _get(choices[0], "delta", {}) or {}
    return str(_get(delta, "content", "") or "")


def _official_judge_prompt(question, gold, prediction, question_type):
    kind = str(question_type or "")
    if kind == "abstention":
        instruction = "Decide whether the response correctly recognizes that the question cannot be answered from the history."
        reference_label = "Why it is unanswerable"
    elif kind == "temporal-reasoning":
        instruction = "Decide whether the response is equivalent to the correct answer. Accept a one-unit date-duration error. Reject incomplete answers."
        reference_label = "Correct answer"
    elif kind == "knowledge-update":
        instruction = "Decide whether the response contains the latest correct fact. Older facts may also appear, but the updated answer must be present."
        reference_label = "Correct answer"
    elif kind == "single-session-preference":
        instruction = "Decide whether the response uses the remembered user preference appropriately; it need not mention every rubric item."
        reference_label = "Preference rubric"
    else:
        instruction = "Decide whether the response contains an equivalent complete answer or all steps needed to derive it. Reject partial answers."
        reference_label = "Correct answer"
    return (
        "{0} Reply yes or no only.\n\nQuestion: {1}\n\n{2}: {3}\n\nModel response: {4}\n\nCorrect?"
    ).format(instruction, question, reference_label, gold, prediction)


def _usage(value):
    prompt = int(_get(value, "prompt_tokens", 0) or _get(value, "input_tokens", 0) or 0)
    completion = int(_get(value, "completion_tokens", 0) or _get(value, "output_tokens", 0) or 0)
    total = int(_get(value, "total_tokens", 0) or prompt + completion)
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


def _cost(usage, input_price, output_price):
    return round((usage.get("prompt_tokens", 0) * input_price + usage.get("completion_tokens", 0) * output_price) / 1_000_000, 8)


def _get(value, key, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)
