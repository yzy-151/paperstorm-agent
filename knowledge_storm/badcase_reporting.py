"""Auditable milestone manifests and bad-case dossiers."""

import hashlib
import json
import os
import random
import re
import statistics
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


MILESTONES = ("P1", "P1+P2", "P1+P2+P3", "P1+P2+P3+P4")
_SENSITIVE_KEY = re.compile(
    r"(?:^|_)(?:api_?)?(?:secret|token|key|password)(?:$|_)", re.I
)
_SENSITIVE_OPTION = re.compile(
    r"(?:--?(?:api[-_]?key|access[-_]?token|secret|token))(?:\s+|=)\S+",
    re.I,
)


def validate_milestone(milestone):
    if milestone not in MILESTONES:
        raise ValueError("milestone must be one of: {0}".format(", ".join(MILESTONES)))
    return milestone


@dataclass(frozen=True)
class CaseDossier:
    case_id: str
    milestone: str
    question: str
    before: object
    root_cause: object
    change: object
    after: object
    residual_risk: object = ""

    def to_dict(self):
        validate_milestone(self.milestone)
        return json.loads(json.dumps(asdict(self), ensure_ascii=False))


def write_case_dossiers(path, dossiers):
    lines = []
    for dossier in dossiers:
        row = dossier.to_dict() if isinstance(dossier, CaseDossier) else dossier
        lines.append(json.dumps(row, ensure_ascii=False, sort_keys=True))
    _atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


write_case_dossiers_jsonl = write_case_dossiers


def build_milestone_manifest(
    *,
    milestone,
    git_sha,
    dataset_path,
    split,
    models,
    top_k,
    seed,
    command,
    started_at,
    finished_at,
    api_usage,
    host_profile,
    dataset_digest=None
):
    validate_milestone(milestone)
    path = Path(dataset_path)
    manifest = {
        "milestone": milestone,
        "git_sha": str(git_sha),
        "dataset_path": str(dataset_path),
        "dataset_digest": str(dataset_digest or _path_digest(path)),
        "split": split,
        "models": models,
        "top_k": int(top_k),
        "seed": int(seed),
        "command": command,
        "started_at": started_at,
        "finished_at": finished_at,
        "api_usage": api_usage,
        "host_profile": host_profile,
    }
    return _sanitize(manifest)


def write_milestone_manifest(path, manifest):
    safe_manifest = _sanitize(manifest)
    payload = json.dumps(
        safe_manifest, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    _atomic_write_text(path, payload)


def paired_bootstrap_ci(baseline, candidate, samples=2000, seed=55):
    baseline = list(baseline)
    candidate = list(candidate)
    if not baseline or not candidate:
        raise ValueError("paired bootstrap input cannot be empty")
    if len(baseline) != len(candidate):
        raise ValueError("baseline and candidate must have the same length")
    samples = int(samples)
    if samples < 1:
        raise ValueError("samples must be positive")
    differences = [float(new) - float(old) for old, new in zip(baseline, candidate)]
    generator = random.Random(seed)
    means = sorted(
        statistics.mean(generator.choice(differences) for _ in differences)
        for _ in range(samples)
    )
    return {
        "delta": round(statistics.mean(differences), 6),
        "low": round(means[int(0.025 * (samples - 1))], 6),
        "high": round(means[int(0.975 * (samples - 1))], 6),
        "sample_count": len(differences),
    }


paired_bootstrap_confidence_interval = paired_bootstrap_ci


def _path_digest(path):
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if path.is_dir():
        for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
            digest.update(item.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
            with item.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()
    raise ValueError("dataset path does not exist: {0}".format(path))


def _sanitize(value):
    if isinstance(value, dict):
        return {
            str(key): _sanitize(item)
            for key, item in value.items()
            if not _SENSITIVE_KEY.search(str(key))
        }
    if isinstance(value, (list, tuple)):
        output = []
        redact_next = False
        for item in value:
            text = str(item)
            if redact_next:
                output.append("[REDACTED]")
                redact_next = False
            elif re.fullmatch(r"--?(?:api[-_]?key|access[-_]?token|secret|token)", text, re.I):
                output.append("[REDACTED_OPTION]")
                redact_next = True
            else:
                output.append(_sanitize(item))
        return output
    if isinstance(value, str):
        return _SENSITIVE_OPTION.sub("[REDACTED]", value)
    return value


def _atomic_write_text(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=".{0}.".format(path.name),
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    except Exception:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        raise
