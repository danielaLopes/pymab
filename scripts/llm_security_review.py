#!/usr/bin/env python3
"""Run an optional LLM-backed security review over tracked source files."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import Any

API_URL = "https://api.openai.com/v1/responses"
DEFAULT_MODEL = "gpt-5.2"
DEFAULT_OUTPUT = Path("reports/security/llm-security-review.md")
DEFAULT_JSON_OUTPUT = Path("reports/security/llm-security-review.json")
DEFAULT_MAX_CHARS = 180_000
SEVERITY_ORDER = {"low": 1, "medium": 2, "high": 3, "critical": 4}
REVIEW_SUFFIXES = {".py", ".toml", ".yml", ".yaml", ".sh"}
REVIEW_NAMES = {"Makefile"}
SKIP_PARTS = {".git", ".venv", ".uv-cache", "build", "dist", "reports"}


@dataclass(frozen=True)
class ReviewFile:
    path: Path
    content: str


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--fail-on",
        choices=tuple(SEVERITY_ORDER),
        default=os.getenv("LLM_SECURITY_FAIL_ON", "high"),
        help="Fail when a finding at this severity or higher has medium/high confidence.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=int(os.getenv("LLM_SECURITY_MAX_CHARS", str(DEFAULT_MAX_CHARS))),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument(
        "--required",
        action="store_true",
        default=os.getenv("LLM_SECURITY_REQUIRED") == "1",
        help="Fail instead of skipping when OPENAI_API_KEY is missing.",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        message = "OPENAI_API_KEY is not set; skipping LLM security review."
        if args.required:
            print(message, file=sys.stderr)
            return 2
        print(message)
        return 0

    files = collect_review_files(max_chars=args.max_chars)
    if not files:
        print("No reviewable files found.", file=sys.stderr)
        return 2

    review = request_review(
        api_key=api_key,
        model=args.model,
        repository_snapshot=render_snapshot(files),
    )
    write_reports(review, output=args.output, json_output=args.json_output)

    findings = review.get("findings", [])
    blocking = [
        finding for finding in findings if should_block(finding, fail_on=args.fail_on)
    ]
    print(
        f"LLM security review wrote {args.output} and found {len(findings)} finding(s)."
    )
    if blocking:
        print(
            f"Failing on {len(blocking)} finding(s) at {args.fail_on}+ severity "
            "with medium/high confidence.",
            file=sys.stderr,
        )
        return 1
    return 0


def collect_review_files(*, max_chars: int) -> list[ReviewFile]:
    paths = tracked_paths()
    selected: list[ReviewFile] = []
    used_chars = 0
    for path in paths:
        if not is_reviewable(path):
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        entry_size = len(content) + len(str(path)) + 64
        if used_chars + entry_size > max_chars:
            continue
        selected.append(ReviewFile(path=path, content=content))
        used_chars += entry_size
    return selected


def tracked_paths() -> list[Path]:
    git = which("git")
    if git is None:
        raise RuntimeError("git executable was not found on PATH")
    result = subprocess.run(  # noqa: S603 - fixed git command, no shell.
        [git, "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [Path(line) for line in result.stdout.splitlines() if line.strip()]


def is_reviewable(path: Path) -> bool:
    if any(part in SKIP_PARTS for part in path.parts):
        return False
    return path.name in REVIEW_NAMES or path.suffix in REVIEW_SUFFIXES


def render_snapshot(files: list[ReviewFile]) -> str:
    chunks = []
    for item in files:
        chunks.append(f"### {item.path}\n```text\n{item.content}\n```")
    return "\n\n".join(chunks)


def request_review(
    *, api_key: str, model: str, repository_snapshot: str
) -> dict[str, Any]:
    if not API_URL.startswith("https://api.openai.com/"):
        raise RuntimeError("API_URL must point to the OpenAI HTTPS API endpoint.")
    payload = {
        "model": model,
        "max_output_tokens": 6000,
        "instructions": (
            "You are a senior application security reviewer. Review the supplied "
            "Python library source for exploitable security vulnerabilities, unsafe "
            "CI/tooling behavior, credential leaks, injection risks, unsafe file or "
            "process handling, dependency-audit blind spots, and supply-chain risks. "
            "Do not report style issues, theoretical issues without a plausible "
            "attack path, or issues already covered by the static tooling unless "
            "there is concrete project-specific evidence. Prefer false negatives "
            "over noisy false positives."
        ),
        "input": (
            "Review this repository snapshot. Return only findings that include a "
            "specific file path, evidence, exploit scenario, severity, confidence, "
            "and remediation. If no meaningful vulnerabilities are present, return "
            "an empty findings list.\n\n"
            f"{repository_snapshot}"
        ),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "security_review",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["summary", "findings"],
                    "properties": {
                        "summary": {"type": "string"},
                        "findings": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": [
                                    "title",
                                    "severity",
                                    "confidence",
                                    "file",
                                    "line",
                                    "cwe",
                                    "evidence",
                                    "exploit_scenario",
                                    "recommendation",
                                ],
                                "properties": {
                                    "title": {"type": "string"},
                                    "severity": {
                                        "type": "string",
                                        "enum": ["low", "medium", "high", "critical"],
                                    },
                                    "confidence": {
                                        "type": "string",
                                        "enum": ["low", "medium", "high"],
                                    },
                                    "file": {"type": "string"},
                                    "line": {"type": ["integer", "null"]},
                                    "cwe": {"type": ["string", "null"]},
                                    "evidence": {"type": "string"},
                                    "exploit_scenario": {"type": "string"},
                                    "recommendation": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            }
        },
    }
    request = urllib.request.Request(  # noqa: S310 - constant HTTPS OpenAI API URL.
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(  # noqa: S310 - validated constant HTTPS endpoint.
            request, timeout=120
        ) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API request failed: {exc.code} {detail}") from exc

    output_text = body.get("output_text")
    if not isinstance(output_text, str):
        output_text = extract_output_text(body)
    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Model did not return valid JSON: {output_text}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("Model returned a non-object JSON response.")
    return parsed


def extract_output_text(body: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in body.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                parts.append(content["text"])
    return "\n".join(parts)


def should_block(finding: dict[str, Any], *, fail_on: str) -> bool:
    severity = str(finding.get("severity", "")).lower()
    confidence = str(finding.get("confidence", "")).lower()
    return SEVERITY_ORDER.get(severity, 0) >= SEVERITY_ORDER[
        fail_on
    ] and confidence in {"medium", "high"}


def write_reports(review: dict[str, Any], *, output: Path, json_output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")
    findings = review.get("findings", [])
    lines = ["# LLM Security Review", "", str(review.get("summary", "")).strip(), ""]
    if not findings:
        lines.append("No reportable findings.")
    else:
        for index, finding in enumerate(findings, start=1):
            lines.extend(
                [
                    f"## {index}. {finding.get('title', 'Untitled finding')}",
                    "",
                    f"- Severity: `{finding.get('severity')}`",
                    f"- Confidence: `{finding.get('confidence')}`",
                    f"- Location: `{finding.get('file')}:{finding.get('line')}`",
                    f"- CWE: `{finding.get('cwe')}`",
                    "",
                    f"Evidence: {finding.get('evidence')}",
                    "",
                    f"Exploit scenario: {finding.get('exploit_scenario')}",
                    "",
                    f"Recommendation: {finding.get('recommendation')}",
                    "",
                ]
            )
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
