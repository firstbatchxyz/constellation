"""Generative LLM labeling for canonical rollout datasets."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from constellation.categorization import task_focused_text
from constellation.io import iter_jsonl, write_jsonl
from constellation.schema import CanonicalSample
from constellation.taxonomy import CapabilityTaxonomy, DomainTaxonomy

DEFAULT_LLM_LABEL_MODEL = "Qwen/Qwen3.5-0.8B"
LLM_LABEL_METHOD = "llm_json_v2"

CODING_FRAME_CUES: tuple[str, ...] = (
    "competitive programming problem",
    "contest information",
    "sample input",
    "sample output",
)

DOMAIN_GUARDRAIL_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "CODING_SOFTWARE",
        (
            "codebase",
            "repository",
            "python repository",
            "unit test",
            "traceback",
            "pyproject",
            "package manager",
            "shell session",
            "uv run",
            "pytest",
            "module not found",
            "modulenotfounderror",
        ),
    ),
    (
        "MATHEMATICS",
        ("prove", "proof", "induction", "n^2", "theorem", "algebra", "equation", "geometry"),
    ),
    (
        "MEDICINE_HEALTH",
        (
            "patient with",
            "clinical",
            "diagnosis",
            "differential diagnosis",
            "symptom",
            "fever",
            "cough",
            "oxygen saturation",
            "pneumonia",
            "treatment",
            "physiology",
            "biomedical",
            "medical text",
        ),
    ),
    (
        "DATA_ANALYSIS",
        ("csv", "dataset", "cohort", "retention", "conversion rate", "chart", "statistics", "metric", "anomalous"),
    ),
    (
        "BUSINESS_OPERATIONS",
        (
            "customer",
            "subscription",
            "acquisition channel",
            "support triage",
            "rollout plan",
            "owners",
            "success metrics",
            "operating cadence",
            "revenue",
            "todos",
            "blockers",
            "sub-tasks",
            "subtasks",
        ),
    ),
    (
        "SOCIAL_SCIENCE",
        ("survey study", "housing voucher", "policy", "comparison group", "confounder", "causal interpretation", "school attendance", "education"),
    ),
    (
        "HUMANITIES",
        ("sophocles", "antigone", "divine law", "state authority", "civic duty", "close reading", "poem", "novel", "philosophy", "history"),
    ),
    (
        "WRITING",
        ("draft", "personal essay", "revise", "rewrite", "improve clarity", "prose", "tone", "style", "paragraph"),
    ),
    (
        "SCIENCE",
        ("combustion", "physics", "chemistry", "biology", "ocean acidification", "coral", "calcification", "experiment", "hypothesis"),
    ),
    (
        "GENERAL_KNOWLEDGE",
        ("time zones", "curious reader", "countries", "cities"),
    ),
)

REQUIRED_CAPABILITY_CUES: dict[str, tuple[str, ...]] = {
    "CODEBASE_NAVIGATION": (
        "codebase",
        "repository",
        "repo",
        "search the code",
        "renamed function",
        "symbol",
        "source code",
    ),
    "CODE_EDITING": ("patch", "modify code", "implementation", "refactor", "diff", "fix the implementation"),
    "MULTI_FILE_EDITING": ("multi-file", "multiple files", "across files", "refactor"),
    "TEST_WRITING": ("unit test", "pytest", "jest", "vitest", "test runner", "coverage", "failing test", "write tests", "add tests", "fixture", "assertion"),
    "TERMINAL_WORKFLOW": (
        "shell",
        "terminal",
        "command",
        "cli",
        "uv run",
        "bash",
        "python3 -m",
        "npm",
        "cargo",
        "axolotl",
        "configure fine-tuning",
        "fine-tuning pipeline",
    ),
    "TOOL_USE": ("tool_call", "function call", "tool response", "observation", "browser", "api call", "axolotl"),
    "RETRIEVAL_SEARCH": ("search", "literature", "sources", "cite", "lookup", "retrieve", "documentation"),
    "COMPOSITION": ("compose", "draft", "write an essay", "personal essay", "story", "narrative"),
    "REVISION": ("revise", "rewrite", "improve clarity", "tone", "style", "proofread", "edit the text"),
}

ADD_CAPABILITY_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "STRUCTURED_REASONING",
        (
            "explain why",
            "prove",
            "analyze",
            "compare",
            "evidence",
            "hypothesis",
            "differential diagnosis",
            "causal",
            "confounder",
            "derive",
            "formal",
        ),
    ),
    (
        "PLANNING",
        (
            "plan",
            "rollout plan",
            "phases",
            "owners",
            "risks",
            "cadence",
            "design a survey",
            "define treatment",
            "success metrics",
            "todos",
            "blocked",
            "blockers",
            "sub-tasks",
            "subtasks",
        ),
    ),
    (
        "TERMINAL_WORKFLOW",
        ("axolotl", "configure fine-tuning", "fine-tuning pipeline"),
    ),
    (
        "TOOL_USE",
        ("axolotl", "tool_call", "function call", "tool response", "observation"),
    ),
    (
        "DEBUGGING",
        ("debug", "failing", "fails", "traceback", "diagnose", "broken", "regression", "incorrect output", "modulenotfounderror"),
    ),
    (
        "ERROR_RECOVERY",
        ("recover", "retry", "failed", "fails", "fallback", "without discarding", "modulenotfounderror"),
    ),
)


class OptionalLLMLabelingDependencyError(RuntimeError):
    """Raised when llm-label dependencies are not installed."""


class OpenAICompatibleLabelingError(RuntimeError):
    """Raised when an OpenAI-compatible labeling server returns an error."""


def taxonomy_block(taxonomy: CapabilityTaxonomy, *, title: str) -> str:
    lines = [f"{title}. Use only these exact labels:"]
    for item in taxonomy.capabilities:
        lines.append(f"- {item.name}: {item.description}")
    return "\n".join(lines)


def build_llm_label_prompt(
    sample: CanonicalSample,
    *,
    capability_taxonomy: CapabilityTaxonomy,
    domain_taxonomy: DomainTaxonomy,
    max_chars: int,
) -> str:
    text = task_focused_text(sample, max_chars=max_chars)
    return "\n".join(
        [
            "You label ONE task trajectory for specialist model distillation.",
            "Return one strict JSON object and no surrounding prose.",
            "Ignore the order of labels in the taxonomy; it is not a prior.",
            "Do not default to CODING_SOFTWARE. Use CODING_SOFTWARE only when the task explicitly involves code, repositories, shell commands, tests, package managers, or developer tooling.",
            "",
            taxonomy_block(capability_taxonomy, title="Capability taxonomy"),
            "",
            taxonomy_block(domain_taxonomy, title="Domain taxonomy"),
            "",
            "JSON schema:",
            '{"capabilities":["LABEL"],"domains":["LABEL"],"confidence":0.0,"rationale":"short evidence"}',
            "",
            "Rules:",
            "- Multi-label both axes, but prefer a small precise set over broad coverage.",
            "- Keep rationale under 12 words.",
            "- Capabilities describe behavior being taught; domains describe subject matter.",
            "- Decide domains from the user's task topic, not from the fact this is a dataset or distillation pipeline.",
            "- Use exact labels from the taxonomies only.",
            "- Use [] when no label clearly applies on an axis.",
            "- STRUCTURED_REASONING applies to proofs, causal explanations, differential diagnosis, evidence analysis, study design, quantitative reasoning, and stepwise argumentation.",
            "- COMPOSITION/REVISION require writing or editing prose as the task, not merely prose in the answer.",
            "- TERMINAL_WORKFLOW requires shell/CLI interaction to be central, not just incidental code.",
            "- TEST_WRITING requires software tests, not a school exam, medical test, survey test, or scientific experiment.",
            "- Scientific specialists should usually be SCIENCE plus a behavioral capability such as STRUCTURED_REASONING.",
            "- Do not infer programming language or framework labels; those are intentionally absent.",
            "",
            "Domain guardrails:",
            "- Patient symptoms, diagnosis, treatment, physiology, or biomedical content => MEDICINE_HEALTH, not CODING_SOFTWARE.",
            "- CSVs, cohorts, metrics, statistics, charts, or empirical datasets => DATA_ANALYSIS; add BUSINESS_OPERATIONS only when the task is about business operations or customers.",
            "- Surveys, policy, economics, psychology, sociology, education, politics, or causal social systems => SOCIAL_SCIENCE.",
            "- History, literature, philosophy, religion, culture, or art interpretation => HUMANITIES.",
            "- Drafting, revising, style, rhetoric, essays, or prose quality => WRITING.",
            "- Physics, chemistry, biology, climate, experiments, mechanisms, or natural evidence => SCIENCE.",
            "- Algebra, proof, geometry, equations, or formal derivation => MATHEMATICS.",
            "- Broad factual explanation with no specialized domain => GENERAL_KNOWLEDGE.",
            "",
            "Calibration examples:",
            'Task: "Explain why a candle flame goes out under a jar using oxygen and combustion evidence."',
            'Output: {"capabilities":["STRUCTURED_REASONING"],"domains":["SCIENCE"],"confidence":0.9,"rationale":"causal scientific explanation"}',
            'Task: "Build a differential diagnosis from fever, cough, chest pain, and oxygen saturation."',
            'Output: {"capabilities":["STRUCTURED_REASONING"],"domains":["MEDICINE_HEALTH"],"confidence":0.9,"rationale":"clinical reasoning"}',
            'Task: "Analyze subscription cohort retention from a CSV and choose a chart."',
            'Output: {"capabilities":["STRUCTURED_REASONING"],"domains":["DATA_ANALYSIS","BUSINESS_OPERATIONS"],"confidence":0.9,"rationale":"metrics and cohort analysis"}',
            'Task: "Create a rollout plan with owners, metrics, risks, and weekly cadence."',
            'Output: {"capabilities":["PLANNING"],"domains":["BUSINESS_OPERATIONS"],"confidence":0.9,"rationale":"operational planning"}',
            'Task: "Debug a Python repo, inspect traceback, patch code, and rerun tests."',
            'Output: {"capabilities":["DEBUGGING","CODEBASE_NAVIGATION","CODE_EDITING","TEST_WRITING"],"domains":["CODING_SOFTWARE"],"confidence":0.9,"rationale":"software debugging workflow"}',
            "",
            "Trajectory metadata:",
            json.dumps(
                {
                    "id": sample.id,
                    "source_dataset": sample.source_dataset,
                    "sample_type": sample.sample_type,
                },
                sort_keys=True,
            ),
            "",
            "Trajectory to label:",
            text,
            "",
            "JSON output:",
        ]
    )


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object from a model response."""
    start = text.find("{")
    if start < 0:
        raise ValueError("response did not contain a JSON object")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                value = json.loads(text[start : index + 1])
                if not isinstance(value, dict):
                    raise ValueError("JSON payload must be an object")
                return value

    raise ValueError("response JSON object was not closed")


def _raw_decode_key(text: str, key: str) -> Any:
    key_start = text.find(f'"{key}"')
    if key_start < 0:
        return None
    colon = text.find(":", key_start)
    if colon < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(text[colon + 1 :].lstrip())
    except json.JSONDecodeError:
        return None
    return value


def extract_label_payload(text: str) -> dict[str, Any]:
    """Parse a complete label JSON object, with recovery for truncated rationale text."""
    try:
        return extract_json_object(text)
    except ValueError as exc:
        capabilities = _raw_decode_key(text, "capabilities")
        domains = _raw_decode_key(text, "domains")
        confidence = _raw_decode_key(text, "confidence")
        if isinstance(capabilities, list) and isinstance(domains, list):
            return {
                "capabilities": capabilities,
                "domains": domains,
                "confidence": confidence if confidence is not None else 0.0,
                "rationale": "",
                "_partial_json_recovery": str(exc),
            }
        raise


def _as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def contains_any(text: str, cues: tuple[str, ...]) -> bool:
    return any(cue in text for cue in cues)


def has_coding_frame(sample: CanonicalSample, text: str) -> bool:
    return sample.sample_type == "coding" or contains_any(text, CODING_FRAME_CUES)


def normalize_taxonomy_labels(
    labels: list[str],
    taxonomy: CapabilityTaxonomy,
    *,
    max_labels: int,
) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for label in labels:
        canonical = taxonomy.normalize_label(label) or (label if label in taxonomy.names else None)
        if canonical is None or canonical in seen:
            continue
        normalized.append(canonical)
        seen.add(canonical)
        if max_labels > 0 and len(normalized) >= max_labels:
            break
    return normalized


def apply_label_guardrails(
    sample: CanonicalSample,
    *,
    capabilities: list[str],
    domains: list[str],
    max_capabilities: int,
    max_domains: int,
) -> dict[str, Any]:
    text = task_focused_text(sample, max_chars=24000).lower()
    coding_frame = has_coding_frame(sample, text)
    if coding_frame:
        final_domains = ["CODING_SOFTWARE"]
    else:
        matched_domains = [
            domain
            for domain, cues in DOMAIN_GUARDRAIL_CUES
            if contains_any(text, cues)
        ]
        final_domains = matched_domains[:max_domains] if matched_domains else list(domains)

    final_capabilities: list[str] = []
    dropped_capabilities: list[str] = []
    for label in capabilities:
        required_cues = REQUIRED_CAPABILITY_CUES.get(label)
        if required_cues and not contains_any(text, required_cues):
            dropped_capabilities.append(label)
            continue
        if label not in final_capabilities:
            final_capabilities.append(label)

    added_capabilities: list[str] = []
    for label, cues in ADD_CAPABILITY_CUES:
        if label not in final_capabilities and contains_any(text, cues):
            final_capabilities.append(label)
            added_capabilities.append(label)

    if max_capabilities > 0:
        overflow = final_capabilities[max_capabilities:]
        final_capabilities = final_capabilities[:max_capabilities]
        dropped_capabilities.extend(overflow)

    return {
        "capabilities": final_capabilities,
        "domains": final_domains,
        "metadata": {
            "applied": bool(
                dropped_capabilities
                or added_capabilities
                or final_domains != domains
            ),
            "dropped_capabilities": dropped_capabilities,
            "added_capabilities": added_capabilities,
            "model_domains": domains,
            "guardrail_domains": final_domains,
            "coding_frame": coding_frame,
        },
    }


def normalize_llm_payload(
    payload: dict[str, Any],
    *,
    capability_taxonomy: CapabilityTaxonomy,
    domain_taxonomy: DomainTaxonomy,
    max_capabilities: int,
    max_domains: int,
) -> dict[str, Any]:
    confidence = payload.get("confidence", 0.0)
    try:
        confidence_value = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence_value = 0.0

    rationale = payload.get("rationale", "")
    if not isinstance(rationale, str):
        rationale = ""

    return {
        "capabilities": normalize_taxonomy_labels(
            _as_string_list(payload.get("capabilities")),
            capability_taxonomy,
            max_labels=max_capabilities,
        ),
        "domains": normalize_taxonomy_labels(
            _as_string_list(payload.get("domains")),
            domain_taxonomy,
            max_labels=max_domains,
        ),
        "confidence": confidence_value,
        "rationale": rationale[:1000],
        "partial_json_recovery": payload.get("_partial_json_recovery"),
    }


def label_sample_with_llm_response(
    sample: CanonicalSample,
    *,
    response_text: str,
    capability_taxonomy: CapabilityTaxonomy,
    domain_taxonomy: DomainTaxonomy,
    model_name: str,
    max_capabilities: int,
    max_domains: int,
    label_guardrails: bool = True,
) -> tuple[CanonicalSample, bool]:
    metadata = dict(sample.metadata)
    parsed_ok = True
    try:
        payload = extract_label_payload(response_text)
        normalized = normalize_llm_payload(
            payload,
            capability_taxonomy=capability_taxonomy,
            domain_taxonomy=domain_taxonomy,
            max_capabilities=max_capabilities,
            max_domains=max_domains,
        )
        if label_guardrails:
            guarded = apply_label_guardrails(
                sample,
                capabilities=normalized["capabilities"],
                domains=normalized["domains"],
                max_capabilities=max_capabilities,
                max_domains=max_domains,
            )
            normalized["capabilities"] = guarded["capabilities"]
            normalized["domains"] = guarded["domains"]
            normalized["guardrails"] = guarded["metadata"]
    except ValueError as exc:
        parsed_ok = False
        normalized = {
            "capabilities": [],
            "domains": [],
            "confidence": 0.0,
            "rationale": "",
            "parse_error": str(exc),
            "raw_response_preview": response_text[:500],
        }

    metadata["capability_labeling"] = {
        "taxonomy_version": capability_taxonomy.version,
        "method": LLM_LABEL_METHOD,
        "model": model_name,
        "confidence": normalized["confidence"],
        "rationale": normalized["rationale"],
        "previous_capabilities": sample.capabilities,
    }
    metadata["domain_labeling"] = {
        "taxonomy_version": domain_taxonomy.version,
        "method": LLM_LABEL_METHOD,
        "model": model_name,
        "confidence": normalized["confidence"],
        "rationale": normalized["rationale"],
        "previous_domains": sample.domains,
    }
    if not parsed_ok:
        metadata["llm_labeling_error"] = {
            "method": LLM_LABEL_METHOD,
            "model": model_name,
            "parse_error": normalized["parse_error"],
            "raw_response_preview": normalized["raw_response_preview"],
        }
    if "guardrails" in normalized:
        metadata["label_guardrails"] = {
            "method": "post_llm_guardrails_v1",
            **normalized["guardrails"],
        }
    if normalized.get("partial_json_recovery"):
        metadata["llm_labeling_recovery"] = {
            "method": LLM_LABEL_METHOD,
            "model": model_name,
            "recovery": normalized["partial_json_recovery"],
            "raw_response_preview": response_text[:500],
        }

    sample.capabilities = normalized["capabilities"]
    sample.domains = normalized["domains"]
    sample.metadata = metadata
    return sample, parsed_ok


def require_generation_model(
    model_name: str,
    *,
    device: int | None,
    dtype: str,
    trust_remote_code: bool,
) -> tuple[Any, Any, Any, str]:
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoProcessor,
            AutoTokenizer,
        )
    except ImportError as exc:
        raise OptionalLLMLabelingDependencyError(
            "llm-label requires lightweight labeling dependencies. Install on the GPU machine with:\n"
            "uv pip install -r requirements/labeling.txt"
        ) from exc

    if device is None:
        resolved_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        resolved_device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")

    if dtype == "auto":
        if resolved_device.type == "cuda" and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            torch_dtype = torch.bfloat16
        elif resolved_device.type == "cuda":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
    else:
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        backend = "causal_lm"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        processor = tokenizer
    except (OSError, ValueError):
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        backend = "image_text_to_text"
    model.to(resolved_device)
    model.eval()
    return processor, model, resolved_device, backend


def _move_inputs_to_device(inputs: Any, device: Any) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def _pad_token_id(processor: Any) -> int | None:
    tokenizers = [processor, getattr(processor, "tokenizer", None)]
    for tokenizer in tokenizers:
        if tokenizer is None:
            continue
        token_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None)
        if token_id is not None:
            return int(token_id)
    return None


def normalize_openai_base_url(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/v1/chat/completions"):
        return base[: -len("/chat/completions")]
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def openai_chat_completion_url(api_base: str) -> str:
    return f"{normalize_openai_base_url(api_base)}/chat/completions"


def openai_message_content(prompt: str, *, model_name: str, content_format: str) -> Any:
    resolved_format = content_format
    if resolved_format == "auto":
        lowered = model_name.lower()
        resolved_format = "parts" if "qwen3.5" in lowered or "vl" in lowered else "string"
    if resolved_format == "parts":
        return [{"type": "text", "text": prompt}]
    return prompt


def parse_openai_chat_response(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise OpenAICompatibleLabelingError("chat response did not contain choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise OpenAICompatibleLabelingError("chat response choice did not contain a message")
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "\n".join(parts).strip()
    raise OpenAICompatibleLabelingError("chat response message did not contain text content")


def label_response_schema(
    *,
    capability_taxonomy: CapabilityTaxonomy,
    domain_taxonomy: DomainTaxonomy,
    max_capabilities: int,
    max_domains: int,
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "capabilities": {
                "type": "array",
                "items": {"type": "string", "enum": list(capability_taxonomy.names)},
                "maxItems": max_capabilities,
                "uniqueItems": True,
            },
            "domains": {
                "type": "array",
                "items": {"type": "string", "enum": list(domain_taxonomy.names)},
                "maxItems": max_domains,
                "uniqueItems": True,
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "rationale": {
                "type": "string",
                "maxLength": 240,
            },
        },
        "required": ["capabilities", "domains", "confidence", "rationale"],
        "additionalProperties": False,
    }


def make_openai_chat_generator(
    *,
    api_base: str,
    api_key: str | None,
    model_name: str,
    max_new_tokens: int,
    request_timeout: float,
    content_format: str,
    response_schema: dict[str, Any] | None,
) -> Callable[[str], str]:
    url = openai_chat_completion_url(api_base)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    def generate(prompt: str) -> str:
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": openai_message_content(
                        prompt,
                        model_name=model_name,
                        content_format=content_format,
                    ),
                }
            ],
            "temperature": 0,
            "max_tokens": max_new_tokens,
        }
        if response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "constellation_labels",
                    "schema": response_schema,
                },
            }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=request_timeout) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise OpenAICompatibleLabelingError(
                f"OpenAI-compatible server returned HTTP {exc.code}: {body[:1000]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise OpenAICompatibleLabelingError(
                f"could not reach OpenAI-compatible server at {url}: {exc.reason}"
            ) from exc
        return parse_openai_chat_response(response_payload)

    return generate


def make_generator(
    *,
    processor: Any,
    model: Any,
    device: Any,
    backend: str,
    max_input_tokens: int,
    max_new_tokens: int,
) -> Callable[[str], str]:
    def generate(prompt: str) -> str:
        if backend == "image_text_to_text":
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ]
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = _move_inputs_to_device(inputs, device)
        else:
            messages = [{"role": "user", "content": prompt}]
            try:
                rendered = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                rendered = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except (AttributeError, ValueError):
                rendered = prompt

            inputs = processor(
                rendered,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_tokens,
            )
            inputs = _move_inputs_to_device(inputs, device)

        import torch

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }
        pad_token_id = _pad_token_id(processor)
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                **generation_kwargs,
            )
        input_length = inputs["input_ids"].shape[-1]
        return processor.decode(outputs[0][input_length:], skip_special_tokens=True).strip()

    return generate


def llm_label_jsonl(
    *,
    input_path: str | Path,
    output_path: str | Path,
    taxonomy_path: str | Path,
    domain_taxonomy_path: str | Path,
    model_name: str = DEFAULT_LLM_LABEL_MODEL,
    max_capabilities: int = 4,
    max_domains: int = 2,
    max_chars: int = 12000,
    max_input_tokens: int = 8192,
    max_new_tokens: int = 384,
    backend: str = "auto",
    api_base: str | None = None,
    api_key: str | None = None,
    request_timeout: float = 120.0,
    api_content_format: str = "auto",
    concurrency: int = 1,
    structured_output: bool = True,
    label_guardrails: bool = True,
    device: int | None = None,
    dtype: str = "auto",
    trust_remote_code: bool = False,
    limit: int | None = None,
    generator: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    capability_taxonomy = CapabilityTaxonomy.load(taxonomy_path)
    domain_taxonomy = DomainTaxonomy.load(domain_taxonomy_path)
    resolved_api_base = api_base or os.environ.get("CONSTELLATION_LLM_API_BASE") or os.environ.get("SGLANG_API_BASE")
    resolved_api_key = api_key or os.environ.get("CONSTELLATION_LLM_API_KEY") or os.environ.get("SGLANG_API_KEY")
    generator_backend = "custom"
    if generator is None:
        use_api_backend = backend in {"openai-compatible", "sglang"} or (
            backend == "auto" and resolved_api_base
        )
        if use_api_backend:
            if not resolved_api_base:
                resolved_api_base = "http://127.0.0.1:30000/v1"
            generator_backend = "sglang" if backend == "sglang" else "openai-compatible"
            response_schema = (
                label_response_schema(
                    capability_taxonomy=capability_taxonomy,
                    domain_taxonomy=domain_taxonomy,
                    max_capabilities=max_capabilities,
                    max_domains=max_domains,
                )
                if structured_output
                else None
            )
            generator = make_openai_chat_generator(
                api_base=resolved_api_base,
                api_key=resolved_api_key,
                model_name=model_name,
                max_new_tokens=max_new_tokens,
                request_timeout=request_timeout,
                content_format=api_content_format,
                response_schema=response_schema,
            )
        else:
            generator_backend = "transformers"
            processor, model, resolved_device, transformer_backend = require_generation_model(
                model_name,
                device=device,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
            generator = make_generator(
                processor=processor,
                model=model,
                device=resolved_device,
                backend=transformer_backend,
                max_input_tokens=max_input_tokens,
                max_new_tokens=max_new_tokens,
            )

    capability_counts = {label: 0 for label in capability_taxonomy.names}
    domain_counts = {label: 0 for label in domain_taxonomy.names}
    written = 0
    empty = 0
    parse_errors = 0
    effective_concurrency = max(1, concurrency if generator_backend in {"openai-compatible", "sglang"} else 1)

    def label_row(row: dict[str, Any]) -> tuple[CanonicalSample, bool]:
        sample = CanonicalSample.from_dict(row)
        prompt = build_llm_label_prompt(
            sample,
            capability_taxonomy=capability_taxonomy,
            domain_taxonomy=domain_taxonomy,
            max_chars=max_chars,
        )
        response = generator(prompt)
        return label_sample_with_llm_response(
            sample,
            response_text=response,
            capability_taxonomy=capability_taxonomy,
            domain_taxonomy=domain_taxonomy,
            model_name=model_name,
            max_capabilities=max_capabilities,
            max_domains=max_domains,
            label_guardrails=label_guardrails,
        )

    def record(sample: CanonicalSample, parsed_ok: bool) -> None:
        nonlocal written, empty, parse_errors
        if not parsed_ok:
            parse_errors += 1
        if not sample.capabilities and not sample.domains:
            empty += 1
        for label in sample.capabilities:
            capability_counts[label] = capability_counts.get(label, 0) + 1
        for label in sample.domains:
            domain_counts[label] = domain_counts.get(label, 0) + 1
        written += 1

    def rows() -> Any:
        submitted = 0
        if effective_concurrency <= 1:
            for row in iter_jsonl(input_path):
                if limit is not None and submitted >= limit:
                    break
                sample, parsed_ok = label_row(row)
                record(sample, parsed_ok)
                submitted += 1
                yield sample.to_dict()
            return

        pending = []
        with ThreadPoolExecutor(max_workers=effective_concurrency) as executor:
            for row in iter_jsonl(input_path):
                if limit is not None and submitted >= limit:
                    break
                pending.append(executor.submit(label_row, row))
                submitted += 1
                if len(pending) >= effective_concurrency:
                    sample, parsed_ok = pending.pop(0).result()
                    record(sample, parsed_ok)
                    yield sample.to_dict()
            for future in pending:
                sample, parsed_ok = future.result()
                record(sample, parsed_ok)
                yield sample.to_dict()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, rows())
    return {
        "input": str(input_path),
        "output": str(output_path),
        "model": model_name,
        "method": LLM_LABEL_METHOD,
        "backend": generator_backend,
        "api_base": normalize_openai_base_url(resolved_api_base) if resolved_api_base else None,
        "concurrency": effective_concurrency,
        "structured_output": structured_output if generator_backend in {"openai-compatible", "sglang"} else False,
        "label_guardrails": label_guardrails,
        "taxonomy_version": capability_taxonomy.version,
        "domain_taxonomy_version": domain_taxonomy.version,
        "written": written,
        "empty": empty,
        "empty_rate": round(empty / written, 4) if written else 0.0,
        "parse_errors": parse_errors,
        "max_labels": {
            "capabilities": max_capabilities,
            "domains": max_domains,
        },
        "limits": {
            "max_chars": max_chars,
            "max_input_tokens": max_input_tokens,
            "max_new_tokens": max_new_tokens,
        },
        "capability_counts": {
            key: value for key, value in capability_counts.items() if value
        },
        "domain_counts": {key: value for key, value in domain_counts.items() if value},
    }
