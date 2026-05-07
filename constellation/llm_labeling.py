"""Generative LLM labeling for canonical rollout datasets."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from constellation.categorization import task_focused_text
from constellation.io import iter_jsonl, write_jsonl
from constellation.schema import CanonicalSample
from constellation.taxonomy import CapabilityTaxonomy, DomainTaxonomy

DEFAULT_LLM_LABEL_MODEL = "Qwen/Qwen3-0.6B"


class OptionalLLMLabelingDependencyError(RuntimeError):
    """Raised when llm-label dependencies are not installed."""


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
            "You label rollout trajectories for specialist model distillation.",
            "Return one strict JSON object and no surrounding prose.",
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
            "- Capabilities describe behavior being taught; domains describe subject matter.",
            "- Use exact labels from the taxonomies only.",
            "- Use [] when no label clearly applies on an axis.",
            "- COMPOSITION/REVISION require writing or editing prose as the task, not merely prose in the answer.",
            "- TERMINAL_WORKFLOW requires shell/CLI interaction to be central, not just incidental code.",
            "- Scientific specialists should usually be SCIENCE plus a behavioral capability such as STRUCTURED_REASONING.",
            "- Do not infer programming language or framework labels; those are intentionally absent.",
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


def _as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


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
) -> tuple[CanonicalSample, bool]:
    metadata = dict(sample.metadata)
    parsed_ok = True
    try:
        payload = extract_json_object(response_text)
        normalized = normalize_llm_payload(
            payload,
            capability_taxonomy=capability_taxonomy,
            domain_taxonomy=domain_taxonomy,
            max_capabilities=max_capabilities,
            max_domains=max_domains,
        )
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
        "method": "llm_json_v1",
        "model": model_name,
        "confidence": normalized["confidence"],
        "rationale": normalized["rationale"],
        "previous_capabilities": sample.capabilities,
    }
    metadata["domain_labeling"] = {
        "taxonomy_version": domain_taxonomy.version,
        "method": "llm_json_v1",
        "model": model_name,
        "confidence": normalized["confidence"],
        "rationale": normalized["rationale"],
        "previous_domains": sample.domains,
    }
    if not parsed_ok:
        metadata["llm_labeling_error"] = {
            "method": "llm_json_v1",
            "model": model_name,
            "parse_error": normalized["parse_error"],
            "raw_response_preview": normalized["raw_response_preview"],
        }

    sample.capabilities = normalized["capabilities"]
    sample.domains = normalized["domains"]
    sample.metadata = metadata
    return sample, parsed_ok


def require_causal_lm(
    model_name: str,
    *,
    device: int | None,
    dtype: str,
    trust_remote_code: bool,
) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
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

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(resolved_device)
    model.eval()
    return tokenizer, model, resolved_device


def make_generator(
    *,
    tokenizer: Any,
    model: Any,
    device: Any,
    max_input_tokens: int,
    max_new_tokens: int,
) -> Callable[[str], str]:
    def generate(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except (AttributeError, ValueError):
            rendered = prompt

        inputs = tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        import torch

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        input_length = inputs["input_ids"].shape[-1]
        return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()

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
    device: int | None = None,
    dtype: str = "auto",
    trust_remote_code: bool = False,
    limit: int | None = None,
    generator: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    capability_taxonomy = CapabilityTaxonomy.load(taxonomy_path)
    domain_taxonomy = DomainTaxonomy.load(domain_taxonomy_path)
    if generator is None:
        tokenizer, model, resolved_device = require_causal_lm(
            model_name,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        generator = make_generator(
            tokenizer=tokenizer,
            model=model,
            device=resolved_device,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
        )

    capability_counts = {label: 0 for label in capability_taxonomy.names}
    domain_counts = {label: 0 for label in domain_taxonomy.names}
    written = 0
    empty = 0
    parse_errors = 0

    def rows() -> Any:
        nonlocal written, empty, parse_errors
        for row in iter_jsonl(input_path):
            if limit is not None and written >= limit:
                break
            sample = CanonicalSample.from_dict(row)
            prompt = build_llm_label_prompt(
                sample,
                capability_taxonomy=capability_taxonomy,
                domain_taxonomy=domain_taxonomy,
                max_chars=max_chars,
            )
            response = generator(prompt)
            sample, parsed_ok = label_sample_with_llm_response(
                sample,
                response_text=response,
                capability_taxonomy=capability_taxonomy,
                domain_taxonomy=domain_taxonomy,
                model_name=model_name,
                max_capabilities=max_capabilities,
                max_domains=max_domains,
            )
            if not parsed_ok:
                parse_errors += 1
            if not sample.capabilities and not sample.domains:
                empty += 1
            for label in sample.capabilities:
                capability_counts[label] = capability_counts.get(label, 0) + 1
            for label in sample.domains:
                domain_counts[label] = domain_counts.get(label, 0) + 1
            written += 1
            yield sample.to_dict()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, rows())
    return {
        "input": str(input_path),
        "output": str(output_path),
        "model": model_name,
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
