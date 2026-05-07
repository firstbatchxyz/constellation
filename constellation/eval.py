"""Lightweight generation eval harness for debugging pilot comparisons."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from constellation.config import load_json_config
from constellation.formatting import render_eval_prompt
from constellation.io import iter_jsonl, write_jsonl
from constellation.schema import CanonicalSample

TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(?P<body>.*?)\s*</tool_call>", re.DOTALL)


class OptionalEvalDependencyError(RuntimeError):
    """Raised when generation eval dependencies are not installed."""


def require_eval_dependencies() -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise OptionalEvalDependencyError(
            "eval requires generation dependencies. Install them on the GPU machine with:\n"
            "uv pip install torch transformers accelerate"
        ) from exc
    return {
        "torch": torch,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
    }


def valid_tool_call(text: str) -> bool:
    match = TOOL_CALL_PATTERN.search(text)
    if not match:
        return False
    body = match.group("body").strip()
    if not body:
        return False
    try:
        json.loads(body)
        return True
    except json.JSONDecodeError:
        return body.startswith("{") and body.endswith("}")


def observation_text(sample: CanonicalSample) -> str:
    for turn in sample.messages:
        if turn.type == "observation":
            return turn.content
    return ""


def word_overlap_score(reference: str, candidate: str) -> float | None:
    reference_words = {
        word.lower()
        for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", reference)
        if len(word) >= 4
    }
    if not reference_words:
        return None
    candidate_words = {word.lower() for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", candidate)}
    return len(reference_words & candidate_words) / len(reference_words)


def patch_intent(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in ("diff --git", "@@", "patch", "fix", "edit"))


def score_generation(sample: CanonicalSample, text: str) -> dict[str, Any]:
    observation = observation_text(sample)
    overlap = word_overlap_score(observation, text) if observation else None
    return {
        "valid_tool_call": valid_tool_call(text),
        "observation_overlap": overlap,
        "observation_grounded": overlap is not None and overlap >= 0.1,
        "patch_intent": patch_intent(text),
    }


def load_eval_samples(path: str | Path, limit: int | None) -> list[CanonicalSample]:
    samples: list[CanonicalSample] = []
    for row in iter_jsonl(path):
        samples.append(CanonicalSample.from_dict(row))
        if limit is not None and len(samples) >= limit:
            break
    if not samples:
        raise ValueError(f"{path} contains no eval samples")
    return samples


def generate_one(
    *,
    deps: dict[str, Any],
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> tuple[str, float, int]:
    torch = deps["torch"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = int(inputs["input_ids"].shape[-1])
    started = time.perf_counter()
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if temperature > 0:
        generate_kwargs["temperature"] = temperature
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generate_kwargs,
        )
    latency = time.perf_counter() - started
    generated_ids = outputs[0][input_length:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    return text, latency, int(generated_ids.shape[-1])


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(row["model_name"], []).append(row)

    summary: dict[str, Any] = {}
    for model_name, model_rows in by_model.items():
        count = len(model_rows)
        grounded_rows = [row for row in model_rows if row["metrics"]["observation_overlap"] is not None]
        summary[model_name] = {
            "count": count,
            "valid_tool_call_rate": sum(row["metrics"]["valid_tool_call"] for row in model_rows) / count,
            "patch_intent_rate": sum(row["metrics"]["patch_intent"] for row in model_rows) / count,
            "observation_grounded_rate": (
                sum(row["metrics"]["observation_grounded"] for row in grounded_rows) / len(grounded_rows)
                if grounded_rows
                else None
            ),
            "avg_latency_seconds": sum(row["latency_seconds"] for row in model_rows) / count,
            "avg_generated_tokens": sum(row["generated_tokens"] for row in model_rows) / count,
        }
    return summary


def write_markdown_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Debugging Eval Summary",
        "",
        "| Model | Valid tool calls | Observation grounded | Patch intent | Latency | Tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_name, metrics in summary.items():
        grounded = metrics["observation_grounded_rate"]
        grounded_text = "n/a" if grounded is None else f"{grounded:.3f}"
        lines.append(
            "| {model} | {tool:.3f} | {grounded} | {patch:.3f} | {latency:.3f}s | {tokens:.1f} |".format(
                model=model_name,
                tool=metrics["valid_tool_call_rate"],
                grounded=grounded_text,
                patch=metrics["patch_intent_rate"],
                latency=metrics["avg_latency_seconds"],
                tokens=metrics["avg_generated_tokens"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_eval_from_config(config_path: str | Path) -> dict[str, Any]:
    config = load_json_config(config_path)
    deps = require_eval_dependencies()
    torch = deps["torch"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(config["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_eval_samples(config["eval_file"], config.get("limit"))

    rows: list[dict[str, Any]] = []
    for model_config in config["models"]:
        model_name = model_config["name"]
        model_path = model_config["model_name_or_path"]
        tokenizer = deps["AutoTokenizer"].from_pretrained(
            model_path,
            trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = deps["AutoModelForCausalLM"].from_pretrained(
            model_path,
            trust_remote_code=bool(model_config.get("trust_remote_code", False)),
            torch_dtype=torch.bfloat16 if bool(config.get("bf16", True)) else None,
        ).to(device)
        model.eval()

        for sample in samples:
            prompt = render_eval_prompt(sample, mode=config.get("prompt_mode", "initial"))
            text, latency, generated_tokens = generate_one(
                deps=deps,
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=int(config.get("max_new_tokens", 512)),
                temperature=float(config.get("temperature", 0.0)),
            )
            rows.append(
                {
                    "model_name": model_name,
                    "model_name_or_path": model_path,
                    "sample_id": sample.id,
                    "prompt": prompt,
                    "generation": text,
                    "latency_seconds": latency,
                    "generated_tokens": generated_tokens,
                    "metrics": score_generation(sample, text),
                }
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_results(rows)
    write_jsonl(output_dir / "generations.jsonl", rows)
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown_summary(output_dir / "metrics.md", summary)
    return summary
