# Constellation

Capability-distilled specialist network experiments for Qwen3-4B descendants.

The initial target is not routing or serving. The first target is proving that a
single capability specialist beats a matched general-agent distilled control on
held-out tasks from the same broad distribution.

## Current Scope

- Canonical JSONL schema for agent and reasoning trajectories.
- Parsers for AgentTrove-style and Hermes-style traces.
- Capability labeling helpers.
- Basic malformed/repetitive trace filters.
- A lightweight quality scorer for pilot data curation.

No model weights, datasets, or training artifacts are checked into this repo.

## Commands

Use `uv` for local and GPU-machine commands:

```bash
uv run python -m unittest discover -s tests
```

## Data Access Policy

Do not download full datasets locally into this workspace.

Use one of:

- Hugging Face streaming / iterable datasets
- remote object storage
- remote training-node scratch space
- tiny synthetic fixtures for tests

Local files in this repo should be code, configs, docs, schemas, and small test
fixtures only.

## Core Training Rule

Tool observations are context, not targets.

During SFT, loss should be applied only to assistant-generated spans:

- reasoning / scratchpad spans, if included for the experiment
- tool calls
- final answers

Loss should be masked for:

- system and user prompts
- tool observations
- environment output

This matters because predicting observations teaches the model to hallucinate
the environment instead of using it.

## First Experiment

Use one H100 for a small matched-control pilot:

1. Stream a slice of AgentTrove and Hermes traces into canonical records.
2. Filter to successful, coherent, non-repetitive trajectories.
3. Build two training sets with matched token budgets:
   - `general_agentic_mix`
   - one narrow specialist, probably `DEBUGGING` or `TERMINAL_WORKFLOW`
4. Full-SFT both descendants from the same Qwen3-4B checkpoint.
5. Evaluate against base Qwen3-4B, the general distilled control, and the
   specialist on held-out tasks grouped by task/repo/source to avoid leakage.

See [docs/PILOT.md](docs/PILOT.md) for the concrete pilot shape.

## External Data Notes

As of May 7, 2026:

- [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) is listed as
  Apache-2.0 on Hugging Face.
- [open-thoughts/AgentTrove](https://huggingface.co/datasets/open-thoughts/AgentTrove)
  is a large open agent-trajectory corpus using a ShareGPT/terminus-style
  message layout.
- [lambda/hermes-agent-reasoning-traces](https://huggingface.co/datasets/lambda/hermes-agent-reasoning-traces)
  contains multi-turn tool-calling traces with real tool execution results.

Reasoning-distill datasets generated from commercial APIs should stay in a
quarantine bucket until provenance, license, and provider terms are reviewed.
The HF dataset license field alone is not enough for a clean training decision.
