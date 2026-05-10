![Constellation](docs/constellation.png)

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
uv run python -m constellation.cli --help
```

GPU-only training dependencies are intentionally not required for local tests.
Install them on the GPU machine when you are ready to run streaming/training:

```bash
uv pip install -r requirements/train.txt
```

The default SFT backend uses Hugging Face `Trainer` on Accelerate with custom
labels so the canonical turn mask is exact. `TRL` is included in the GPU
requirements and can be enabled with `"use_trl_sft_trainer": true` after a smoke
run confirms the installed TRL version accepts the custom masked dataset.

## Data Access Policy

Do not download full datasets locally into this workspace.

Use one of:

- Hugging Face streaming / iterable datasets
- remote object storage
- remote training-node scratch space
- tiny synthetic fixtures for tests

Local files in this repo should be code, configs, docs, schemas, and small test
fixtures only.

## Dataset Labeling

All rollout sources should be normalized into shared capability and domain
taxonomies:

- [configs/capability_taxonomy.json](configs/capability_taxonomy.json)
- [configs/domain_taxonomy.json](configs/domain_taxonomy.json)

Use [docs/DATASET.md](docs/DATASET.md) for the dataset-first relabeling and
prompt/ICL labeling flow. The first distillation registry supports up to 20
specialists in [configs/specialist_targets.json](configs/specialist_targets.json).
ModernBERT can still be useful as an encoder scorer, but not as a decoder-style
prompt model that emits labels.

For the main dataset build, use `llm-label` with a small instruction model. The
default is `Qwen/Qwen3.5-0.8B` after the Qwen3-0.6B probe showed too much
coding-domain bias. It reads the taxonomy and emits strict JSON labels. The
older `model-label` NLI scorer is still useful as a very fast baseline/audit
path. Weak relabeling is only a fallback/audit path.

For speed, run the labeler behind SGLang and call `llm-label --backend sglang`
so model weights stay resident while Constellation streams JSONL rows through
the OpenAI-compatible API.

The SGLang/OpenAI-compatible path uses JSON Schema structured output by default
to constrain label arrays to known taxonomy enums.

`llm-label` also records post-LLM guardrail edits in metadata so calibration
cleanup is auditable.

For production-sized dataset builds on the GPU node, use the systemd helper so
SGLang and the sharded canonical/labeling job survive SSH disconnects:

```bash
git pull
scripts/final_dataset_systemd.sh setup

journalctl --user -u constellation-final-dataset -f
```

The helper writes canonical shards, labels them through the local SGLang server,
and produces final label/target reports under `CONSTELLATION_RUNS_DIR/final`.
It also reapplies `nvidia-cudnn-cu12==9.16.0.29` before SGLang starts, because
some `torch`/`sglang` installs resolve back to CuDNN 9.10 while SGLang rejects
that PyTorch/CuDNN combination.
If a source stream is still writing a `.tmp` canonical directory, completed
shards can be labeled in parallel with:

```bash
scripts/final_dataset_systemd.sh label-available-shards
```

For a fresh GPU node that should resume from already-uploaded labeled
AgentTrove shards and continue Hermes formatting/labeling, use the gold-path
resume helper:

```bash
git clone https://github.com/firstbatchxyz/constellation.git /home/ubuntu/constellation
cd /home/ubuntu/constellation
scripts/resume_labeling_node.sh start
```

It downloads uploaded AgentTrove labels from Hugging Face, marks them done,
starts SGLang, streams Hermes Kimi/GLM into canonical shards, and keeps a
background label loop running over closed shards. Check progress with:

```bash
scripts/resume_labeling_node.sh status
```

On RTX 6000 / Blackwell CUDA 13 nodes, prefer the Docker launcher to avoid local
CUDA toolkit and JIT kernel drift:

```bash
scripts/run_sglang_rtx6000.sh --detach --pull --stop-existing
```

`resume_labeling_node.sh` automatically uses this Docker path when it detects an
RTX 6000/Blackwell GPU and Docker is available.

For CPU-only formatter nodes, install `uv`, stream Hermes into canonical shards,
and optionally upload complete canonical shards with:

```bash
git clone https://github.com/firstbatchxyz/constellation.git /home/ubuntu/constellation
cd /home/ubuntu/constellation
scripts/format_cpu_node.sh start
```

The formatter does not use LLMs or GPUs. It only streams, parses, filters, and
writes canonical JSONL shards. On high-core CPU nodes it starts multiple HF
stream shards per Hermes source; tune with `FORMAT_PARALLELISM_PER_SOURCE`.
Check progress with:

```bash
scripts/format_cpu_node.sh status
```

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

## Pilot Commands

Run these on the GPU machine. They stream only the requested row count and write
curated artifacts under `CONSTELLATION_RUNS_DIR` or `~/constellation-runs`.

```bash
export CONSTELLATION_RUNS_DIR=~/constellation-runs

uv run python -m constellation.cli stream-convert \
  --source agenttrove \
  --max-rows 10 \
  --output '{runs_dir}/canonical/agenttrove.debugging_probe.jsonl' \
  --skip-errors

uv run python -m constellation.cli stream-convert \
  --source hermes-kimi \
  --max-rows 10 \
  --output '{runs_dir}/canonical/hermes_kimi.debugging_probe.jsonl' \
  --skip-errors

uv run python -m constellation.cli build-subsets \
  --input ~/constellation-runs/canonical/agenttrove.debugging_probe.jsonl \
          ~/constellation-runs/canonical/hermes_kimi.debugging_probe.jsonl \
  --output-dir '{runs_dir}/subsets' \
  --max-train-tokens 200000

uv run python -m constellation.cli train-sft --config configs/train_debugger_sft.json
uv run python -m constellation.cli train-sft --config configs/train_general_agent_sft.json
uv run python -m constellation.cli eval --config configs/eval_debugging.json
```

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
