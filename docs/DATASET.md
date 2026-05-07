# Dataset-First Build

The merged dataset should use one shared capability taxonomy across all rollout
sources, even when source datasets expose different category names or no
categories at all.

## Labeling Flow

1. Stream or convert each rollout source into canonical JSONL.
2. Relabel canonical rows with the shared taxonomy.
3. Inspect label counts and evidence.
4. Export prompt/ICL labeling jobs for uncertain or high-value rows.
5. Merge reviewed prompt-label outputs back into canonical metadata.
6. Optionally export encoder/prototype data for ModernBERT-style scoring.

The first pass is weak supervision, not the final classifier. It combines:

- source category aliases
- taxonomy keyword cues
- existing canonical capabilities
- label evidence stored in metadata

## Commands

Relabel a canonical shard:

```bash
uv run python -m constellation.cli relabel-capabilities \
  --input ~/constellation-runs/canonical/agenttrove.debugging_probe.jsonl \
  --output '{runs_dir}/labeled/agenttrove.debugging_probe.labeled.jsonl'
```

Export prompt/ICL labeling jobs:

```bash
uv run python -m constellation.cli export-labeling-prompts \
  --input '{runs_dir}/labeled/agenttrove.debugging_probe.labeled.jsonl' \
  --examples '{runs_dir}/labeled/agenttrove.debugging_probe.labeled.jsonl' \
  --output '{runs_dir}/labeling/prompts.jsonl'
```

Export rows for optional encoder/prototype scoring:

```bash
uv run python -m constellation.cli export-classifier-data \
  --input '{runs_dir}/labeled/agenttrove.debugging_probe.labeled.jsonl' \
  --output '{runs_dir}/classifier/encoder_seed.jsonl'
```

Render taxonomy docs for review:

```bash
uv run python -m constellation.cli taxonomy-docs \
  --output '{runs_dir}/taxonomy/capability_taxonomy.md'
```

## Prompt/ICL vs ModernBERT

Prompt + ICL labeling should use a generative instruction model. It can read
the taxonomy, see examples, and emit strict JSON labels.

ModernBERT is encoder-only. It should not be treated as a decoder-style ICL
model that emits labels from a prompt. If we use ModernBERT without fine-tuning,
use it for encoder-style scoring:

- embed trajectory text and label descriptions
- rank labels by similarity to taxonomy descriptions and reviewed examples
- use nearest reviewed examples as evidence for a generative labeler

- `text`
- `labels`
- `label_vector`
- `taxonomy_version`
- source metadata

## Review Gate

Before trusting automated labels, manually inspect at least:

- 50 positive examples per high-priority capability
- 50 uncertain or unlabeled examples
- cross-source label distribution
- examples where source aliases and keyword cues disagree

Prompt/ICL or encoder scoring should not replace review until it beats weak
labels on a held-out reviewed set.
