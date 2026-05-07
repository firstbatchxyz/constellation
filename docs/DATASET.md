# Dataset-First Build

The merged dataset should use one shared capability taxonomy across all rollout
sources, even when source datasets expose different category names or no
categories at all.

## Labeling Flow

1. Stream or convert each rollout source into canonical JSONL.
2. Relabel canonical rows with the shared taxonomy.
3. Inspect label counts and evidence.
4. Export reviewed rows into a ModernBERT-style classifier dataset.
5. Fine-tune a multi-label encoder classifier.
6. Use the classifier to relabel larger streamed shards.

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

Export training rows for a ModernBERT-style multi-label classifier:

```bash
uv run python -m constellation.cli export-classifier-data \
  --input '{runs_dir}/labeled/agenttrove.debugging_probe.labeled.jsonl' \
  --output '{runs_dir}/classifier/modernbert_seed.jsonl'
```

Render taxonomy docs for review:

```bash
uv run python -m constellation.cli taxonomy-docs \
  --output '{runs_dir}/taxonomy/capability_taxonomy.md'
```

## ModernBERT Role

ModernBERT is an encoder backbone, not a finished classifier for this taxonomy.
Use it after we have seed labels:

- weak labels from the relabeling step
- reviewed examples for each capability
- held-out reviewed eval labels

Train it as multi-label classification over `configs/capability_taxonomy.json`.
The exported classifier JSONL contains:

- `text`
- `labels`
- `label_vector`
- `taxonomy_version`
- source metadata

## Review Gate

Before training a classifier, manually inspect at least:

- 50 positive examples per high-priority capability
- 50 uncertain or unlabeled examples
- cross-source label distribution
- examples where source aliases and keyword cues disagree

The classifier should not replace review until it beats weak labels on a held-out
reviewed set.
