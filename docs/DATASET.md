# Dataset-First Build

The merged dataset should use shared taxonomy axes across all rollout sources,
even when source datasets expose different category names or no categories at
all.

Use two orthogonal axes:

- `capabilities`: what behavior the trajectory teaches, such as planning,
  retrieval, composition, revision, debugging, or structured reasoning.
- `domains`: what subject matter the trajectory covers, such as science,
  mathematics, humanities, writing, medicine, social science, or software.

This lets us build specialists from intersections. For example, scientific
reasoning is not just one flag; it is usually `SCIENCE` plus
`STRUCTURED_REASONING`, and scientific research is `SCIENCE` plus
`RETRIEVAL_SEARCH`.

## Labeling Flow

1. Stream or convert each rollout source into canonical JSONL.
2. Relabel canonical rows with the shared taxonomy.
3. Inspect capability and domain label counts/evidence.
4. Export prompt/ICL labeling jobs for uncertain or high-value rows.
5. Merge reviewed prompt-label outputs back into canonical metadata.
6. Optionally export encoder/prototype data for ModernBERT-style scoring.

The first pass is weak supervision, not the final classifier. It combines:

- source category aliases
- taxonomy keyword cues
- existing canonical capabilities
- label evidence stored in metadata

The weak pass is intentionally conservative. It is meant to produce candidate
labels and evidence, not trusted gold labels. Do not use weak-label outputs as
ICL examples unless you explicitly pass `--allow-weak-examples` for debugging.

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
  --output '{runs_dir}/labeling/prompts.jsonl'
```

After you have reviewed or prompt-labeled examples, pass those reviewed labels
as `--examples`.

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

uv run python -m constellation.cli taxonomy-docs \
  --taxonomy configs/domain_taxonomy.json \
  --output '{runs_dir}/taxonomy/domain_taxonomy.md'
```

List the first 20 distillation targets:

```bash
uv run python -m constellation.cli list-specialist-targets
```

Build one target-specific matched subset:

```bash
uv run python -m constellation.cli build-subsets \
  --input '{runs_dir}/labeled/merged.labeled.jsonl' \
  --output-dir '{runs_dir}/subsets' \
  --target-id science_reasoner \
  --max-train-tokens 2000000
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
- `domains`
- `label_vector`
- `domain_vector`
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
