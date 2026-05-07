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
2. Label canonical rows with the shared taxonomy.
3. Inspect capability and domain label counts/evidence.
4. Export prompt/ICL labeling jobs for uncertain or high-value rows.
5. Merge reviewed prompt-label outputs back into canonical metadata.
6. Optionally export encoder/prototype data for ModernBERT-style scoring.

The preferred first pass is a small generative labeler, because it can read the
taxonomy, understand the rollout, and emit exact JSON labels without tuning.
`Qwen/Qwen3.5-0.8B` is the default after the Qwen3-0.6B probe showed too much
CODING_SOFTWARE bias on non-coding domains. It is exposed by Transformers as an
image-text-to-text model, but `llm-label` uses it text-only for taxonomy JSON.
You can pass `--model Qwen/Qwen3-0.6B` when speed is more important than
cross-domain fidelity.

The deterministic weak pass is not the final classifier. It combines:

- source category aliases
- taxonomy keyword cues
- existing canonical capabilities
- label evidence stored in metadata

The weak pass is intentionally conservative. It is meant to produce candidate
labels and evidence, not trusted gold labels. For the main dataset build, prefer
the LLM labeler below.

## Commands

Label a canonical shard with the Qwen JSON labeler:

```bash
uv run python -m constellation.cli llm-label \
  --input ~/constellation-runs/canonical/agenttrove.debugging_probe.jsonl \
  --output '{runs_dir}/labeled/agenttrove.debugging_probe.labeled.jsonl'
```

Run a small smoke first on the merged rollout file:

```bash
uv run python -m constellation.cli sample-jsonl \
  --input '{runs_dir}/merged/rollouts.canonical.jsonl' \
  --output '{runs_dir}/merged/rollouts.stratified_smoke.jsonl' \
  --group-by source_dataset \
  --max-per-group 10 \
  --limit 100

uv run python -m constellation.cli llm-label \
  --input '{runs_dir}/merged/rollouts.stratified_smoke.jsonl' \
  --output '{runs_dir}/merged/rollouts.qwen35_08_labeled.smoke.jsonl'

uv run python -m constellation.cli label-report \
  --input '{runs_dir}/merged/rollouts.qwen35_08_labeled.smoke.jsonl' \
  --top-examples 2
```

Install GPU labeling dependencies with:

```bash
uv pip install -r requirements/labeling.txt
```

The labeling requirements pin a current Transformers version because Qwen3 and
Qwen3.5 tokenizers/model classes need recent support.

For larger labeling passes, serve the labeler with SGLang and send requests to
its OpenAI-compatible API. In one GPU terminal:

```bash
uv pip install -r requirements/serve.txt

uv run python -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-0.8B \
  --host 127.0.0.1 \
  --port 30000 \
  --mem-fraction-static 0.75
```

Then label from another terminal without reloading weights:

```bash
uv run python -m constellation.cli llm-label \
  --backend sglang \
  --api-base http://127.0.0.1:30000/v1 \
  --concurrency 16 \
  --input '{runs_dir}/labeling/domain_probes.canonical.jsonl' \
  --output '{runs_dir}/labeling/domain_probes.qwen35_08_sglang.jsonl'
```

For SGLang/OpenAI-compatible backends, `llm-label` uses structured output by
default: it sends a JSON Schema with enum-constrained capability/domain labels,
required fields, bounded label arrays, and no extra properties. Use
`--no-structured-output` only when testing a server that does not support
schema-constrained decoding.

`llm-label` applies lightweight post-LLM guardrails by default. These are not a
replacement for review; they drop clearly incompatible coding-only capabilities
from non-coding tasks and add obvious broad labels such as `STRUCTURED_REASONING`
or `PLANNING` when strong task cues are present. Use `--no-label-guardrails` to
inspect raw model labels.

SGLang exposes OpenAI-compatible `/v1/chat/completions`, so the same
`--backend openai-compatible` path can point at vLLM or another compatible
server later.

Use the NLI zero-shot scorer as a faster baseline or distribution sanity check:

```bash
uv run python -m constellation.cli model-label \
  --input '{runs_dir}/merged/rollouts.canonical.jsonl' \
  --output '{runs_dir}/merged/rollouts.nli_labeled.jsonl'
```

The default NLI model is `cross-encoder/nli-MiniLM2-L6-H768`.

Calibrate Qwen on handpicked cross-domain probes before trusting a full merge:

```bash
uv run python -m constellation.cli write-labeling-probes \
  --output '{runs_dir}/labeling/domain_probes.canonical.jsonl'

uv run python -m constellation.cli llm-label \
  --input '{runs_dir}/labeling/domain_probes.canonical.jsonl' \
  --output '{runs_dir}/labeling/domain_probes.qwen35_08_labeled.jsonl'

uv run python -m constellation.cli probe-report \
  --input '{runs_dir}/labeling/domain_probes.qwen35_08_labeled.jsonl'
```

Use the deterministic weak relabeler only for quick audits, fallback operation,
or to inspect cue/evidence quality:

```bash
uv run python -m constellation.cli relabel-capabilities \
  --input ~/constellation-runs/canonical/agenttrove.debugging_probe.jsonl \
  --output '{runs_dir}/labeled/agenttrove.debugging_probe.weak.jsonl'
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
