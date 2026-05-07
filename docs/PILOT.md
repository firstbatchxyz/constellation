# H100 Pilot

## Goal

Test the core hypothesis with the smallest experiment that can fail clearly:

> A Qwen3-4B specialist trained on capability-specific trajectories beats a
> matched Qwen3-4B general-agent distilled model on held-out tasks from that
> same capability.

## Do First

1. Stream a small slice from AgentTrove.
2. Stream a small slice from Hermes traces.
3. Run canonical validation, basic filtering, and quality scoring.
4. Freeze a held-out eval split before training.

Do not download full datasets into the local workspace. Use streaming reads,
remote object storage, or training-node scratch paths for any larger artifacts.

Split by grouped identifiers, not random rows:

- `task_id`
- repository name
- source dataset
- prompt hash

## First Specialist

Start with `DEBUGGING` or `TERMINAL_WORKFLOW`.

These are good first targets because they can be evaluated more concretely than
planning style:

- command validity
- test pass/fail
- patch application
- recovery after an error observation
- final answer consistency with tool evidence

## Training Set Shape

Create two matched training sets:

- `general_agentic_mix`: broad successful traces from all agentic capabilities
- `debugging_specialist`: debugging-heavy traces with a small anchor mix

Use a matched token budget. Do not let the specialist win just because it saw
more tokens.

Suggested pilot mix:

- 75-85% target capability
- 10-20% general agent/tool etiquette
- 5% ordinary instruction/chat/code QA anchor examples

## SFT Masking

Apply loss only to assistant-authored turns:

- `reasoning`
- `tool_call`
- `final`

Mask loss for:

- `system`
- `user`
- `tool` / `observation`

Tool observations are environment evidence. The model should condition on them,
not learn to fabricate them.

## 1x H100 Training Starting Point

Use this as a conservative first config, then tune from observed memory:

- bf16 full SFT
- FlashAttention
- gradient checkpointing
- max sequence length: 8192 first, then 16384 if memory allows
- micro batch size: 1
- gradient accumulation: 16-64
- learning rate: 1e-5 to 2e-5
- warmup ratio: 0.03
- epochs: 1 for the pilot
- optimizer: AdamW 8-bit or ZeRO/FSDP-backed AdamW if available

The first run should be a data and eval validation run, not a leaderboard run.

## Evaluation

Compare:

- base Qwen3-4B checkpoint
- general-agent distilled descendant
- specialist descendant

Track:

- task success rate
- valid tool-call rate
- observation-grounded recovery rate
- patch/test success
- command efficiency
- trajectory length
- latency and memory
- post-quantization regression

## Data Quarantine

Keep commercial-API reasoning distills out of the first pilot unless each source
passes a manifest review:

- upstream model/provider
- generation method
- whether traces are actual hidden reasoning, synthetic rationales, or summaries
- dataset license
- provider terms for distillation and redistribution
- manual inspection status
- safety/alignment caveats

Agent trajectories with real tool observations are higher signal for this
project than generic long-form reasoning traces.
