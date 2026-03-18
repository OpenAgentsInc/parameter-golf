# Psionic Parameter Golf Spec

## Purpose

This document defines a serious plan for competing in Parameter Golf with a
Psionic-native stack instead of a Python-first training stack.

The goal is not "add some Psionic wrappers around the current scripts."
The goal is:

- model definition in Psionic
- data loading and packing in Psionic
- training loop and optimizer execution in Psionic
- evaluation and `val_bpb` accounting in Psionic
- artifact quantization, compression, and roundtrip validation in Psionic
- challenge logs and submission artifacts produced from that stack

Python is allowed only as an optional compatibility shim if the challenge
submission rules force a `train_gpt.py` entrypoint.

## Executive Summary

We should pursue this, but we need to be honest about current posture.

Psionic is now a credible Rust-native ML foundation:

- it has a real array layer
- it has a real `nn` layer
- it has optimizer and training substrate
- it has distributed helpers
- it has model IO, artifact, and eval infrastructure
- it has both CUDA- and Metal-facing work
- it has an explicit MLX-class framework program for local Apple workflows

Psionic is not yet a drop-in Parameter Golf competitor.

The missing pieces are not small:

- no current Parameter Golf-specific causal decoder lane
- no current FineWeb/SP1024 challenge data path
- no current exact `val_bpb` reproduction harness for the challenge
- no current end-to-end 8xH100 decoder training lane proved at challenge speed
- no current challenge-compliant artifact accounting story for a multi-crate Rust runtime

The right plan is:

1. build a Psionic-native reference lane that exactly reproduces the current
   challenge baseline behavior and metrics
2. only then widen into parameter-efficient architecture research
3. treat leaderboard claims as blocked until the challenge packaging and code
   accounting story is explicit

## Hard Constraints From Parameter Golf

The current challenge shape matters more than our implementation preferences.

- Artifact cap: `16,000,000` bytes total
- Main leaderboard cap: train in under `10` minutes on `8xH100`
- Metric: tokenizer-agnostic FineWeb validation compression in `bits per byte`
- Evaluation must be self-contained and reproducible
- No external downloads or network during evaluation
- Current repo language strongly assumes a `train_gpt.py`-style submission
- Current baseline accounting effectively counts code bytes from the script plus
  compressed model bytes

This creates a non-trivial issue for a native Psionic submission:

- a real multi-crate Rust runtime is not free code
- a tiny Python launcher that shells out to a large Rust binary is not an
  honest interpretation of "all counted code should live in `train_gpt.py`"
- we should not pretend challenge accounting is solved just because Psionic
  code lives in another repo

## What "Use Psionic Instead Of Python" Should Mean

For this effort, "Psionic instead of Python" should mean:

- the training and eval substrate is Psionic-owned end to end
- no hidden PyTorch training loop does the real work
- no MLX runtime dependency does the real work
- no FFI proxy to a Python trainer is treated as a Psionic implementation
- artifacts, lineage, metrics, and refusal boundaries stay Psionic-owned

Acceptable temporary exceptions:

- a thin `train_gpt.py` wrapper if challenge submission rules require it
- offline comparison harnesses that verify parity against the current Python
  baseline
- a staged migration where we first match current behavior before replacing the
  full submission path

Not acceptable:

- using Psionic only for metadata while Python still owns model execution
- using PyTorch for the real train loop and calling it "Psionic-backed"
- ignoring code-size accounting because the Rust code lives outside the record
  folder

## Current Psionic Posture Relevant To This Effort

### What Psionic already gives us

Psionic already has strong substrate we can reuse:

- `psionic-array`
  - lazy array facade, explicit `eval`, bounded CPU/Metal/CUDA surface
- `psionic-nn`
  - module tree, parameter identity, save/load, optimizers, quantized wrappers
- `psionic-train`
  - training core, checkpoint lineage, optimizer substrate, mixed-precision and
    distributed optimizer contracts
- `psionic-distributed`
  - public distributed-group and collective helper surfaces
- `psionic-data`
  - dataset manifests, tokenizer digests, packing policies, batch planning
- `psionic-eval`
  - benchmark package and eval-run infrastructure
- `psionic-models`
  - reusable model-family definitions and metadata
- `psionic-mlx-*`
  - bounded MLX-class local package ecosystem for Apple-oriented workflows

This is enough foundation to justify a real buildout.

### What Psionic does not yet give us

Psionic does not yet ship the exact lane we need:

- no compact challenge decoder family matching the current GPT baseline
- no FineWeb shard reader or challenge-specific streamed token loader
- no SentencePiece byte-accounting path for challenge `val_bpb`
- no Parameter Golf artifact packer and roundtrip validator
- no proven `8xH100`, `<=10min`, small-LLM train lane
- no explicit challenge submission wrapper or accounting contract

## Current Challenge Baseline We Need To Match First

Before doing any architecture research, we should port the current baseline
shape into Psionic and prove parity.

The reference target today is:

- SP-1024 tokenizer
- 9 layers
- width 512
- 8 attention heads
- 4 KV heads
- tied embeddings
- sequence length 1024
- `524,288` train tokens per step
- `val_bpb` computed over the fixed FineWeb validation split
- post-train int8 plus zlib roundtrip as the reported artifact metric

This is not because the baseline is optimal.
It is because without a parity lane we will not know whether future gains come
from Psionic, from architecture changes, or from metric drift.

## Main Strategic Decision

We should treat this as two separate deliverables.

### Deliverable A: Psionic-native research and training lane

This is the real engineering target.

It should own:

- model execution
- training
- evaluation
- quantization and compression
- artifact lineage
- benchmark reporting

### Deliverable B: challenge-compliant submission lane

This may or may not be identical to Deliverable A depending on the final rule
interpretation.

It should own:

- final `train_gpt.py` or other accepted entrypoint
- artifact byte accounting
- record-folder output
- submission metadata

We should not assume A automatically solves B.

## Required Changes In Psionic

The work should be mapped into existing Psionic subsystem names, not a shadow
"parameter golf framework" living off to the side.

| Area | Current posture | Required Psionic work |
| --- | --- | --- |
| `psionic-data` | generic dataset manifests and packing exist | add FineWeb shard contract, challenge validation-split manifest, SentencePiece tokenizer binding, deterministic streamed token loader, and byte-count lookup generation for exact `val_bpb` |
| `psionic-models` | AttnRes, Tassadar, GPT-OSS, and other bounded model families exist | add a compact causal decoder family for small-vocab training with tied embeddings, RoPE, GQA, RMSNorm, and challenge-friendly parameter accounting |
| `psionic-train` | training core and optimizer substrate exist, but not this lane | add an end-to-end causal LM trainer, grad accumulation, challenge wallclock budgeting, parameter-group optimizer splits, and comparable logging |
| `psionic-nn` | modules and optimizer shells exist | add any missing causal-LM primitives needed by the decoder family, especially if current attention or embedding surfaces are too narrow |
| `psionic-array` and backends | bounded CPU/Metal/CUDA array surface exists | add or widen kernels and execution paths required for competitive small-decoder training throughput |
| `psionic-distributed` | distributed helpers exist | add the exact data-parallel or sharded training path we will use on `8xH100`, with honest capability and refusal posture |
| `psionic-eval` | benchmark and eval substrate exists | add Parameter Golf eval report generation, exact `val_loss` plus `val_bpb`, and challenge roundtrip validation |
| `psionic-train` plus `psionic-eval` | model IO and reports exist | add int8 plus zlib export and reload path, artifact byte accounting, and acceptance tests against the challenge scripts |
| packaging and compatibility | not solved | decide whether to ship a native Rust entrypoint, a tiny Python wrapper, or a non-record-only Psionic lane until rules are clarified |

## The Minimum Honest Psionic Roadmap

### Phase 0: rules and accounting closure

Before chasing leaderboard claims, settle the submission contour.

Required outcomes:

- explicit decision on how Rust runtime code counts toward the `16MB` artifact
- explicit decision on whether a wrapper-based submission is acceptable
- explicit decision on whether a pure non-Python lane is only for non-record
  submissions until rules are clarified

If we do not have this, we can still build the system, but we should label it
`research` or `non-record` rather than "leaderboard-ready."

### Phase 1: data and metric parity

Build the challenge oracle in Psionic before model work.

Required work:

- FineWeb shard reader for the current binary shard format
- fixed validation split loader
- SentencePiece token metadata extraction
- exact token-byte accounting matching current repo logic
- parity tests against `train_gpt.py` and `train_gpt_mlx.py`

Exit criteria:

- on the same fixed tokens and logits, Psionic computes the same `val_loss`
  and `val_bpb` as the current Python scripts within tight numerical tolerance

### Phase 2: model parity

Port the current baseline model exactly.

Required work:

- tied embedding support
- GQA attention
- RoPE
- RMSNorm
- baseline residual mixing path
- challenge-comparable parameter counting

Exit criteria:

- same seed and same weights produce parity logits and loss on a fixed fixture

### Phase 3: single-device training parity

Build the smallest complete train loop first.

Required work:

- single-device forward and backward
- optimizer group split comparable to baseline
- mixed precision discipline
- checkpoint and restart
- raw plus int8 plus zlib artifact roundtrip

Exit criteria:

- 1-device Psionic run trains, validates, exports, reloads, and reproduces the
  roundtrip metric path without Python owning execution

### Phase 4: multi-GPU throughput closure

This is the first point where the challenge becomes real.

Required work:

- the exact `8xH100` execution strategy
- communication pattern choice
- throughput-focused kernel and optimizer work
- wallclock budget enforcement
- identical validation and export path under distributed execution

Exit criteria:

- stable `8xH100` training run on the baseline family
- logs comparable to the current challenge records
- measured wallclock and memory behavior we can reason about honestly

### Phase 5: architecture research

Only after parity exists should we search for better models.

Good candidate directions:

- shared-depth or recurrent blocks
- stronger parameter tying
- lower-rank or structured projections
- better post-train quantization and compression
- challenge-specific tokenizer changes if we can prove `val_bpb` correctness
- AttnRes-like or recurrent ideas if they improve the parameter-efficiency
  frontier without breaking challenge compliance

Exit criteria:

- every claimed improvement beats the Psionic baseline under the same metric
  and artifact accounting

### Phase 6: submission packaging

Turn the strongest lane into an accepted challenge entry.

Required work:

- final wrapper or entrypoint story
- record-folder output generation
- README and `submission.json` generation
- reproducibility notes
- code-size accounting that does not cheat

Exit criteria:

- a record or non-record submission we would be comfortable defending publicly

## Recommended Crate Placement

To keep this legible, the first implementation should land under current crate
boundaries like this:

- `crates/psionic-data/src/parameter_golf.rs`
  - shard format, tokenizer binding, fixed split manifest, byte lookup helpers
- `crates/psionic-models/src/parameter_golf_decoder.rs`
  - baseline compact causal decoder family
- `crates/psionic-train/src/parameter_golf.rs`
  - reference training loop and artifact export
- `crates/psionic-eval/src/parameter_golf.rs`
  - validation, `val_bpb`, and roundtrip report
- `crates/psionic-train/examples/parameter_golf_reference_run.rs`
  - first runnable end-to-end reference lane

If the lane proves real and broad enough, then we can split it more finely.
We should not start by inventing a large new top-level subsystem.

## Optimizer And Systems Requirements

The current Python baseline is not just "a transformer and Adam."
It uses a challenge-specific systems stack:

- bf16 compute
- optimizer grouping
- Muon for matrix-shaped parameters
- fused or compiled execution
- high-throughput distributed training
- wallclock-aware warmdown behavior

Psionic therefore needs explicit decisions on:

- whether we reproduce Muon exactly or use a clearly justified alternative
- whether the first truthful path is DDP-style, tensor-parallel, FSDP-style, or
  another split
- which kernels must be widened for RoPE, GQA attention, RMSNorm, embedding,
  and logits
- whether the first milestone is CUDA-only, with Apple/MLX-like local iteration
  as a separate convenience path

My recommendation:

- reproduce the baseline optimizer behavior as closely as possible first
- do not substitute a different optimizer just because it already exists in
  Psionic
- keep Apple local iteration as useful but secondary
- focus real challenge closure on CUDA and `8xH100`

## Submission Compliance Risks

These are the main reasons this can fail even if the Rust engineering works.

### Risk 1: code-size accounting

The current challenge language is script-centric.
A large Rust runtime plus a tiny Python launcher may be treated as invalid or at
least out of spirit.

### Risk 2: build-time accounting

If evaluation requires compiling a large Rust workspace, build time and
dependency shape may break the under-10-minute execution story even before
training starts.

### Risk 3: fairness optics

If we exploit a loophole where baseline accounting ignores external runtime
code, the result may be rejected even if it technically runs.

### Risk 4: metric drift

Tokenizer-agnostic `val_bpb` is easy to get subtly wrong.
We need exact parity tests before any optimizer or architecture research.

### Risk 5: over-claiming readiness

Psionic being a strong substrate does not mean the Parameter Golf lane is ready.
We should preserve explicit claim discipline:

- `data/eval parity`
- `single-device training parity`
- `distributed baseline parity`
- `research-only architecture variant`
- `submission-ready`

## What We Should Not Do

- Do not start by designing a brand-new architecture before the baseline ports.
- Do not claim leaderboard readiness before code accounting is settled.
- Do not hide Python execution behind Psionic naming.
- Do not bypass existing Psionic crates with a one-off challenge runtime.
- Do not optimize only for local Apple workflows and call the result challenge
  closure.
- Do not treat non-record experimentation as evidence that the 10-minute track
  is solved.

## Recommended Immediate Next Tranche

The next tranche should be small, concrete, and hard to fake.

1. Implement exact FineWeb shard reading and SentencePiece byte accounting in
   Psionic.
2. Build a Psionic eval harness that reproduces `val_loss` and `val_bpb` from
   the current repo on a frozen validation fixture.
3. Port the current 9x512 tied-embedding baseline architecture into
   `psionic-models`.
4. Build a single-device reference trainer that can run the model, export a raw
   artifact, export an int8 plus zlib artifact, reload it, and re-evaluate.
5. Decide whether the first public-facing result should be a non-record
   submission until the code-accounting story is explicit.

## Success Criteria

This effort is successful only if we can say all of the following honestly:

- Psionic owns the real training and eval path.
- The challenge metric is reproduced exactly.
- The baseline model is matched before we widen claims.
- Multi-GPU throughput is measured, not assumed.
- Artifact accounting is honest.
- The final result is either challenge-compliant or explicitly labeled as a
  non-record or research lane.

## Bottom Line

We should do this.

Psionic is now strong enough to justify a real Parameter Golf lane, and the
challenge is a good forcing function for compact-model training, artifact
compression, and honest benchmark packaging.

But we should be precise about what is currently true:

- Psionic is a credible foundation
- Psionic is not yet a direct Parameter Golf submission stack
- the fastest route is baseline parity first, research second, submission
  packaging third
- the biggest non-technical blocker is the current challenge assumption that
  submission code lives in a `train_gpt.py`-shaped artifact

That means the correct near-term target is:

> build a Psionic-native reference lane that exactly matches the current
> challenge oracle, then turn it into either a rule-compliant submission path
> or an explicit non-record track entry, without cheating on code accounting or
> execution ownership
