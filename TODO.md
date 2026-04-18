# PowerInfer modernization tracker

This file tracks the current modernization work for the repository.

## Baseline

- [x] Audit the current repository structure, build flow, tests, and documentation baseline
- [x] Confirm the existing root build and test flow still works locally (`cmake`, `make`, `ctest`)

## Documentation and positioning

- [x] Refresh the root README around the current PowerInfer story: activation locality, sparse inference, SmallThinker, and practical deployment
- [x] Replace stale links, outdated release notes, and legacy branding drift where they reduce trust
- [x] Add a clearer navigation path between the root project, `smallthinker`, examples, docs, and Python helpers
- [x] Create a fine-tuning addendum at the end to show how to fine-tune supported models

## Hallucination reduction and cognitive load

- [x] Define a practical "cognitive load" feature set for inference that reduces hallucinations without bloating latency
- [x] Promote structured outputs, constrained grammars, and schema-first generation as the default low-hallucination path
- [x] Add task-oriented guidance for when to use short answers, bounded reasoning, or strict JSON generation
- [x] Identify evaluation tasks for hallucination reduction, JSON validity, and factual consistency

## Compression and quantization

- [x] Review whether TurboQuant-style compression fits the current PowerInfer architecture and sparse-routing assumptions
- [x] Identify the highest-value tensors and paths for future compression experiments
- [x] Separate "research exploration" items from production-ready compression work

## Product and developer experience

- [x] Review the server UX and API surface for modern local-app and agent workflows
- [x] Improve quickstart paths so a new user can build, run, and verify a model faster
- [x] Add benchmark and validation guidance that covers speed, memory, and output quality together
