# Next assignment: fresh upstream training

The user approved the next two-computer plan on November 16 and authorized autonomous execution and coordination through branch pushes. The completed 10-stage experiment is preserved: baseline 0.8668672; combined and ENN did not improve it. Do not repeat that run.

**Second computer:** fetch `origin/research/upstream-reproduction` and follow its **COORDINATION.md**. Create a separate worktree/branch `research/upstream-worker`, then launch `scripts/upstream_exchange.py --worker --publish` inside tmux with eight compute threads. The document has exact commands. Do not check out another branch in the completed experiment's worktree.

The worker prepares source/data and the reference, trains the bounded eight-model upstream replay, and pushes compressed fresh development predictions and receipts after each job to `research/upstream-worker`. This includes actual prediction artifacts so the first Mac can independently rescore them. No qualifying outcomes are used. Preserve partial outputs and publish failures; do not force-push or delete them to restart.

**First Mac:** finish its six existing assigned stages, publish their verified results, fetch worker progress, independently verify fresh outputs, then run the predeclared fresh-only base-vs-chain blends. Its durable coordinator lives on `research/upstream-reproduction` and does not execute remote jobs.

Instructions: https://github.com/mquan/netflix_challenge/blob/research/upstream-reproduction/COORDINATION.md

Status: https://github.com/mquan/netflix_challenge/blob/research/upstream-reproduction/COORDINATION_STATUS.md

A branch push is the handoff channel, not a remote launch. The second computer's session should pick up this assignment and execute it without waiting for the user to return.
