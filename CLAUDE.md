# Netflix Prize research repo — session entry point

1. **Read [STATUS.md](STATUS.md) first.** It holds the goal, score, current
   plan step, in-flight jobs, next actions and open decisions. Then read the
   operating rules in [GOAL.md](GOAL.md).
2. Before launching anything, check `ps` and the in-flight table. Another
   agent (Codex) works on this branch: claim work in GOAL.md's claims table
   before starting, and never stop or edit its jobs or frozen files.
3. **Update STATUS.md** whenever state changes (a job starts or finishes, a
   result lands, the plan or a decision changes) and before ending a session.
   Commit and push after each step.
