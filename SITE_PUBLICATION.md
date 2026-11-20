# Own-training site publication

The active website reports this project's own training results.
`scripts/build_research_journey.py::own_training_milestones` excludes all
archived Public bank strategies and the former qualifying metric from
both generated payload/score tables and static plots. Fresh own qualifying
results use the distinct fresh_qualifying metric. Original2006-2009
competition references remain attributed historical comparisons.

Do not restore public-bank result cards, breakthrough section or markers.
The current progress link is OWN_PROGRESS.md. Keep source research
archives intact in the research branch, including historical PROGRESS.md.
When publishing changes to gh-pages, skip retired BREAKTHROUGH.md,
PUBLIC_BANK.md,QUALIFYING_PROGRESS.md,PROGRESS.md and
assets/qualifying_progress.svg; remove their active Pages copies.
Do not regenerate the old public qualifying SVG.

Preserve concurrent presentation edits and use normal forward pushes.
Current source/layout contains no operational dashboard. Validate every
current own-score proof before generation, then verify live HTML/SVG/PNG
bytes against the committed source and confirm no public-bank milestones
are present in the site's embedded data.
