Results from the 27 Nov 2026 overnight run (git 3af4679), BEFORE the candidate-pool fix.
With 6 goals and 20 candidates, the old code gave goals 1-5 four candidates each and goal 6 none.
Affected: ours, no_transformer, no_social, no_ranking (no_ranking crashed). Not affected: cv, lstm, no_cvae, no_goals.
Kept only as a record of the bug; not valid results for the full model.
