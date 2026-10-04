# Feedback Log

Every piece of human feedback on RAGNova, and what we changed because of it (ROADMAP.md, "Human testing / feedback loops"). How the loop works is in [Chapter 12 Part 6](chapters/ch12-integration-testing-and-feedback.md).

Raw ratings live in `feedback/feedback.jsonl`, written by the chat page's rating form. `python scripts/summarize_feedback.py` turns them into the numbers below. **Outside testers and the team are reported separately** (Chapter 3 §3.5.3): the team rating its own system is a sanity check, not evidence.

---

## Tester task card (hand this to each outside tester)

1. Type your name or initials in the sidebar.
2. Ask **five questions of your own** about campus notices, the library, IT onboarding or the project rules. Don't look at the files first.
3. Ask **one question by voice** (the mic button).
4. Ask **one question with an image** (a screenshot or a photo of a notice).
5. Rate every answer 1–5: 1 = wrong or unhelpful, 3 = partly right, 5 = correct and the sources back it up. Add a comment whenever something surprised you.

Please think aloud while you do it. We want to hear where you got stuck, not only the ratings.

---

## Sessions

| Date | Tester (initials) | Outside or team? | Ratings | Average | Notes |
|---|---|---|---|---|---|
| | | | | | |

## Summary (paste from `scripts/summarize_feedback.py`)

| Group | Ratings | Average | Rated ≤ 2 |
|---|---|---|---|
| Outside testers | | | |
| Team | | | |

## Findings and what we changed

One row per finding. A finding without a change is fine, as long as the reason is written down.

| # | What the tester saw | Why it happened | What we changed | Commit |
|---|---|---|---|---|
| F1 | | | | |
