# Error Analysis — ArvyaX ML Assignment
### 10 Failure Cases: What Went Wrong and Why

---

## Overview

We ran 5-fold cross-validation on the training set and collected out-of-fold
predictions to identify real failure cases — not hypothetical ones.

Overall model accuracy : 0.583 (F1 macro)
Total failures         : 499 / 1200 training samples

---

## Case 01

Text      : "ended up half relaxed half distracted. the rain helped a little."
True      : mixed
Predicted : restless
Intensity : 5  |  Word count: 11
Context   : sleep=5.0, stress=5, energy=5, time=morning

What went wrong:
The text contains both relaxation and distraction signals. The model latched
onto "distracted" — a strong restless indicator — and ignored "half relaxed."
With stress=5 and energy=5, metadata also pointed toward activation.

Why it failed:
Contradictory text where one emotion word dominates. TF-IDF cannot reason
about the "half... half..." qualifier that signals a balanced mixed state.

How to improve:
Sentiment-aware embeddings that capture qualification and balance would
handle this better than bag-of-words.

---

## Case 02

Text      : "The forest sounds were nice but I kept second-guessing everything."
True      : restless
Predicted : mixed
Intensity : 3  |  Word count: 13
Context   : sleep=6.0, stress=3, energy=3, time=afternoon

What went wrong:
"Second-guessing everything" is a strong restless signal, but "nice" and
moderate context values (stress=3, energy=3) pulled the prediction toward mixed.

Why it failed:
Moderate metadata creates ambiguity. When stress and energy are both at 3,
the model cannot clearly distinguish restless from mixed on context alone.

How to improve:
Phrase-level features treating "second-guessing" as a compound anxiety
indicator would help. More training examples of this pattern are needed.

---

## Case 03

Text      : "ok."
True      : neutral
Predicted : calm
Intensity : 2  |  Word count: 1
Context   : sleep=7.0, stress=2, energy=3, time=morning

What went wrong:
Single-word entry with no emotional content. The model defaulted to calm
based on low stress and morning context.

Why it failed:
Ultra-short text gives the model almost nothing to work with. "ok" can mean
anything from genuine neutrality to suppressed distress.

How to improve:
The uncertain_flag correctly catches this (word_count <= 4). In production,
the system prompts the user for more detail rather than acting on a guess.

---

## Case 04

Text      : "I don't know, I just felt weird the whole time. Not bad, just off."
True      : mixed
Predicted : neutral
Intensity : 3  |  Word count: 16
Context   : sleep=6.5, stress=2, energy=2, time=evening

What went wrong:
"Not bad, just off" is a mixed signal — negation combined with something
being wrong. Low stress and energy suggested neutral to the model.

Why it failed:
Negation handling. TF-IDF treats "not" and "bad" as separate tokens and
cannot understand that "not bad" means something different from "bad" alone.

How to improve:
Character n-grams or negation-aware features would capture "not bad" as a
distinct phrase with its own emotional signature.

---

## Case 05

Text      : "Felt surprisingly at ease. Everything felt possible."
True      : focused
Predicted : calm
Intensity : 4  |  Word count: 7
Context   : sleep=8.0, stress=1, energy=5, time=morning

What went wrong:
"At ease" and "everything felt possible" are both positive but the model
chose calm over focused. High energy and low stress should suggest readiness
for focus, but the text has no task-oriented language.

Why it failed:
Calm and focused share many positive sentiment markers. Without explicit
task-related words, the model cannot reliably distinguish them.

How to improve:
Semantic distinction between calm (passive positive) and focused (active
positive) requires understanding intent. Richer metadata like calendar
context or user goals would help bridge this gap.

---

## Case 06

Text      : "I was calm at first but then started thinking about everything I haven't done."
True      : overwhelmed
Predicted : restless
Intensity : 4  |  Word count: 18
Context   : sleep=5.5, stress=4, energy=2, time=night

What went wrong:
The entry describes a transition from calm to overwhelm. The model predicted
restless — plausible given stress=4 and night context — but "everything I
haven't done" is an overwhelmed pattern, not just restlessness.

Why it failed:
Temporal structure in text. The model cannot understand that the entry
describes a sequence and that the ending state matters more than the start.

How to improve:
Recency weighting — giving more importance to the latter part of the text —
would help with entries that describe emotional transitions.

---

## Case 07

Text      : "the mountain session was peaceful. i feel ready."
True      : focused
Predicted : calm
Intensity : 3  |  Word count: 8
Context   : sleep=7.0, stress=2, energy=3, time=morning

What went wrong:
"Peaceful" strongly activates calm. "I feel ready" is a focused signal but
only 3 words. The longer calm phrase dominated.

Why it failed:
Short focused statements lose to longer calm-associated phrases in TF-IDF
weighted representations. Feature weight is proportional to term frequency,
not emotional salience.

How to improve:
Position-weighted features or sentence embeddings that weight conclusive
statements more heavily would handle this better.

---

## Case 08

Text      : "stressed and tired but somehow got through it."
True      : restless
Predicted : overwhelmed
Intensity : 4  |  Word count: 9
Context   : sleep=4.5, stress=5, energy=1, time=evening

What went wrong:
"Stressed and tired" with sleep=4.5 and stress=5 created a very strong
overwhelmed signal. The model ignored "but somehow got through it" — which
indicates resilience and resolution, more consistent with restless.

Why it failed:
Contrastive clauses are hard for TF-IDF. The stressed/tired/sleep pattern
numerically overwhelmed the resilience signal.

How to improve:
Contrastive conjunction detection — "but" should down-weight preceding
negative tokens. This requires syntactic awareness beyond bag-of-words.

---

## Case 09

Text      : "nice session. felt better."
True      : calm
Predicted : neutral
Intensity : 2  |  Word count: 4
Context   : sleep=6.0, stress=2, energy=3, time=afternoon

What went wrong:
Very short text with generic positive words. "Better" implies improvement
from a prior state but gives no absolute emotional anchor.

Why it failed:
Comparative language requires knowing the baseline. "Felt better" without
context could mean anything. The model defaulted to neutral.

How to improve:
Using previous_day_mood as a stronger prior. If previous_day_mood=overwhelmed
and text says "felt better", the likely current state is calm or neutral.
The model underutilizes previous_day_mood in short-text cases.

---

## Case 10

Text      : "couldn't stop thinking but in a good way. ideas kept coming."
True      : focused
Predicted : restless
Intensity : 5  |  Word count: 12
Context   : sleep=7.0, stress=3, energy=5, time=morning

What went wrong:
"Couldn't stop thinking" is a strong restless indicator in most contexts.
The qualifier "in a good way" and "ideas kept coming" completely reverse the
meaning — but TF-IDF treats all tokens equally.

Why it failed:
Semantic inversion via qualification. The same phrase means the opposite
depending on its qualifier. Bag-of-words models fundamentally cannot handle
this pattern.

How to improve:
This is a genuine limitation of TF-IDF. A transformer-based embedding like
MiniLM would encode full phrase context and likely get this right.

---

## Summary of Failure Patterns

Pattern                       Cases     Key Issue
Contradictory text signals    01, 08    Model picks dominant signal, misses balance
Negation and qualification    04, 10    TF-IDF cannot handle "not bad" or "in a good way"
Ultra-short text              03, 09    Insufficient signal, high uncertainty
Calm vs focused ambiguity     05, 07    Positive states share too many surface features
Temporal transitions          06        Ending state matters more than starting state
Contrastive clauses           08, 10    "but somehow" reverses the emotional trajectory

---

## Key Takeaway

The model's failures cluster around linguistically complex patterns that TF-IDF
cannot handle by design. These are not fixable by tuning hyperparameters —
they require better text representations.

For production:
1. Use lightweight sentence embeddings (MiniLM-L6, 22MB) instead of TF-IDF
2. Weight previous_day_mood more heavily for short-text cases
3. Flag uncertain predictions and prompt users for clarification
4. Keep the decision engine separate from the classifier — even imperfect
   state predictions lead to reasonable wellness recommendations

The system knows when it is unsure. That is by design.