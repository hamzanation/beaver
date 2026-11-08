# BEAVER few-shot retrospective

## What the current evidence supports

The current few-shot DW comparison is small but very informative. The data points already suggest that the large model has a stronger ceiling, but that the small model can become much more competitive when the prompt setting is improved.

The clearest pattern is that the transition from setting 0 to setting 1 produces the largest return for the small model. In the current sheet, `gpt-5.4-mini` goes from 5 exact matches at setting 0 to 13 at setting 1, and then 18 at setting 2. The large model also improves, but the jump is much smaller in relative terms because it already starts from a more capable baseline.

## What appears to be happening

The subtask metrics show that the biggest gains happen in the structural steps of the pipeline:

- table retrieval improves sharply,
- column mapping improves sharply,
- join key alignment improves sharply.

This suggests that extra structure in the setting helps the model ground itself in the schema and compose the relational logic more effectively. The query decomposition score also improves, but it remains the weakest of the major subtasks in absolute terms.

That is important because it hints at a real conclusion: the remaining difficulty is not merely retrieving schema elements. It is keeping the overall reasoning structure aligned with the gold decomposition.

## Practical interpretation

From a resource perspective, the small model is much more economical. Its latency stays near 3 seconds, while the large model is roughly 30-40 seconds in the same setting ladder. That means the larger model is not just “better.” It is better at a materially higher operational cost.

So the most defensible conclusion is:

- yes, larger models win on overall correctness,
- but structured settings are a powerful way to recover part of the gap for smaller models,
- and the main unresolved challenge is logical decomposition rather than raw table/column lookup.

## Why this matters for the narrative

This analysis should not be framed as “size always wins.” A better framing is:

- model size sets the ceiling,
- setting quality determines how much of that ceiling the smaller model can approach,
- and the remaining output gap appears to come from decomposition and composition fidelity, not from simple retrieval.

That is the cleanest story the current evidence can support.
