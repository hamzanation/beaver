# BEAVER Benchmark: Overview

## 1. What BEAVER is
- BEAVER is a text-to-SQL benchmark for enterprise-style databases.
- It is designed to test whether language models can translate natural-language questions into SQL over realistic, complex schemas.
- The benchmark focuses on harder reasoning problems than many public text-to-SQL datasets.

## 2. Why it matters
- Enterprise schemas are often more complex than academic datasets.
- Questions may require domain knowledge, joins, and multi-step decomposition.
- BEAVER is useful because it evaluates more than just syntax: it also tests schema linking and reasoning quality.

## 3. What is included
- Datasets: dw, dw_real, neutron, nova
- Methods: Few-shot, DAIL-SQL, DIN-SQL, ReFoRCE
- Evaluation types:
  - Execution accuracy
  - Fine-grained subtask evaluation

## 4. Datasets
### dw (Data Warehouse)
- A primary evaluation dataset covering real data warehouse questions and schemas.
- Focuses on analytical queries over enterprise-style databases.
- Contains questions from actual query logs with normalized tables and complex joins.

### dw_real
- A variant of dw with additional real-world complexity and edge cases.
- Used to test how methods generalize to different data warehouse configurations.

### neutron
- Another enterprise dataset with domain-specific schema patterns.
- Tests generalization across different enterprise environments.

### nova
- A fourth evaluation domain for diverse schema structures.
- Provides additional coverage of different data modeling approaches.

All datasets include:
- Natural-language questions
- Target SQL queries
- Table schemas with column names and types
- Example rows from tables
- Metadata for schema linking (tables, columns, joins, domain knowledge)

## 5. Methods overview
### Few-shot
- A simple prompt-based baseline.
- Provides one example question-to-SQL pair plus the current question and schema.
- The model is asked to generate SQL without additional reasoning steps.
- Good baseline for comparing more complex approaches.

### DAIL-SQL (Demonstration and Exploration-based In-Context Learning)
- Uses example selection to choose the most relevant examples from the dataset.
- Aims to show the model examples similar to the current question.
- Based on semantic similarity or feature-based selection.
- More sophisticated than random few-shot examples.

### DIN-SQL
- Uses schema linking and query decomposition.
- Breaks the generation into steps:
  1. Schema linking: identifies which tables and columns are relevant
  2. Query decomposition: splits complex queries into sub-queries
  3. SQL generation: produces the final SQL
- Self-correction loop: if SQL fails, it attempts to fix the query.

### ReFoRCE (Reflexive Reformulation with Code Execution)
- An agentic method with reinforcement learning-like components.
- Features:
  1. **Candidate generation**: generates multiple SQL candidates
  2. **Majority voting**: selects the most common output
  3. **Column exploration**: iteratively refines column selections
  4. **Execution feedback**: uses actual execution results to guide refinement
- More computationally expensive but often produces higher accuracy.

## 6. How the benchmark is run
1. Select a dataset, such as dw.
2. Load the question and schema data.
3. Build a prompt with the question and relevant tables.
4. Ask the model to generate SQL.
5. Execute the generated SQL and compare it to the gold SQL result.
6. Optionally evaluate subtasks such as table retrieval, column mapping, join-key detection, domain knowledge, and query decomposition.

## 7. What is in the prompt
### Always present
- A system instruction telling the model to produce MySQL SQL
- A few-shot example
- The current user question
- Formatted table descriptions with example rows and columns

### Optionally present
- Mapping from phrases in the question to columns
- Join keys between tables
- Domain knowledge hints
- Query decomposition hints

## 8. What the settings mean
- Setting 0: baseline end-to-end setting with no extra hints
- Setting 1: adds gold tables, column mapping, and join keys
- Setting 2: adds domain knowledge and query decomposition on top of setting 1

## 9. The dw dataset
- dw is one of the BEAVER benchmark datasets.
- It includes questions, schema information, example prompts, and SQL references.
- In the repository, it is used to run experiments for the few-shot method.

## 10. Local results from eval/unified-output
The workspace contains local results for Few-shot runs on dw with setting 1.

### Summary of local results
| Model | Exact matches | Accuracy | Table retrieval F1 | Column mapping F1 | Join key F1 | Domain knowledge F1 | Query decomposition |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen2.5-coder-14b-instruct | 8/100 | 8.0% | 0.7846 | 0.6409 | 0.6590 | 0.1906 | 0.3385 |
| gpt-5.4-mini | 15/100 | 15.0% | 0.9481 | 0.8364 | 0.8248 | 0.2497 | 0.4813 |
| gpt-5.5 | 20/100 | 20.0% | 0.9151 | 0.7963 | 0.7979 | 0.2430 | 0.5253 |

### Interpretation
- The local results already show that the benchmark is challenging.
- The GPT-5.5 run outperformed the other two on exact-match accuracy in this local evaluation.
- The subtask scores suggest that table retrieval and column mapping are relatively stronger than domain knowledge and query decomposition.
- **Note on Claude models**: While the repository supports Anthropic's Claude models, no local Claude results are currently available in the unified-output directory. Claude can be used via the `--model claude` flag when running evaluations.

## 11. Paper-level context
The BEAVER paper reports that strong agentic approaches using GPT-5.2 reached about 10.8% execution accuracy, while adding oracle-style subtask hints increased accuracy to about 30.1%.

## 12. Bottom line
- BEAVER is a realistic and difficult text-to-SQL benchmark.
- Few-shot is a simple but informative baseline; other methods add complexity for potentially higher performance.
- The dw dataset and setting 1 provide a practical setup for comparing models.
- The results show that even strong models still struggle with enterprise-style reasoning.
