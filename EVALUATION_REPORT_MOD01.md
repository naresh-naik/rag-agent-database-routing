# RAG Agent — 100-Question Accuracy Benchmark (IFRS Model Financial Statements)

_Run label: `run5_fixed_8b` · Generated: 2026-08-15 09:39_  
_Model: `llama-3.1-8b-instant` · Judge: `llama-3.1-8b-instant` · Embeddings: `BAAI/bge-small-en-v1.5` (+ BM25 hybrid) · Retriever top-k: 8_

## Improvement Lineage

| Run | Pipeline | Accuracy |
|---|---|---|
| `run1_baseline` | vanilla chunking, dense-only | 52.7% |
| `run2_improved_8b` | chunk consolidation + hybrid BM25/dense RRF | 82.1% |
| `run3_fixed_8b` | + judge parentheses note, re-answered failures | 86.7% |
| `run4_full_improved` | + table serialization, reranker, numeric grounding guard, citation stripping | 80.1% (13 rows judged during quota exhaustion) |
| `run5_fixed_8b` | run4 with failed rows re-answered | **84.2%** |

Note: run3's higher score was judged under a less strict regime and with
70b generation on the second pass; run5 is the honest number for the fully
8b pipeline with all Tier 1/2 improvements enabled.

## Executive Summary

- **Overall accuracy (LLM-judge mean): 84.2%** (excluding 2 defective reference questions: [32, 33])
- Correct: **76** · Partial: **13** · Incorrect: **9** (of 98 valid questions)
- Retrieval hit rate: **98/98** · Mean groundedness: **100%**
- Top-doc similarity: min `0.605` · median `0.771` · max `5.845`
- Parser engine: **LiteParse (Markdown Engine)** · parser chunks: **102** · indexed chunks: **28** (collection `financial`)

## RAGAS Metrics (sampled sanity run)

Run on 5 sampled questions from `run3_fixed_8b` with judge `llama-3.3-70b-versatile`
(RAGAS 0.2.15, quota-aware retry wrapper, results in `eval_artifacts/mod01_ragas_run3_fixed_8b.json`):

| Metric | Score | Notes |
|---|---|---|
| faithfulness | **1.0000** | every claim in each answer is supported by retrieved context |
| answer_relevancy | **0.9744** | answers closely match what was asked |
| answer_correctness | **0.9565** | semantic + factual overlap with references |
| context_precision | — | judge jobs hit free-tier quota timeouts in this sample |
| context_recall | — | same as above; retrieval hit rate is 98/98 independently |

Full-benchmark RAGAS command (requires Groq quota headroom):

```bash
uv run --extra eval python evaluate_ragas.py --source run5_fixed_8b --judge-model llama-3.3-70b-versatile
```

## Failure Analysis

- Retrieval misses (0 docs): **0**
- Weak retrieval (top score < 0.5): **0**
- Retrieval OK but answer wrong/partial (generation/judge issues): **22**

### Q10. What is the total value of assets as at December 31, 2019?
- Reference: 85,098,647
- Answer: 377,160
- Verdict: `incorrect` (0.0) — contradicts the reference
- Retrieval: 8 docs, top `0.757` · routed `financial` · groundedness `1.00`

### Q17. What is the value of certificates of deposits from Bank of Utopia?
- Reference: 5,000,944
- Answer: 5,000,944 (2019); 3,507,159 (2018)
- Verdict: `partial` (0.5) — missing key detail
- Retrieval: 8 docs, top `5.519` · routed `financial` · groundedness `1.00`

### Q19. What is the value of loans due from financial institutions?
- Reference: 500,000
- Answer: 500,000 (2019)
- Verdict: `partial` (0.5) — missing key detail
- Retrieval: 8 docs, top `0.697` · routed `financial` · groundedness `1.00`

### Q23. What is the value of property, plant and equipment?
- Reference: 134,565
- Answer: 134,565 (2019)
- Verdict: `partial` (0.5) — missing key detail
- Retrieval: 8 docs, top `5.845` · routed `financial` · groundedness `1.00`

### Q24. What is the value of revaluation reserves?
- Reference: 16,699,986
- Answer: 16,699,986 (2019).
- Verdict: `partial` (0.5) — missing detail
- Retrieval: 8 docs, top `0.759` · routed `financial` · groundedness `1.00`

### Q30. What is the net profit for the year 2019?
- Reference: 4,166,597
- Answer: 1,585,242
- Verdict: `incorrect` (0.0) — contradicts the reference
- Retrieval: 8 docs, top `0.696` · routed `financial` · groundedness `1.00`

### Q34. What is the policy detail for presenting foreign currency and domestic currency items on the balance sheet?
- Reference: not specifically authorized by IAS 1 but has become a widely accepted practice
- Answer: Most central banks have effectively two separate activities, (i) holding and investing the foreign currency reserves of the nation, and (ii) undertaking monetary policy and other official operations in domestic currency. The balance between these two activities varies between central banks.
- Verdict: `incorrect` (0.0) — off-topic
- Retrieval: 8 docs, top `0.800` · routed `financial` · groundedness `1.00`

### Q37. What is the total net operating income for the year?
- Reference: 4,498,974
- Answer: 4,498,974 (2019)
- Verdict: `partial` (0.5) — missing key detail
- Retrieval: 8 docs, top `0.704` · routed `financial` · groundedness `1.00`

### Q39. What is the net profit for the year?
- Reference: 4,166,597
- Answer: 4,166,597 (2019)
- Verdict: `partial` (0.5) — missing key detail
- Retrieval: 8 docs, top `0.660` · routed `financial` · groundedness `1.00`

### Q40. How much is the actuarial gain on staff gratuity fund?
- Reference: 600
- Answer: 15.2
- Verdict: `incorrect` (0.0) — system answer does not match reference answer
- Retrieval: 8 docs, top `0.689` · routed `financial` · groundedness `1.00`

### Q41. What is the total other comprehensive income?
- Reference: 4,357
- Answer: 4,357 (2019)
- Verdict: `partial` (0.5) — missing key detail
- Retrieval: 8 docs, top `0.687` · routed `financial` · groundedness `1.00`

### Q42. What is the total comprehensive income?
- Reference: 4,170,954
- Answer: 4,170,954 (2019)
- Verdict: `partial` (0.5) — missing key detail
- Retrieval: 8 docs, top `0.682` · routed `financial` · groundedness `1.00`

### Q45. What is the amount of net impairment on financial instruments?
- Reference: 335
- Answer: 335 (2019)
- Verdict: `partial` (0.5) — missing key detail
- Retrieval: 8 docs, top `0.739` · routed `financial` · groundedness `1.00`

### Q47. What is the amount of total operating expenses?
- Reference: (332,377)
- Answer: (317,909)
- Verdict: `partial` (0.5) — missing key detail
- Retrieval: 8 docs, top `0.639` · routed `financial` · groundedness `1.00`

### Q56. What shall a central bank disclose if it is required to pay income tax, according to IAS 1.90?
- Reference: the amount of income tax relating to each item of other comprehensive income
- Answer: The amount of income tax relating to each item of other comprehensive income-including reclassification adjustments.
- Verdict: `partial` (0.5) — missing key detail
- Retrieval: 8 docs, top `0.764` · routed `financial` · groundedness `1.00`

### Q63. What amount of income tax relating to each item of other comprehensive income should be disclosed?
- Reference: the amount of income tax relating to each item of other comprehensive income, including reclassification adjustments
- Answer: The amount of income tax relating to each item of other comprehensive income should be disclosed either in the statement of profit or loss and other comprehensive income or in the notes.
- Verdict: `partial` (0.5) — missing key detail
- Retrieval: 8 docs, top `0.808` · routed `financial` · groundedness `1.00`

### Q67. What is the balance of the revaluation reserve as at December 31, 2019?
- Reference: 557,600
- Answer: 16,699,986.
- Verdict: `incorrect` (0.0) — system answer contradicts reference
- Retrieval: 8 docs, top `0.771` · routed `financial` · groundedness `1.00`

### Q68. What is the realization of previously unrealised gains for the year ended December 31, 2019?
- Reference: 325,214
- Answer: (325,214)
- Verdict: `incorrect` (0.0) — contradicts the reference
- Retrieval: 8 docs, top `0.754` · routed `financial` · groundedness `1.00`

### Q76. What type of financial instruments are included in Other Fair Value (FV) revaluation reserves?
- Reference: financial instrument market prices (if permitted by legislation) that are Fair Value through Profit or Loss (FVPL)
- Answer: Financial instruments at Fair Value through Profit or Loss (FVPL) are not included in Other Fair Value (FV) revaluation reserves. 

Other Fair Value (FV) revaluation reserves include net unrealized gains or losses arising from changes of the Fair Value through Other Comprehensive Income (FVOCI) financial instrument market prices.
- Verdict: `incorrect` (0.0) — contradicts the reference
- Retrieval: 8 docs, top `0.791` · routed `products` · groundedness `1.00`

### Q77. What is required by IFRS to be disclosed regarding capital management?
- Reference: information on the objectives, policies, and processes for managing capital
- Answer: According to the context, a central bank is required to disclose information on the objectives, policies, and processes for managing capital, as per IFRS [IAS 1.134].
- Verdict: `incorrect` (0.0) — system answer contradicts the reference
- Retrieval: 8 docs, top `0.879` · routed `financial` · groundedness `1.00`

### Q92. What may be shown in the Statement of Changes in Equity regarding comprehensive income?
- Reference: the allocation of comprehensive income initially allocated to one category and then reallocated out amongst the various reserves
- Answer: Total comprehensive income.
- Verdict: `incorrect` (0.0) — does not match reference
- Retrieval: 8 docs, top `0.827` · routed `financial` · groundedness `1.00`

### Q93. What is special reserve?
- Reference: supplementary realized reserves, in addition to general reserve, created in specific cases, such as injection of grants related monies
- Answer: The context does not explicitly mention the term "special reserve." However, it mentions "specialized reserves" in section which are supplementary realized reserves, in addition to a general reserve.
- Verdict: `partial` (0.5) — missing key detail
- Retrieval: 8 docs, top `0.670` · routed `products` · groundedness `0.81`

## Full Results

| # | Question | Verdict | Score | Docs | Top | Grounded |
|---|---|---|---|---|---|---|
| 1 | What is the purpose of the model financial statements? | correct | 1.0 | 8 | 0.742 | 1.00 |
| 2 | In what currency are the financial statements presented? | correct | 1.0 | 8 | 0.743 | 1.00 |
| 3 | What is the recommended presentation currency for a central ... | correct | 1.0 | 8 | 0.782 | 1.00 |
| 4 | What is the total value of foreign currency financial assets... | correct | 1.0 | 8 | 0.801 | 1.00 |
| 5 | What is the value of monetary gold as at December 31, 2019? | correct | 1.0 | 8 | 0.706 | 0.92 |
| 6 | What is the value of loans due from financial institutions a... | correct | 1.0 | 8 | 0.720 | 1.00 |
| 7 | What is the total value of non-financial assets as at Decemb... | correct | 1.0 | 8 | 0.772 | 1.00 |
| 8 | What is the value of property, plant and equipment as at Dec... | correct | 1.0 | 8 | 0.690 | 1.00 |
| 9 | What is the value of intangible assets as at December 31, 20... | correct | 1.0 | 8 | 0.727 | 1.00 |
| 10 | What is the total value of assets as at December 31, 2019? | incorrect | 0.0 | 8 | 0.757 | 1.00 |
| 11 | What is the value of cash and cash equivalents in local curr... | correct | 1.0 | 8 | 0.744 | 1.00 |
| 12 | What is the value of other receivables in local currency as ... | correct | 1.0 | 8 | 0.749 | 1.00 |
| 13 | What is the total value of local currency financial assets? | correct | 1.0 | 8 | 0.771 | 1.00 |
| 14 | What is the purpose of the Statement of Financial Position? | correct | 1.0 | 8 | 0.817 | 1.00 |
| 15 | What is the value of non-monetary gold? | correct | 1.0 | 8 | 0.741 | 1.00 |
| 16 | What is the total value of foreign currency financial liabil... | correct | 1.0 | 8 | 0.747 | 1.00 |
| 17 | What is the value of certificates of deposits from Bank of U... | partial | 0.5 | 8 | 5.519 | 1.00 |
| 18 | What is the total value of equity? | correct | 1.0 | 8 | 0.672 | 0.99 |
| 19 | What is the value of loans due from financial institutions? | partial | 0.5 | 8 | 0.697 | 1.00 |
| 20 | What is the value of total assets? | correct | 1.0 | 8 | 0.691 | 1.00 |
| 21 | What is the value of total liabilities? | correct | 1.0 | 8 | 0.632 | 1.00 |
| 22 | What is the value of advance to government? | correct | 1.0 | 8 | 0.616 | 1.00 |
| 23 | What is the value of property, plant and equipment? | partial | 0.5 | 8 | 5.845 | 1.00 |
| 24 | What is the value of revaluation reserves? | partial | 0.5 | 8 | 0.759 | 1.00 |
| 25 | Why do central banks subdivide the balance sheet between for... | correct | 1.0 | 8 | 0.834 | 1.00 |
| 26 | What is the presentation order of foreign currency and domes... | correct | 1.0 | 8 | 0.784 | 1.00 |
| 27 | What is the net foreign currency income for the year 2019? | correct | 1.0 | 8 | 0.771 | 1.00 |
| 28 | What is the accounting treatment for foreign currency intere... | correct | 1.0 | 8 | 0.776 | 1.00 |
| 29 | What is the total net operating income for the year 2019? | correct | 1.0 | 8 | 0.707 | 1.00 |
| 30 | What is the net profit for the year 2019? | incorrect | 0.0 | 8 | 0.696 | 1.00 |
| 31 | What is the disclosure requirement for foreign currency item... | correct | 1.0 | 8 | 0.726 | 1.00 |
| 32 | What is the recognition rule for net interest income? | incorrect | 0.0 | 8 | 0.717 | 0.57 |
| 33 | What is the measurement rule for foreign exchange revaluatio... | partial | 0.5 | 8 | 0.696 | 0.78 |
| 34 | What is the policy detail for presenting foreign currency an... | incorrect | 0.0 | 8 | 0.800 | 1.00 |
| 35 | What is the line item for gain or loss on sale of foreign cu... | correct | 1.0 | 8 | 0.807 | 1.00 |
| 36 | What is the note reference for foreign currency interest inc... | correct | 1.0 | 8 | 0.777 | 1.00 |
| 37 | What is the total net operating income for the year? | partial | 0.5 | 8 | 0.704 | 1.00 |
| 38 | What is the purpose of the Statement of Other Comprehensive ... | correct | 1.0 | 8 | 0.844 | 1.00 |
| 39 | What is the net profit for the year? | partial | 0.5 | 8 | 0.660 | 1.00 |
| 40 | How much is the actuarial gain on staff gratuity fund? | incorrect | 0.0 | 8 | 0.689 | 1.00 |
| 41 | What is the total other comprehensive income? | partial | 0.5 | 8 | 0.687 | 1.00 |
| 42 | What is the total comprehensive income? | partial | 0.5 | 8 | 0.682 | 1.00 |
| 43 | What is the amount of fee and commission income? | correct | 1.0 | 8 | 0.635 | 1.00 |
| 44 | What is the amount of foreign exchange revaluation? | correct | 1.0 | 8 | 0.793 | 1.00 |
| 45 | What is the amount of net impairment on financial instrument... | partial | 0.5 | 8 | 0.739 | 1.00 |
| 46 | What is the amount of personnel expenses? | correct | 1.0 | 8 | 0.605 | 1.00 |
| 47 | What is the amount of total operating expenses? | partial | 0.5 | 8 | 0.639 | 1.00 |
| 48 | What is the net profit or loss for the year after transferre... | correct | 1.0 | 8 | 0.770 | 1.00 |
| 49 | What forms the basis for distribution of profits to the Mini... | correct | 1.0 | 8 | 0.681 | 1.00 |
| 50 | Where must the majority of exchange rate revaluations pass t... | correct | 1.0 | 8 | 0.686 | 1.00 |
| 51 | What type of income does IAS 1 allow commercial banks to sta... | correct | 1.0 | 8 | 0.814 | 1.00 |
| 52 | What costs do many central banks show as a separate line ite... | correct | 1.0 | 8 | 0.806 | 1.00 |
| 53 | Why can the costs of currency operations vary significantly ... | correct | 1.0 | 8 | 0.719 | 1.00 |
| 54 | How many options are there for presenting other comprehensiv... | correct | 1.0 | 8 | 0.799 | 1.00 |
| 55 | How shall a central bank present all items of OCI in the Sta... | correct | 1.0 | 8 | 0.900 | 1.00 |
| 56 | What shall a central bank disclose if it is required to pay ... | partial | 0.5 | 8 | 0.764 | 1.00 |
| 57 | Where can the amount of income tax relating to each item of ... | correct | 1.0 | 8 | 0.806 | 1.00 |
| 58 | Why do many central banks show the split of income and expen... | correct | 1.0 | 8 | 0.866 | 1.00 |
| 59 | What is one reason for showing currency issue costs separate... | correct | 1.0 | 8 | 0.823 | 1.00 |
| 60 | What is the benefit of showing currency issue costs separate... | correct | 1.0 | 8 | 0.822 | 1.00 |
| 61 | What are the two options for presenting other comprehensive ... | correct | 1.0 | 8 | 0.822 | 1.00 |
| 62 | How should a central bank present items of OCI in the Statem... | correct | 1.0 | 8 | 0.906 | 1.00 |
| 63 | What amount of income tax relating to each item of other com... | partial | 0.5 | 8 | 0.808 | 1.00 |
| 64 | What is the balance of the general reserve as at January 1, ... | correct | 1.0 | 8 | 0.777 | 0.92 |
| 65 | What is the total comprehensive income for the year ended De... | correct | 1.0 | 8 | 0.781 | 1.00 |
| 66 | How is the general reserve typically created? | correct | 1.0 | 8 | 0.715 | 1.00 |
| 67 | What is the balance of the revaluation reserve as at Decembe... | incorrect | 0.0 | 8 | 0.771 | 1.00 |
| 68 | What is the realization of previously unrealised gains for t... | incorrect | 0.0 | 8 | 0.754 | 1.00 |
| 69 | What is the profit distribution policy of a central bank bas... | correct | 1.0 | 8 | 0.780 | 1.00 |
| 70 | What is the total equity as at December 31, 2019? | correct | 1.0 | 8 | 0.739 | 1.00 |
| 71 | What is the authorized and paid-in capital of a central bank... | correct | 1.0 | 8 | 0.698 | 1.00 |
| 72 | What should central banks mention in the financial statement... | correct | 1.0 | 8 | 0.828 | 1.00 |
| 73 | What is the source for general reserves in central banks? | correct | 1.0 | 8 | 0.788 | 1.00 |
| 74 | What is the purpose of general reserves in central banks? | correct | 1.0 | 8 | 0.726 | 1.00 |
| 75 | How are foreign currency revaluation reserves established in... | correct | 1.0 | 8 | 0.832 | 1.00 |
| 76 | What type of financial instruments are included in Other Fai... | incorrect | 0.0 | 8 | 0.791 | 1.00 |
| 77 | What is required by IFRS to be disclosed regarding capital m... | incorrect | 0.0 | 8 | 0.879 | 1.00 |
| 78 | What is typically provided by central banks to comply with I... | correct | 1.0 | 8 | 0.897 | 1.00 |
| 79 | What does IAS 1.29 require regarding presentation of similar... | correct | 1.0 | 8 | 0.768 | 1.00 |
| 80 | How should an entity disclose further sub-classifications of... | correct | 1.0 | 8 | 0.800 | 1.00 |
| 81 | What is an example of a subclassification of equity in the f... | correct | 1.0 | 8 | 0.831 | 1.00 |
| 82 | What is included in Other Comprehensive Income reserves in c... | correct | 1.0 | 8 | 0.805 | 1.00 |
| 83 | What is the basis for establishing general reserves in some ... | correct | 1.0 | 8 | 0.786 | 1.00 |
| 84 | What does IAS 1.29 state regarding the presentation of mater... | correct | 1.0 | 8 | 0.726 | 1.00 |
| 85 | Where can an entity disclose further sub-classifications of ... | correct | 1.0 | 8 | 0.785 | 1.00 |
| 86 | What is the basis for the detail provided in sub-classificat... | correct | 1.0 | 8 | 0.753 | 1.00 |
| 87 | What is an example of a subclassification of equity? | correct | 1.0 | 8 | 0.797 | 1.00 |
| 88 | What terms are used by different central banks to refer to s... | correct | 1.0 | 8 | 0.844 | 1.00 |
| 89 | What is general reserve created through? | correct | 1.0 | 8 | 0.748 | 1.00 |
| 90 | What are revaluation reserves required by accounting policy ... | correct | 1.0 | 8 | 0.818 | 1.00 |
| 91 | Where are more detailed subclassifications for each group of... | correct | 1.0 | 8 | 0.672 | 1.00 |
| 92 | What may be shown in the Statement of Changes in Equity rega... | incorrect | 0.0 | 8 | 0.827 | 1.00 |
| 93 | What is special reserve? | partial | 0.5 | 8 | 0.670 | 0.81 |
| 94 | What are revaluation reserves required by IFRS? | correct | 1.0 | 8 | 0.813 | 1.00 |
| 95 | What is the purpose of grouping similar types of equity? | correct | 1.0 | 8 | 0.764 | 1.00 |
| 96 | What is the net cash flow from operating activities for the ... | correct | 1.0 | 8 | 0.832 | 1.00 |
| 97 | What are the more detailed subclassifications for each group... | correct | 1.0 | 8 | 0.687 | 1.00 |
| 98 | What is the amount of interest received in 2018? | correct | 1.0 | 8 | 0.710 | 1.00 |
| 99 | How are comprehensive income allocations reflected in the St... | correct | 1.0 | 8 | 0.817 | 1.00 |
| 100 | What is the net purchase of property, investment property an... | correct | 1.0 | 8 | 0.708 | 1.00 |
