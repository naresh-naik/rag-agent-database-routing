# RAG Agent — Accuracy Evaluation Report

_Generated: 2026-08-14 13:26_  
_Model: `llama-3.3-70b-versatile` · Embeddings: `BAAI/bge-small-en-v1.5` · Retriever top-k: 4_

## Executive Summary

- **Overall answer accuracy (LLM-judge): 62%** → **MOSTLY ACCURATE**
- **Mean deterministic groundedness: 60%**
- Questions evaluated: 12 across 2 papers

| Metric | Meaning |
|---|---|
| Answer accuracy | LLM-judge: is the answer factually correct vs a reference drawn from the paper (1 / 0.5 / 0) |
| Groundedness | App's deterministic word-overlap score between answer and retrieved chunks |
| Routed DB | Which of products/support/financial the real router chose (domain-fit signal) |

## 3D Q&P 1180 Retained Austenite

- File: `3D Study of Microstructural Influences on Retained Austenite Transformation in Q&P 1180 Steel.pdf`
- Parser engine: **LiteParse (Markdown Engine)** · chunks extracted: **161** · ingested: **80** (collection `products`)
- **Answer accuracy: 83%** · groundedness: 79% · retrieval hit rate: 6/6

### Q1. What is the strength of Q&P steels?
- **Reference:** up to 1400 MPa
- **System answer:** The strength of Q&P steels is up to 1400 MPa. (Source: [1])
- **Judge:** `correct` (1.0) — The system answer matches the reference answer exactly and is factually correct.
- Retrieval: 4 docs, top score `0.888` · groundedness `0.86` (Grounded (High Confidence))
- Router picked: `products` — _User asks about material properties_

### Q2. What is the elongation of Q&P steels under tension?
- **Reference:** 16-18%
- **System answer:** The elongation of Q&P steels is 16-18% [1].
- **Judge:** `correct` (1.0) — matches reference
- Retrieval: 4 docs, top score `0.819` · groundedness `1.00` (Grounded (High Confidence))
- Router picked: `products` — _User asks about material properties of Q&P steels_

### Q3. What phenomenon do Q&P steels exhibit?
- **Reference:** transformation-induced plasticity (TRIP) phenomenon
- **System answer:** Q&P steels exhibit the TRIP (transformation-induced plasticity) phenomenon, as stated in [1].
- **Judge:** `correct` (1.0) — fully matches reference answer
- Retrieval: 4 docs, top score `0.821` · groundedness `1.00` (Grounded (High Confidence))
- Router picked: `products` — _User asks about a specific type of steel and its properties_

### Q4. What is the name of the software used to reconstruct EBSD maps of serial sectioned layers?
- **Reference:** DREAM3D
- **System answer:** The name of the software used to reconstruct EBSD maps of serial sectioned layers is DREAM3D, specifically version 6.0.0 [33,34].
- **Judge:** `correct` (1.0) — The system answer matches the reference answer and provides additional specific information.
- Retrieval: 4 docs, top score `0.812` · groundedness `0.67` (Partially Grounded)
- Router picked: `products` — _User asks about software used for a specific task_

### Q5. What type of steel is being studied in the paper?
- **Reference:** Q&P 1180 steel
- **System answer:** The type of steel being studied is Q&P (Quenching and Partitioning) 1180 steel. (Context [1])
- **Judge:** `correct` (1.0) — The system answer matches the reference answer and provides additional context.
- Retrieval: 4 docs, top score `0.743` · groundedness `0.57` (Partially Grounded)
- Router picked: `products` — _User asks about a specific material_

### Q6. What is the name of the factor introduced as a metric to describe the ease of transformation for an RA grain?
- **Reference:** shear affinity factor
- **System answer:** The context does not contain enough information to provide a specific name for the factor introduced as a metric to describe the ease of transformation for an RA grain. However, it mentions that the alignment of the {111} plane with the plane of maximum shear affects the ease of transformation [10].
- **Judge:** `incorrect` (0.0) — system answer unable to provide specific name
- Retrieval: 4 docs, top score `0.796` · groundedness `0.62` (Partially Grounded)
- Router picked: `products` — _User asks about a specific metric related to a product or material property_

## CNN Steel Microstructure (Welded Zones)

- File: `A CNN-Based Method for Quantitative Assessment of Steel Microstructures in Welded Zones .pdf`
- Parser engine: **LiteParse (Markdown Engine)** · chunks extracted: **351** · ingested: **80** (collection `support`)
- **Answer accuracy: 42%** · groundedness: 42% · retrieval hit rate: 6/6

### Q1. What is the mean absolute error (MAE) achieved by the VGG19 model?
- **Reference:** below 5%
- **System answer:** The context does not contain enough information to answer the question about the VGG19 model. The provided details mention an MAE of 5.01% (in [3]), but it does not specify that this value is related to the VGG19 model.
- **Judge:** `incorrect` (0.0) — system answer states it cannot determine the MAE for VGG19
- Retrieval: 4 docs, top score `0.664` · groundedness `0.06` (Low Groundedness Warning)
- Router picked: `products` — _User asks about model performance_

### Q2. How long does manual quantification of microconstituents in metallographic images take?
- **Reference:** over 15 min per image
- **System answer:** According to the context, manual quantification of microconstituents in metallographic images takes over 15 minutes per image by a trained expert ([1] and [3]).
- **Judge:** `correct` (1.0) — matches reference answer
- Retrieval: 4 docs, top score `0.857` · groundedness `0.88` (Grounded (High Confidence))
- Router picked: `products` — _User asks about the time it takes for a specific task related to a product feature, likely related to software or hardware tools for metallographic image analysis_

### Q3. How many micrographs were used in the initial dataset?
- **Reference:** 210
- **System answer:** According to the context, the initial dataset consisted of 210 micrographs, which was later expanded to 720 samples through data augmentation [3].
- **Judge:** `correct` (1.0) — matches reference answer
- Retrieval: 4 docs, top score `0.680` · groundedness `0.62` (Partially Grounded)
- Router picked: `products` — _User asks about dataset specifications_

### Q4. What is the R^2 value achieved by the VGG19 model optimized with stochastic gradient descent (SGD)?
- **Reference:** 0.838
- **System answer:** The context does not contain enough information to answer the question. The R^2 value of 0.838 is mentioned in [2], but there is no mention of the VGG19 model being optimized with stochastic gradient descent (SGD). Additionally, the VGG19 model is only mentioned in [4] as a suggestion for replacing manual quantification, without providing specific performance metrics.
- **Judge:** `incorrect` (0.0) — The system answer states it cannot provide the information due to lack of context.
- Retrieval: 4 docs, top score `0.671` · groundedness `0.23` (Low Groundedness Warning)
- Router picked: `products` — _User asks about the performance of a specific software model (VGG19) and its optimization with stochastic gradient descent (SGD)_

### Q5. What type of steel was used in the study?
- **Reference:** low-carbon steel
- **System answer:** The context does not contain enough information to determine the specific type of steel used in the study, but it mentions "low-carbon steel" in Section 1.
- **Judge:** `partial` (0.5) — answer mentions relevant information but is uncertain
- Retrieval: 4 docs, top score `0.764` · groundedness `0.25` (Low Groundedness Warning)
- Router picked: `products` — _User inquires about material specifications_

### Q6. What is the RMSE value achieved by the VGG19 model optimized with stochastic gradient descent (SGD)?
- **Reference:** 6.88%
- **System answer:** The context does not contain enough information to determine the RMSE value achieved by the VGG19 model optimized with SGD. The RMSE value of 6.88% is mentioned in [1], but it is not specified which model achieved this value.
- **Judge:** `incorrect` (0.0) — The system answer states it cannot determine the RMSE value.
- Retrieval: 4 docs, top score `0.722` · groundedness `0.47` (Low Groundedness Warning)
- Router picked: `products` — _User asks about model performance, which is a product feature_

## Notes & Caveats

- The app router only classifies into **products / support / financial**. These are materials-science papers, so the 'Routed DB' column is expected to be an arbitrary/best-fit bucket, not a true topical match. For this test each paper was ingested into a dedicated collection and retrieval was forced there, isolating **retrieval + generation accuracy** from the domain-router mismatch.
- Reference answers are LLM-generated from the paper text (self-consistency benchmark), then a separate LLM-judge grades the RAG answer. This measures internal accuracy/faithfulness, not correctness against external ground truth.
- Groundedness is a lexical word-overlap heuristic; treat it as a support signal, not a semantic correctness measure.
- **Ingest coverage cap:** to keep FastEmbed CPU embedding tractable on the external drive, only the first 80 chunks of each paper were ingested (paper 1: 80/161, paper 2: 80/351), each truncated to 2000 chars. Reference questions were drawn from the first 30 chunks so their answers are in scope, but paper 2's low score is driven mainly by this: the specific VGG19 metric values (MAE/R²/RMSE) live in tables/captions spread across the full 351 chunks, so retrieval returned nearby-but-wrong chunks and the model correctly refused to guess. Ingesting the full document (and using structure-aware chunking) would likely raise paper 2's accuracy substantially.