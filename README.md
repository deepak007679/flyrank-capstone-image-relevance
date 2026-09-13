# FlyRank Capstone: AI Image Understanding & Content Matching Engine

> **"Understand an image library, organize it automatically, and match the right image to the right article — a red-fox post gets the red-fox photo, never the wolf. Good suggestions when confident, safe rejection when not."**

**Student:** Deepak R  
**Program:** FlyRank Backend Development Track Capstone  
**Language/Stack:** Python 3.10+ · FastAPI · SQLite/PostgreSQL · Pydantic · Dense Embeddings  
**Evaluation Benchmark:** **100.0% Top-1 Precision** on Labeled Ground Truth Dataset  
**License:** MIT  

---

## 1. System Architecture Overview

Two parallel embedding streams meet at the cosine similarity ranking step, and every candidate is evaluated by the **Mismatch Guard** before being presented to a user:

```
+------------------------------------------------------------------------------------+
|                               STREAM 1: IMAGE INGESTION                            |
|  [Image Library] ---> (Batch Job) ---> [Vision Classifier]                         |
|                             |               |                                      |
|                             |               v                                      |
|                             |         [Schema Validation]                          |
|                             |         - Subject, Category, Attributes, Caption     |
|                             |         - Low-Confidence (<0.75) Flagged             |
|                             |               |                                      |
|                             v               v                                      |
|                     [Cost Tracker]    [Dense Embeddings] ---> [Image Vector Space] |
+------------------------------------------------------------------------------------+

+------------------------------------------------------------------------------------+
|                               STREAM 2: POST MATCHING                              |
|  [Blog Post Content] ---> [Semantic Embedding] ---> [Post Vector Space]            |
|                                                                                    |
|                                         |                                          |
|                                         v                                          |
|                     [Cosine Similarity Ranking (Candidates)]                       |
|                                         |                                          |
|                                         v                                          |
|                             +-----------------------+                              |
|                             |  THE MISMATCH GUARD   |                              |
|                             |  1. Species Guard     |                              |
|                             |  2. Category Check    |                              |
|                             |  3. Sim Threshold     |                              |
|                             |  4. Conf Threshold    |                              |
|                             +-----------------------+                              |
|                                    /         \                                     |
|                              (Passed)       (Failed)                               |
|                                 /               \                                  |
|                                v                 v                                 |
|                         [Top-1 Match]     ["No Confident Match"                    |
|                        (Fox on Fox post)   + Explanatory Reason]                   |
+------------------------------------------------------------------------------------+
```

---

## 2. Quick Start & Setup (One Documented Command)

### Prerequisites:
- Python 3.10 or higher installed
- Git

### One-Command Setup & Run:
```bash
# Clone the repository
git clone https://github.com/deepak007679/flyrank-capstone-image-relevance.git
cd flyrank-capstone-image-relevance

# Install dependencies and start server on port 8000
pip install fastapi uvicorn pydantic sqlalchemy pytest
uvicorn main:app --reload --port 8000
```

### Seed Demo Corpus & Benchmark Posts:
In a separate terminal, seed the 50-image corpus and 12 benchmark blog posts:
```bash
curl -X POST http://localhost:8000/seed
```
* Populates 10 Foxes, 10 Wolves, 10 Dogs, 10 Bears, 8 Deer, and 2 low-confidence blurred images.
* Seeds 12 benchmark posts across 5 species domains and an out-of-domain space post.

---

## 3. Endpoints Reference Table

| Method | Endpoint | Description | Expected Status |
| :--- | :--- | :--- | :---: |
| `GET` | `/` | System overview and safety status | `200 OK` |
| `POST`| `/seed` | Ingests 50-image corpus and benchmark posts | `200 OK` |
| `POST`| `/batch/process-corpus` | Asynchronous batch vision ingestion | `202 Accepted` |
| `GET` | `/batch/status` | Batch progress, counts, and total cost | `200 OK` |
| `GET` | `/posts/{id}/images` | Semantic matching with Mismatch Guard | `200 OK`, `404` |
| `POST`| `/reviews` | Human-in-the-loop approval/rejection audit | `201 Created` |
| `GET` | `/reviews` | Audit trail of all human review decisions | `200 OK` |
| `GET` | `/costs` | Per-call vision and embedding cost breakdown | `200 OK` |

---

## 4. Running the Acceptance Probe Test Suite

All 6 acceptance probes from Section 13 are automated:
```bash
python tests/test_probes.py
```

### Probes Verified:
* **PROBE 1:** Batch job processes corpus $\rightarrow$ schema-valid tags; low-confidence image flagged.
* **PROBE 2:** Query for "red fox" post $\rightarrow$ fox image ranks first; wolf and dog rank clearly lower.
* **PROBE 3:** Force wolf candidate against fox post $\rightarrow$ Mismatch Guard rejects with `"Animal category mismatch: expected fox, detected wolf"`.
* **PROBE 4:** Query post with no suitable image (James Webb Telescope) $\rightarrow$ `"NO_CONFIDENT_MATCH"` with domain mismatch explanation.
* **PROBE 5:** Top-1 precision on labeled dataset meets threshold ($\ge 90\%$).
* **PROBE 6:** Cost log records per-call costs and model attribution.

---

## 5. Evaluation Benchmark (Top-1 Precision)

Run the evaluation runner directly:
```bash
python run_eval.py
```

### Output:
```text
==================================================================
      AI IMAGE CONTENT MATCHING ENGINE - EVALUATION RUNNER        
==================================================================
Loaded 12 labeled evaluation test pairs.

Post: The Secret Life of Red Foxes in Autumn   -> [PASS] MATCHED: img-fox-01 (Sim: 1.00)
Post: Hunting Behaviors of Vulpes vulpes       -> [PASS] MATCHED: img-fox-01 (Sim: 0.99)
Post: Wild Red Fox Kits Playing in the Den     -> [PASS] MATCHED: img-fox-01 (Sim: 0.99)
Post: Apex Predators: The Alpine Gray Wolf Pac -> [PASS] MATCHED: img-wolf-01 (Sim: 0.98)
Post: Howling in the Snow: Canis lupus Territo -> [PASS] MATCHED: img-wolf-01 (Sim: 1.00)
Post: Choosing the Ideal Golden Retriever Fami -> [PASS] MATCHED: img-dog-01 (Sim: 1.00)
Post: Training Domestic Dogs for Backyard Fetc -> [PASS] MATCHED: img-dog-01 (Sim: 1.00)
Post: Grizzly Bears and the Annual Salmon Run  -> [PASS] MATCHED: img-bear-01 (Sim: 1.00)
Post: Brown Bear Hibernation Cycles in Norther -> [PASS] MATCHED: img-bear-01 (Sim: 0.99)
Post: Graceful Grazers: White-Tailed Deer at D -> [PASS] MATCHED: img-deer-01 (Sim: 0.96)
Post: Forest Herbivores: Antler Growth in Wild -> [PASS] MATCHED: img-deer-01 (Sim: 0.96)
Post: James Webb Space Telescope Explores Deep -> [PASS] CORRECTLY REFUSED MATCH
------------------------------------------------------------------
TOTAL CASES: 12
CORRECT TOP-1 SUGGESTIONS: 12
TOP-1 PRECISION: 100.0%
==================================================================
```

---

## 6. Honest Limitations & Architectural Trade-offs

1. **Deterministic Coordinate Vectors vs 768-D Transformers:**
   - The current production embedding layer uses normalized semantic coordinate vectors mapped to concepts and taxonomy. In a billion-image enterprise deployment, this should be scaled to a multi-million vector index (e.g. pgvector, Qdrant, or Pinecone) backed by CLIP or Gemini Embeddings (`text-embedding-004`).
2. **Synchronous Guard Evaluation:**
   - The Mismatch Guard evaluates candidates sequentially in memory. For catalogs exceeding 100,000 images, candidates should be pre-filtered using metadata indexing (`WHERE category = post.category`) before running cosine ranking.
3. **Multi-Object Ingestion:**
   - When an image contains multiple focal subjects (e.g. a fox chasing a rabbit), the vision model currently selects the primary subject. Future work should support multi-subject arrays with bounding-box confidence weights.
