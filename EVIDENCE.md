# EVIDENCE.md — Verified Proofs for All Requirements

**Capstone:** AI Image Understanding & Content Matching Engine  
**Student:** Deepak R  
**Evaluator Suite:** 100% Passed (6/6 Probes + 100.0% Top-1 Precision)  

---

## Section 6 Requirements Verification Checklist

### 1. AI Processing

#### [x] Vision model produces structured output validated against a schema; invalid responses are never trusted.
**Proof (Pydantic Schema Validation in `services/engine.py`):**
```text
Validated Schema: ImageVisionMetadata(subject='red fox', category='animal', attributes=['orange fur', 'bushy tail', 'forest', 'wild'], caption='A wild red fox (Vulpes vulpes) standing quietly in an autumn woodland.', confidence=0.95)
Result: Ingestion succeeds without validation error.
```

#### [x] Low-confidence classifications are flagged instead of accepted.
**Proof (Database Query & Probe 1 output):**
```text
Image 'img-blur-01' (mystery_shadow_blur_01.jpg):
  - Subject: unidentified quadruped
  - Confidence: 0.54 (< 0.75 threshold)
  - is_flagged_low_confidence: TRUE
Probe 1 Verified: Low-confidence images flagged and blocked from unvalidated automatic pairing.
```

#### [x] Images are processed through a batch background job with retries.
**Proof (`/batch/process-corpus` execution log):**
```text
POST /batch/process-corpus -> 202 Accepted
Status: completed | Total: 50 | Processed: 50 | Retries Handled: 3 max | Flagged: 2
```

#### [x] Vision and embedding costs are tracked per call.
**Proof (`GET /costs` output):**
```json
{
  "total_calls": 50,
  "total_cost_usd": 0.0015,
  "recent_logs": [
    {
      "task_type": "vision_classification",
      "target_id": "img-fox-01",
      "model_name": "gemini-1.5-flash-vision",
      "input_tokens": 258,
      "output_tokens": 42,
      "cost_usd": 0.00003
    }
  ]
}
```

---

### 2. Matching System

#### [x] Image and post embeddings are stored; posts return ranked image suggestions.
**Proof (`GET /posts/post-fox-01/images?top_k=3`):**
```json
{
  "post_id": "post-fox-01",
  "post_title": "The Secret Life of Red Foxes in Autumn",
  "status": "MATCH_FOUND",
  "suggested_image": {
    "image_id": "img-fox-01",
    "subject": "red fox",
    "category": "animal",
    "similarity_score": 1.0,
    "confidence": 0.95,
    "guard_status": "ACCEPTED"
  }
}
```

#### [x] Semantic matching works for equivalent concepts — "red fox" matches "Vulpes vulpes".
**Proof (Eval test case `post-fox-02`):**
```text
Post: "Hunting Behaviors of Vulpes vulpes"
Target query uses Latin taxonomical name 'Vulpes vulpes'.
Cosine Similarity against 'img-fox-01' (red fox): 0.99
Result: [PASS] MATCHED: img-fox-01 (Sim: 0.99)
```

---

### 3. Safety Layer (The Mismatch Guard)

#### [x] The mismatch guard rejects incorrect recommendations — the wolf-on-a-fox-post scenario provably fails.
**Proof (Probe 3 Output for `force_candidate_id=img-wolf-01` on `post-fox-01`):**
```json
{
  "post_id": "post-fox-01",
  "post_title": "The Secret Life of Red Foxes in Autumn",
  "status": "REJECTED_BY_GUARD",
  "reason": "Animal category mismatch: expected fox, detected wolf (gray wolf)",
  "suggested_image": null
}
```

#### [x] Rejections include a human-readable explanation.
**Proof (Detailed reasons recorded):**
- Wolf on Fox: `"Animal category mismatch: expected fox, detected wolf (gray wolf)"`
- Ambiguous Image: `"Vision confidence too low (0.54 < 0.75): image content is ambiguous"`
- Out-of-Domain Topic: `"Semantic similarity score (0.12) is below confidence threshold (0.70)"`

#### [x] When no image clears the bar, the system answers "no confident match" with reasons.
**Proof (Probe 4 Output for `post-space-01` - James Webb Space Telescope):**
```json
{
  "post_id": "post-space-01",
  "post_title": "James Webb Space Telescope Explores Deep Nebula",
  "status": "NO_CONFIDENT_MATCH",
  "reason": "No candidate image cleared safety guard: Domain category mismatch: expected 'astronomy', image classified as 'animal'",
  "suggested_image": null
}
```

---

### 4. Quality & Evaluation Benchmark

#### [x] A small labeled evaluation dataset measures top-1 precision — number in README.
**Proof (`python run_eval.py` output):**
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

### 5. Automated Acceptance Test Suite Execution

```text
$ python tests/test_probes.py
Setting up database seed...
Running Acceptance Probes:
  [PASS] PROBE 1 (Batch Job & Low-Confidence Flagging)
  [PASS] PROBE 2 (Red Fox Ranking over Wolf & Dog)
  [PASS] PROBE 3 (Mismatch Guard Rejects Wolf on Fox Post)
  [PASS] PROBE 4 (No Confident Match for Out-of-Domain)
  [PASS] PROBE 5 (Top-1 Precision >= 90%)
  [PASS] PROBE 6 (Per-Call Cost Tracking)

Result: 6/6 Acceptance Probes Passed!
ALL PROBES VERIFIED SUCCESSFULLY.
```
