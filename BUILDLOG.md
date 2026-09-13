# BUILDLOG.md — AI Usage & Engineering Decisions

**Student:** Deepak R  
**Project:** AI Image Understanding & Content Matching Engine  
**Program:** FlyRank Backend Internship Capstone  

---

## 1. Overview of AI Partnership

This project was built following the **4D AI Fluency Framework** (Delegation, Description, Discernment, Diligence). Claude and Gemini served as thinking partners for vector coordinate modeling, vision prompt construction, and adversarial testing against the Mismatch Guard.

---

## 2. Where AI Helped Most

1. **Structured Vision Prompting & Pydantic Validation:**
   - AI formulated the Pydantic schema for `ImageVisionMetadata`, ensuring all vision classifications require discrete fields (`subject`, `category`, `attributes`, `caption`, `confidence`).
2. **Dense Vector Mapping for Semantic Synonyms:**
   - Guided the mathematical representation ensuring Latin names (e.g., *Vulpes vulpes*) map directly to common English concepts (*red fox*) in normalized vector space.
3. **Comprehensive Evaluation Dataset Construction:**
   - Assisted in generating 12 diverse evaluation pairings across 5 animal categories and out-of-domain edge cases.

---

## 3. Where AI Was Wrong or Hallucinated (And What I Fixed)

### Error 1: Blind Cosine Ranking Without a Guard
* **What AI initially suggested:** Simply ranking images by cosine similarity and returning `candidates[0]`.
* **Why it failed:** When querying for "red fox in the forest", a high-resolution photo of a gray wolf scored 0.88 similarity because both are woodland canids. The AI blindly recommended the wolf for the fox post!
* **My fix:** Built the dedicated **Mismatch Guard** module. Before any image is presented to the user, the guard cross-checks species and domain categories. If an animal mismatch is detected (fox vs wolf), the guard immediately rejects it with a human-readable explanation (`Animal category mismatch: expected fox, detected wolf`), preventing silent errors.

### Error 2: Slicing `top_k` Before Guard Validation
* **What AI generated:** The AI sliced the top 5 raw cosine candidates before checking the guard:
  ```python
  # AI's flawed initial ranking
  top_candidates = scored_candidates[:top_k]
  valid_match = next((c for c in top_candidates if c.guard_status == "ACCEPTED"), None)
  ```
* **Why it failed:** In cases where several visually similar wolves scored higher raw cosine numbers than dogs on a domestic dog post, all 5 top slots were occupied by rejected wolves! The valid dog image at rank 6 was never checked, returning false "No Confident Match".
* **My fix:** Changed the evaluation loop to evaluate the guard across the candidate space and prioritize valid matches that cleared all safety rules before building the top candidate list.

---

## 4. Explaining Key Code Lines (Evaluator Spot-Check Prep)

* **Mismatch Guard Decision Function (`services/engine.py:160-209`):**
  > *"The Mismatch Guard decouples raw mathematical vector proximity from semantic truth. A wolf and a fox share high vector similarity because both are wild carnivores in forests. The guard introduces a strict species and category barrier. If the post target is 'fox' and the vision classifier detected 'wolf', the guard overrides the high similarity score and flags the pairing as rejected."*

* **Normalized Cosine Similarity (`services/engine.py:65-72`):**
  > *"We calculate the dot product of two L2-normalized unit vectors. Because the vectors have unit length ($\|v\| = 1$), the dot product equals the cosine of the angle between them, providing an exact metric bounded between 0.0 and 1.0."*

* **Low-Confidence Flagging (`main.py:85-92`):**
  > *"Rather than letting an ambiguous image with a low-confidence vision tag slip into production recommendations, the batch worker checks if `confidence < 0.75`. If true, `is_flagged_low_confidence` is set to True. The mismatch guard automatically rejects flagged images from automated pairings."*
