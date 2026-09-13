"""
Acceptance Probe Verification Test Suite
FlyRank Capstone: AI Image Understanding & Content Matching Engine
Author: Deepak R

Verifies all 6 acceptance probes specified in Section 13:
- PROBE 1: Batch job runs on corpus -> schema-valid tags; low-confidence image flagged.
- PROBE 2: Query for 'red fox' article -> fox image ranks first; wolf and dog rank lower.
- PROBE 3: Force wolf as candidate for fox post -> guard rejects with category mismatch reason.
- PROBE 4: Query post with no suitable image -> 'no confident match' with explanation.
- PROBE 5: Eval script reports top-1 precision on labeled dataset.
- PROBE 6: Cost log attributes every vision/embedding call with cost entries.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import (
    seed_corpus, match_images_for_post, get_costs,
    SessionLocal, ImageRecord, CostRecord
)
from run_eval import run_evaluation


def setup_module():
    db = SessionLocal()
    try:
        seed_corpus(db)
    finally:
        db.close()


# -----------------------------------------------------------------------------
# PROBE 1: Batch job tags corpus + flags low-confidence images
# -----------------------------------------------------------------------------
def test_probe_1_batch_processing_and_low_confidence_flagging():
    db = SessionLocal()
    try:
        images = db.query(ImageRecord).all()
        assert len(images) >= 40, "Corpus must contain at least 40 images"
        
        for img in images:
            assert img.subject is not None, f"Image {img.id} missing subject"
            assert img.category is not None, f"Image {img.id} missing category"
            assert isinstance(img.attributes, list), f"Image {img.id} attributes must be list"
            assert len(img.attributes) > 0, f"Image {img.id} attributes cannot be empty"
            assert img.caption is not None, f"Image {img.id} missing caption"
            assert 0.0 <= img.confidence <= 1.0, f"Image {img.id} invalid confidence range"

        flagged = db.query(ImageRecord).filter(ImageRecord.is_flagged_low_confidence == True).all()
        assert len(flagged) >= 1, "At least one low-confidence image must be flagged"
        assert flagged[0].confidence < 0.75, "Flagged image must have confidence below 0.75"
    finally:
        db.close()


# -----------------------------------------------------------------------------
# PROBE 2: 'Red fox' article ranks fox first; wolf and dog clearly lower
# -----------------------------------------------------------------------------
def test_probe_2_red_fox_ranking_over_wolf_and_dog():
    db = SessionLocal()
    try:
        data = match_images_for_post(post_id="post-fox-01", top_k=10, db=db)
        assert data.status == "MATCH_FOUND"
        
        suggested = data.suggested_image
        assert suggested is not None
        assert "fox" in suggested.subject.lower(), "Top suggested image must be a fox"
        fox_score = suggested.similarity_score

        wolf_candidate = next((c for c in data.all_candidates if "wolf" in c.subject.lower()), None)
        dog_candidate = next((c for c in data.all_candidates if "dog" in c.subject.lower()), None)

        if wolf_candidate:
            assert fox_score > wolf_candidate.similarity_score, "Fox image must rank higher than wolf"
        if dog_candidate:
            assert fox_score > dog_candidate.similarity_score, "Fox image must rank higher than dog"
    finally:
        db.close()


# -----------------------------------------------------------------------------
# PROBE 3: Force wolf as candidate for fox post -> guard rejects with explanation
# -----------------------------------------------------------------------------
def test_probe_3_mismatch_guard_rejects_wolf_on_fox_post():
    db = SessionLocal()
    try:
        data = match_images_for_post(post_id="post-fox-01", force_candidate_id="img-wolf-01", db=db)
        assert data.status == "REJECTED_BY_GUARD"
        assert data.suggested_image is None
        
        reason = (data.reason or "").lower()
        assert "mismatch" in reason or "wolf" in reason or "fox" in reason
        assert "expected fox, detected wolf" in reason
    finally:
        db.close()


# -----------------------------------------------------------------------------
# PROBE 4: Query post with no suitable image -> 'no confident match' + reasons
# -----------------------------------------------------------------------------
def test_probe_4_no_confident_match_when_no_image_clears_bar():
    db = SessionLocal()
    try:
        data = match_images_for_post(post_id="post-space-01", db=db)
        assert data.status == "NO_CONFIDENT_MATCH"
        assert data.suggested_image is None
        assert "no candidate image cleared safety guard" in (data.reason or "").lower()
    finally:
        db.close()


# -----------------------------------------------------------------------------
# PROBE 5: Top-1 precision on labeled eval set
# -----------------------------------------------------------------------------
def test_probe_5_evaluation_top1_precision():
    precision = run_evaluation()
    assert precision >= 90.0, f"Top-1 Precision must be at least 90%, got {precision}%"


# -----------------------------------------------------------------------------
# PROBE 6: Cost log attributes every vision/embedding call with cost
# -----------------------------------------------------------------------------
def test_probe_6_cost_tracking_per_call():
    db = SessionLocal()
    try:
        data = get_costs(db=db)
        assert data["total_calls"] > 0, "Cost log must record AI calls"
        assert data["total_cost_usd"] > 0.0, "Total cost must be calculated"
        
        entry = data["recent_logs"][0]
        assert hasattr(entry, "task_type")
        assert hasattr(entry, "model_name")
        assert hasattr(entry, "cost_usd")
        assert entry.cost_usd > 0.0
    finally:
        db.close()


if __name__ == "__main__":
    print("Setting up database seed...")
    setup_module()
    print("Running Acceptance Probes:")
    
    tests = [
        ("PROBE 1 (Batch Job & Low-Confidence Flagging)", test_probe_1_batch_processing_and_low_confidence_flagging),
        ("PROBE 2 (Red Fox Ranking over Wolf & Dog)", test_probe_2_red_fox_ranking_over_wolf_and_dog),
        ("PROBE 3 (Mismatch Guard Rejects Wolf on Fox Post)", test_probe_3_mismatch_guard_rejects_wolf_on_fox_post),
        ("PROBE 4 (No Confident Match for Out-of-Domain)", test_probe_4_no_confident_match_when_no_image_clears_bar),
        ("PROBE 5 (Top-1 Precision >= 90%)", test_probe_5_evaluation_top1_precision),
        ("PROBE 6 (Per-Call Cost Tracking)", test_probe_6_cost_tracking_per_call)
    ]
    
    passed_count = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
            passed_count += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            
    print(f"\nResult: {passed_count}/{len(tests)} Acceptance Probes Passed!")
    if passed_count == len(tests):
        print("ALL PROBES VERIFIED SUCCESSFULLY.")
        sys.exit(0)
    else:
        sys.exit(1)
