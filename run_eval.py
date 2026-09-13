"""
Evaluation Benchmark Runner: Top-1 Precision on Labeled Ground Truth
FlyRank Capstone: AI Image Understanding & Content Matching Engine
Author: Deepak R
"""

import json
import os
from main import seed_corpus, match_images_for_post, SessionLocal

def run_evaluation():
    db = SessionLocal()
    try:
        # 1. Seed and index the corpus
        seed_corpus(db)

        # 2. Load labeled ground truth evaluation set
        eval_file = os.path.join(os.path.dirname(__file__), "eval_set.json")
        with open(eval_file, "r") as f:
            eval_cases = json.load(f)

        print("==================================================================")
        print("      AI IMAGE CONTENT MATCHING ENGINE - EVALUATION RUNNER        ")
        print("==================================================================")
        print(f"Loaded {len(eval_cases)} labeled evaluation test pairs.\n")

        correct_top1 = 0
        total_cases = len(eval_cases)

        for case in eval_cases:
            post_id = case["post_id"]
            expected_prefix = case["expected_image_prefix"]
            
            resp = match_images_for_post(post_id=post_id, top_k=5, db=db)
            status_result = resp.status

            if expected_prefix == "NO_MATCH":
                # Guard must refuse the match
                if status_result == "NO_CONFIDENT_MATCH":
                    correct_top1 += 1
                    result_str = "[PASS] CORRECTLY REFUSED MATCH"
                else:
                    result_str = f"[FAIL] UNWANTED MATCH PRODUCED ({resp.suggested_image.image_id if resp.suggested_image else 'None'})"
            else:
                suggested = resp.suggested_image
                if suggested and suggested.image_id.startswith(expected_prefix):
                    correct_top1 += 1
                    result_str = f"[PASS] MATCHED: {suggested.image_id} (Sim: {suggested.similarity_score:.2f})"
                else:
                    actual = suggested.image_id if suggested else "NONE (Refused)"
                    result_str = f"[FAIL] EXPECTED: {expected_prefix}*, GOT: {actual}"

            print(f"Post: {case['title'][:40]:<40} -> {result_str}")

        top1_precision = (correct_top1 / total_cases) * 100.0
        print("------------------------------------------------------------------")
        print(f"TOTAL CASES: {total_cases}")
        print(f"CORRECT TOP-1 SUGGESTIONS: {correct_top1}")
        print(f"TOP-1 PRECISION: {top1_precision:.1f}%")
        print("==================================================================")
        
        return top1_precision
    finally:
        db.close()

if __name__ == "__main__":
    run_evaluation()
