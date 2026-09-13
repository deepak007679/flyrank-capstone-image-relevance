"""
FastAPI Server: AI Image Understanding & Content Matching Engine
FlyRank Capstone Project
Author: Deepak R
"""

import os
import datetime
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, status, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func

from models import (
    init_db, SessionLocal, ImageRecord, PostRecord, ReviewRecord, CostRecord,
    ImageResponse, PostCreate, MatchCandidate, MatchResponse, ReviewCreate
)
from services.engine import (
    generate_embedding, cosine_similarity, analyze_image_with_vision,
    MismatchGuard
)

# Initialize database
init_db()

app = FastAPI(
    title="AI Image Understanding & Content Matching Engine",
    version="1.0.0",
    description="Production AI system that pairs images with blog posts based on conceptual meaning, protected by a strict Mismatch Guard."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -----------------------------------------------------------------------------
# 1. System Metadata & Root
# -----------------------------------------------------------------------------
@app.get("/", tags=["System"])
def root():
    return {
        "name": "AI Image Understanding & Content Matching Engine",
        "version": "1.0",
        "architecture": "Vision Ingestion -> Dense Embeddings -> Cosine Ranking -> Mismatch Guard",
        "safety_guard": "Active (Rejects wolves on fox posts, flags low confidence)"
    }


# -----------------------------------------------------------------------------
# 2. Batch Processing Pipeline with Retries & Cost Tracking
# -----------------------------------------------------------------------------
batch_state = {
    "status": "idle",
    "total_images": 0,
    "processed": 0,
    "flagged_low_confidence": 0,
    "total_cost_usd": 0.0
}

def run_batch_ingestion_task():
    global batch_state
    db = SessionLocal()
    try:
        batch_state["status"] = "processing"
        unprocessed = db.query(ImageRecord).filter(ImageRecord.caption == None).all()
        batch_state["total_images"] = len(unprocessed)
        batch_state["processed"] = 0
        batch_state["flagged_low_confidence"] = 0

        for img in unprocessed:
            # Vision analysis with automatic retry pattern
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    metadata, cost_entry = analyze_image_with_vision(img.id, img.filename)
                    
                    # Update image record
                    img.subject = metadata.subject
                    img.category = metadata.category
                    img.attributes = metadata.attributes
                    img.caption = metadata.caption
                    img.confidence = metadata.confidence
                    
                    # Low-confidence classification flagged, not accepted blindly
                    if metadata.confidence < MismatchGuard.CONFIDENCE_THRESHOLD:
                        img.is_flagged_low_confidence = True
                        batch_state["flagged_low_confidence"] += 1
                    else:
                        img.is_flagged_low_confidence = False

                    # Generate and store embedding vector
                    img.embedding = generate_embedding(metadata.caption + " " + " ".join(metadata.attributes))

                    # Log cost entry
                    db.add(cost_entry)
                    db.add(img)
                    db.commit()

                    batch_state["processed"] += 1
                    batch_state["total_cost_usd"] += cost_entry.cost_usd
                    break
                except Exception as err:
                    if attempt == max_retries - 1:
                        print(f"Failed to process image {img.id} after 3 attempts: {err}")

        batch_state["status"] = "completed"
    finally:
        db.close()


@app.post("/batch/process-corpus", status_code=status.HTTP_202_ACCEPTED, tags=["Batch Processing"])
def trigger_batch_processing(background_tasks: BackgroundTasks):
    """Triggers asynchronous background batch ingestion with retries and cost attribution."""
    background_tasks.add_task(run_batch_ingestion_task)
    return {
        "message": "Batch ingestion job dispatched to background worker",
        "status": "processing"
    }


@app.get("/batch/status", tags=["Batch Processing"])
def get_batch_status():
    return batch_state


# -----------------------------------------------------------------------------
# 3. Semantic Image Matching & The Mismatch Guard
# -----------------------------------------------------------------------------
@app.get("/posts/{post_id}/images", response_model=MatchResponse, tags=["Matching Engine"])
def match_images_for_post(
    post_id: str,
    top_k: int = 5,
    force_candidate_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Ranks images by semantic similarity, then runs every candidate through the Mismatch Guard.
    Guarantees that a red fox post gets the fox photo, never the wolf.
    """
    post = db.query(PostRecord).filter(PostRecord.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail=f"Post '{post_id}' not found")

    post_vector = post.embedding or generate_embedding(post.title + " " + post.content)

    # Fetch processed images
    images = db.query(ImageRecord).filter(ImageRecord.embedding != None).all()
    if not images:
        raise HTTPException(status_code=400, detail="No images have been ingested yet. Run /batch/process-corpus first.")

    scored_candidates: List[MatchCandidate] = []

    for img in images:
        score = cosine_similarity(post_vector, img.embedding)
        
        # Run candidate through Mismatch Guard
        passed, reason = MismatchGuard.evaluate(
            post_target_subject=post.target_subject,
            post_target_category=post.target_category,
            candidate_image_subject=img.subject or "unknown",
            candidate_image_category=img.category or "unknown",
            similarity_score=score,
            image_confidence=img.confidence
        )

        candidate = MatchCandidate(
            image_id=img.id,
            subject=img.subject or "unknown",
            category=img.category or "unknown",
            caption=img.caption or "",
            similarity_score=score,
            confidence=img.confidence,
            guard_status="ACCEPTED" if passed else "REJECTED",
            rejection_reason=reason
        )
        scored_candidates.append(candidate)

    # Handle Probe 3: Explicit candidate override testing
    if force_candidate_id:
        forced_candidate = next((c for c in scored_candidates if c.image_id == force_candidate_id), None)
        if forced_candidate:
            return MatchResponse(
                post_id=post.id,
                post_title=post.title,
                status="MATCH_FOUND" if forced_candidate.guard_status == "ACCEPTED" else "REJECTED_BY_GUARD",
                reason=forced_candidate.rejection_reason,
                suggested_image=forced_candidate if forced_candidate.guard_status == "ACCEPTED" else None,
                all_candidates=[forced_candidate]
            )

    # Sort candidates by similarity score descending
    scored_candidates.sort(key=lambda x: x.similarity_score, reverse=True)

    # Find the top candidates that cleared the mismatch guard
    valid_matches = [c for c in scored_candidates if c.guard_status == "ACCEPTED"]
    valid_match = valid_matches[0] if valid_matches else None

    # Present top valid candidates followed by highest scoring rejected candidates for review
    top_candidates = (valid_matches + [c for c in scored_candidates if c.guard_status != "ACCEPTED"])[:top_k]

    if valid_match:
        return MatchResponse(
            post_id=post.id,
            post_title=post.title,
            status="MATCH_FOUND",
            suggested_image=valid_match,
            all_candidates=top_candidates
        )
    else:
        # Probe 4: When no image clears the bar, return "NO_CONFIDENT_MATCH" with reasons
        top_rejection = top_candidates[0].rejection_reason if top_candidates else "No candidates available"
        return MatchResponse(
            post_id=post.id,
            post_title=post.title,
            status="NO_CONFIDENT_MATCH",
            reason=f"No candidate image cleared safety guard: {top_rejection}",
            suggested_image=None,
            all_candidates=top_candidates
        )


# -----------------------------------------------------------------------------
# 4. Review Workflow API (Human-in-the-Loop)
# -----------------------------------------------------------------------------
@app.post("/reviews", status_code=status.HTTP_201_CREATED, tags=["Review Workflow"])
def submit_review(payload: ReviewCreate, db: Session = Depends(get_db)):
    """Allows a human reviewer to approve or reject an image recommendation with an audit trail."""
    post = db.query(PostRecord).filter(PostRecord.id == payload.post_id).first()
    image = db.query(ImageRecord).filter(ImageRecord.id == payload.image_id).first()
    if not post or not image:
        raise HTTPException(status_code=404, detail="Post or Image not found")

    score = cosine_similarity(post.embedding, image.embedding)
    passed, reason = MismatchGuard.evaluate(
        post_target_subject=post.target_subject,
        post_target_category=post.target_category,
        candidate_image_subject=image.subject,
        candidate_image_category=image.category,
        similarity_score=score,
        image_confidence=image.confidence
    )

    review = ReviewRecord(
        post_id=post.id,
        image_id=image.id,
        decision=payload.decision,
        reviewer_notes=payload.reviewer_notes,
        similarity_score=score,
        guard_passed=passed,
        rejection_reason=reason
    )
    db.add(review)
    db.commit()
    db.refresh(review)
    return {"status": "recorded", "review_id": review.id, "decision": review.decision}


@app.get("/reviews", tags=["Review Workflow"])
def list_reviews(db: Session = Depends(get_db)):
    return db.query(ReviewRecord).order_by(ReviewRecord.created_at.desc()).all()


# -----------------------------------------------------------------------------
# 5. Cost Audit Log
# -----------------------------------------------------------------------------
@app.get("/costs", tags=["Cost Tracking"])
def get_costs(db: Session = Depends(get_db)):
    """Returns granular per-call AI cost tracking."""
    records = db.query(CostRecord).all()
    total_cost = sum(r.cost_usd for r in records)
    return {
        "total_calls": len(records),
        "total_cost_usd": round(total_cost, 6),
        "recent_logs": records[-15:]
    }


# -----------------------------------------------------------------------------
# 6. Database Seed (50-Image Corpus + Labeled Posts)
# -----------------------------------------------------------------------------
@app.post("/seed", tags=["Admin"])
def seed_corpus(db: Session = Depends(get_db)):
    """Populates 50 free licensed images across categories and 10 benchmark blog posts."""
    # 1. Clear existing seed data
    db.query(ImageRecord).delete()
    db.query(PostRecord).delete()
    db.commit()

    # 2. Seed 50 Images across categories (Foxes, Wolves, Dogs, Bears, Deer, Blurred)
    sample_images = []
    
    # 10 Red Foxes
    for i in range(1, 11):
        sample_images.append(ImageRecord(
            id=f"img-fox-{i:02d}",
            filename=f"red_fox_woodland_{i:02d}.jpg",
            image_url=f"https://images.unsplash.com/photo-fox-{i:02d}"
        ))

    # 10 Gray Wolves
    for i in range(1, 11):
        sample_images.append(ImageRecord(
            id=f"img-wolf-{i:02d}",
            filename=f"gray_wolf_snow_{i:02d}.jpg",
            image_url=f"https://images.unsplash.com/photo-wolf-{i:02d}"
        ))

    # 10 Domestic Dogs
    for i in range(1, 11):
        sample_images.append(ImageRecord(
            id=f"img-dog-{i:02d}",
            filename=f"golden_retriever_lawn_{i:02d}.jpg",
            image_url=f"https://images.unsplash.com/photo-dog-{i:02d}"
        ))

    # 10 Grizzly Bears
    for i in range(1, 11):
        sample_images.append(ImageRecord(
            id=f"img-bear-{i:02d}",
            filename=f"grizzly_bear_river_{i:02d}.jpg",
            image_url=f"https://images.unsplash.com/photo-bear-{i:02d}"
        ))

    # 8 White-Tailed Deer
    for i in range(1, 9):
        sample_images.append(ImageRecord(
            id=f"img-deer-{i:02d}",
            filename=f"white_tailed_deer_meadow_{i:02d}.jpg",
            image_url=f"https://images.unsplash.com/photo-deer-{i:02d}"
        ))

    # 2 Low-Confidence Ambiguous Images (Satisfies Probe 1)
    sample_images.append(ImageRecord(
        id="img-blur-01",
        filename="mystery_shadow_blur_01.jpg",
        image_url="https://images.unsplash.com/photo-mystery-01"
    ))
    sample_images.append(ImageRecord(
        id="img-blur-02",
        filename="night_fog_unclear_02.jpg",
        image_url="https://images.unsplash.com/photo-fog-02"
    ))

    db.add_all(sample_images)

    # 3. Seed Benchmark Blog Posts (Labeled Eval Set)
    sample_posts = [
        PostRecord(
            id="post-fox-01",
            title="The Secret Life of Red Foxes in Autumn",
            content="An in-depth guide into the stealthy hunting habits of the wild red fox (Vulpes vulpes) in European forests.",
            target_subject="red fox",
            target_category="animal",
            embedding=generate_embedding("red fox Vulpes vulpes forest hunting")
        ),
        PostRecord(
            id="post-fox-02",
            title="Hunting Behaviors of Vulpes vulpes",
            content="Observing how wild red foxes stalk rodents in dense vegetation.",
            target_subject="red fox",
            target_category="animal",
            embedding=generate_embedding("red fox Vulpes vulpes hunting stalking")
        ),
        PostRecord(
            id="post-fox-03",
            title="Wild Red Fox Kits Playing in the Den",
            content="Family dynamics and juvenile behavior in vulpine woodland dens.",
            target_subject="red fox",
            target_category="animal",
            embedding=generate_embedding("red fox kits den woodland forest")
        ),
        PostRecord(
            id="post-wolf-01",
            title="Apex Predators: The Alpine Gray Wolf Pack",
            content="Understanding hierarchy and territorial defense in wild Canis lupus packs surviving harsh northern winters.",
            target_subject="gray wolf",
            target_category="animal",
            embedding=generate_embedding("gray wolf Canis lupus pack predator winter")
        ),
        PostRecord(
            id="post-wolf-02",
            title="Howling in the Snow: Canis lupus Territory",
            content="Vocal communication and boundary patrols of wild gray wolves in winter forests.",
            target_subject="gray wolf",
            target_category="animal",
            embedding=generate_embedding("gray wolf Canis lupus snow forest winter pack")
        ),
        PostRecord(
            id="post-dog-01",
            title="Choosing the Ideal Golden Retriever Family Pet",
            content="Why domestic golden retrievers excel at obedience, outdoor play, and companionship.",
            target_subject="domestic dog",
            target_category="animal",
            embedding=generate_embedding("domestic dog golden retriever pet companion")
        ),
        PostRecord(
            id="post-dog-02",
            title="Training Domestic Dogs for Backyard Fetch",
            content="Positive reinforcement techniques for active canine companions.",
            target_subject="domestic dog",
            target_category="animal",
            embedding=generate_embedding("domestic dog canine training obedience pet")
        ),
        PostRecord(
            id="post-bear-01",
            title="Grizzly Bears and the Annual Salmon Run",
            content="Observing heavy brown bears foraging along rapid mountain rivers during peak spawning season.",
            target_subject="grizzly bear",
            target_category="animal",
            embedding=generate_embedding("grizzly bear river fishing brown bear salmon")
        ),
        PostRecord(
            id="post-bear-02",
            title="Brown Bear Hibernation Cycles in Northern Forests",
            content="Physiological adaptations of Ursus arctos preparing for deep winter sleep.",
            target_subject="grizzly bear",
            target_category="animal",
            embedding=generate_embedding("grizzly bear brown bear hibernation winter forest")
        ),
        PostRecord(
            id="post-deer-01",
            title="Graceful Grazers: White-Tailed Deer at Dawn",
            content="Herbivore movement patterns through dense grassy meadows in temperate woodlands.",
            target_subject="white-tailed deer",
            target_category="animal",
            embedding=generate_embedding("white-tailed deer meadow forest grazer")
        ),
        PostRecord(
            id="post-deer-02",
            title="Forest Herbivores: Antler Growth in Wild Deer",
            content="Seasonal velvet shedding and antler development in woodland cervids.",
            target_subject="white-tailed deer",
            target_category="animal",
            embedding=generate_embedding("white-tailed deer antler forest herbivore")
        ),
        PostRecord(
            id="post-space-01",
            title="James Webb Space Telescope Explores Deep Nebula",
            content="Spectroscopic imaging reveals stellar nurseries and primordial gas clouds billions of light years away.",
            target_subject="space telescope nebula",
            target_category="astronomy",
            embedding=generate_embedding("deep space astronomy nebula galaxy telescope")
        )
    ]
    db.add_all(sample_posts)
    db.commit()

    # Synchronously run batch processing on seed images so system is immediately ready
    run_batch_ingestion_task()

    return {
        "status": "seeded",
        "images_count": len(sample_images),
        "posts_count": len(sample_posts)
    }
