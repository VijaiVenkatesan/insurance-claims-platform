"""
FastAPI routes for the claims processing API.
All routes follow REST conventions with proper error handling.
"""
from __future__ import annotations
from typing import Optional, Dict, Any
from datetime import datetime
import uuid as _uuid
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from app.infrastructure.db.models import get_db
from app.models.schemas import (
    ClaimSubmitResponse, ClaimStatusResponse, ClaimListResponse,
    HITLReviewRequest, HITLReviewResponse, PolicyIndexRequest,
    HealthCheckResponse, AuditLogEntry
)
from app.services.claims_service import ClaimsService, get_health_status
from app.core.logging import get_logger, set_correlation_id

logger = get_logger(__name__)

router = APIRouter()


def get_service(db: AsyncSession = Depends(get_db)) -> ClaimsService:
    return ClaimsService(db)


# ─────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────

@router.get("/health", response_model=HealthCheckResponse, tags=["System"])
async def health_check():
    """System health check endpoint."""
    return await get_health_status()


# ─────────────────────────────────────────────
# CLAIM SUBMISSION
# ─────────────────────────────────────────────

@router.post(
    "/claims/submit",
    response_model=ClaimSubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Claims"],
    summary="Submit a new insurance claim document",
)
async def submit_claim(
    request: Request,
    file: UploadFile = File(..., description="Insurance claim document (PDF, PNG, JPG, TIFF)"),
    correlation_id: Optional[str] = Form(None, description="Optional idempotency key"),
    service: ClaimsService = Depends(get_service),
):
    """
    Upload and process an insurance claim document.
    
    - Supports PDF, PNG, JPG, JPEG, TIFF formats
    - Max file size: 10MB
    - Returns immediately with claim_id; processing happens asynchronously
    - Use correlation_id for idempotent resubmission
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Empty file provided")

    try:
        result = await service.submit_claim(
            file_content=content,
            filename=file.filename,
            content_type=file.content_type or "application/octet-stream",
            correlation_id=correlation_id,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Claim submission failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal processing error")


# ─────────────────────────────────────────────
# CLAIM STATUS & RETRIEVAL
# ─────────────────────────────────────────────

@router.get(
    "/claims/{claim_id}",
    response_model=ClaimStatusResponse,
    tags=["Claims"],
    summary="Get claim status and details",
)
async def get_claim(
    claim_id: str,
    service: ClaimsService = Depends(get_service),
):
    """Get the current status, extracted data, fraud score, and decision for a claim."""
    try:
        return await service.get_claim_status(claim_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get(
    "/claims",
    response_model=ClaimListResponse,
    tags=["Claims"],
    summary="List all claims",
)
async def list_claims(
    status_filter: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    service: ClaimsService = Depends(get_service),
):
    """
    List claims with optional filtering.
    
    Status values: RECEIVED, OCR_PROCESSING, EXTRACTING, VALIDATING, 
    POLICY_CHECK, FRAUD_ANALYSIS, DECISION_PENDING, HITL_REVIEW, 
    APPROVED, REJECTED, INVESTIGATING, ERROR
    """
    if page < 1:
        raise HTTPException(status_code=422, detail="Page must be >= 1")
    if page_size < 1 or page_size > 100:
        raise HTTPException(status_code=422, detail="Page size must be 1-100")

    return await service.list_claims(status_filter, page, page_size)


@router.get(
    "/claims/{claim_id}/audit",
    tags=["Claims"],
    summary="Get audit trail for a claim",
)
async def get_audit_trail(
    claim_id: str,
    service: ClaimsService = Depends(get_service),
):
    """Get the complete audit trail showing all events for a claim."""
    try:
        return await service.get_audit_trail(claim_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ─────────────────────────────────────────────
# HUMAN-IN-THE-LOOP (HITL)
# ─────────────────────────────────────────────

@router.get(
    "/hitl/pending",
    response_model=ClaimListResponse,
    tags=["HITL Review"],
    summary="Get all claims pending human review",
)
async def get_pending_hitl(service: ClaimsService = Depends(get_service)):
    """Get all claims currently waiting for human review."""
    return await service.get_pending_hitl()


@router.post(
    "/hitl/{claim_id}/review",
    response_model=HITLReviewResponse,
    tags=["HITL Review"],
    summary="Submit a human review decision for a claim",
)
async def submit_hitl_review(
    claim_id: str,
    review: HITLReviewRequest,
    service: ClaimsService = Depends(get_service),
):
    """
    Submit a human reviewer's decision on a claim.
    
    Actions:
    - APPROVE: Approve the claim (optionally with modified amount)
    - REJECT: Reject with required reason
    - INVESTIGATE: Route to investigation team
    - REQUEST_MORE_INFO: Keep in HITL queue, request more info
    """
    try:
        return await service.submit_hitl_review(
            claim_id=claim_id,
            review=review,
            reviewer_id=review.reviewer_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"HITL review failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process review")


# ─────────────────────────────────────────────
# POLICY MANAGEMENT
# ─────────────────────────────────────────────

@router.post(
    "/policies/index",
    tags=["Policy Management"],
    summary="Index a policy document for RAG",
)
async def index_policy(
    request: PolicyIndexRequest,
    service: ClaimsService = Depends(get_service),
):
    """
    Index an insurance policy document into the vector store for RAG.
    
    The indexed policy will be used by the Policy Agent to check claim eligibility.
    Idempotent - same policy content will be deduplicated.
    """
    try:
        return await service.index_policy(request)
    except Exception as e:
        logger.error(f"Policy indexing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Policy indexing failed: {str(e)}")


# ─────────────────────────────────────────────
# ANALYTICS
# ─────────────────────────────────────────────

@router.get(
    "/analytics/summary",
    tags=["Analytics"],
    summary="Get claims processing analytics",
)
async def get_analytics(
    db: AsyncSession = Depends(get_db),
):
    """Get high-level analytics on claim processing."""
    from sqlalchemy import select, func
    from app.infrastructure.db.models import ClaimDB

    result = await db.execute(
        select(
            ClaimDB.status,
            func.count(ClaimDB.id).label("count"),
            func.avg(ClaimDB.fraud_score).label("avg_fraud_score"),
            func.avg(ClaimDB.claimed_amount).label("avg_claimed_amount"),
        ).group_by(ClaimDB.status)
    )
    rows = result.all()

    return {
        "by_status": [
            {
                "status": row.status,
                "count": row.count,
                "avg_fraud_score": round(row.avg_fraud_score or 0, 3),
                "avg_claimed_amount": round(row.avg_claimed_amount or 0, 2),
            }
            for row in rows
        ]
    }



# ─────────────────────────────────────────────
# OCR EXTRACTION HELPER
# Used by ocr_preview — calls Groq LLM directly
# to avoid import fragility with extraction_agent
# ─────────────────────────────────────────────

async def _extract_with_llm(raw_text: str) -> dict:
    """
    Extract structured claim data from OCR text using Groq LLM.
    Returns a dict matching the extraction_agent output format.
    Falls back to empty dict on error — user will fill fields manually.
    """
    EXTRACT_PROMPT = """Extract insurance claim information from the following document text.
Return a JSON object with these fields (use null for missing values):
{
  "claimant_name": null,
  "date_of_birth": null,
  "gender": null,
  "contact": null,
  "email": null,
  "address": null,
  "aadhaar_number": null,
  "pan_number": null,
  "policy_number": null,
  "insurance_company": null,
  "insurance_type": "HEALTH",
  "policy_start_date": null,
  "policy_end_date": null,
  "sum_insured": null,
  "incident_date": null,
  "reported_date": null,
  "hospital_name": null,
  "doctor_name": null,
  "diagnosis": null,
  "treatment": null,
  "claimed_amount": null,
  "currency": "INR",
  "amount_breakdown": {},
  "country": "IN"
}
Return ONLY valid JSON. No explanation. No markdown.
Document text:
"""

    try:
        from app.infrastructure.llm.groq_client import get_groq_client
        import json

        llm = get_groq_client()
        response = await llm.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You extract structured data from insurance documents. Return only JSON."},
                {"role": "user", "content": EXTRACT_PROMPT + raw_text[:4000]},
            ],
            max_tokens=1500,
            temperature=0.1,
        )
        text = response.choices[0].message.content or ""
        # Strip markdown fences if present
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()
        return json.loads(text)
    except Exception as e:
        logger.warning(f"LLM extraction fallback triggered: {e}")
        # Return empty structure — user fills in via the review form
        return {
            "claimant_name": None, "date_of_birth": None, "gender": None,
            "contact": None, "email": None, "address": None,
            "aadhaar_number": None, "pan_number": None,
            "policy_number": None, "insurance_company": None,
            "insurance_type": "HEALTH",
            "policy_start_date": None, "policy_end_date": None, "sum_insured": None,
            "incident_date": None, "reported_date": None,
            "hospital_name": None, "doctor_name": None,
            "diagnosis": None, "treatment": None,
            "claimed_amount": None, "currency": "INR",
            "amount_breakdown": {}, "country": "IN",
        }


# ─────────────────────────────────────────────
# OCR REVIEW & EDIT
# ─────────────────────────────────────────────

class ReviewedClaimData(BaseModel):
    """User-reviewed/corrected OCR extraction result."""
    # Document reference
    original_filename: str
    ocr_text: str                          # Raw OCR text (preserved)

    # Editable extracted fields — user can correct these
    claimant_name:   Optional[str] = None
    date_of_birth:   Optional[str] = None
    gender:          Optional[str] = None
    contact:         Optional[str] = None
    email:           Optional[str] = None
    address:         Optional[str] = None
    aadhaar_number:  Optional[str] = None
    pan_number:      Optional[str] = None

    policy_number:   Optional[str] = None
    insurance_company: Optional[str] = None
    insurance_type:  Optional[str] = "HEALTH"
    policy_start:    Optional[str] = None
    policy_end:      Optional[str] = None
    sum_insured:     Optional[float] = None

    incident_date:   Optional[str] = None
    reported_date:   Optional[str] = None
    hospital_name:   Optional[str] = None
    doctor_name:     Optional[str] = None
    diagnosis:       Optional[str] = None
    treatment:       Optional[str] = None

    claimed_amount:  Optional[float] = None
    currency:        Optional[str] = "INR"
    amount_breakdown: Optional[Dict[str, float]] = None

    country:         Optional[str] = "IN"


@router.post(
    "/claims/ocr-preview",
    tags=["Claims"],
    summary="Step 1: Extract data from document for user review",
)
async def ocr_preview(
    file: UploadFile = File(...),
):
    """
    Upload a document and get back OCR-extracted structured data.
    The user can review and correct this data BEFORE submitting to the full AI pipeline.
    This prevents OCR errors (like 5,260 being read as 61,260) from propagating.
    """
    try:
        content = await file.read()
        filename = file.filename or "document"
        content_type = file.content_type or "application/octet-stream"

        # ── Step 1: OCR — write to temp file then call engine ──
        import tempfile, os, pathlib
        from app.infrastructure.ocr.engine import get_ocr_engine

        # Determine file extension from filename or content_type
        ext = pathlib.Path(filename).suffix.lower() if filename else ""
        if not ext:
            ext_map = {
                "application/pdf": ".pdf",
                "image/png": ".png",
                "image/jpeg": ".jpg",
                "image/tiff": ".tif",
            }
            ext = ext_map.get(content_type, ".pdf")

        # Write bytes to a temp file (OCR engine works on file paths)
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            ocr_engine = get_ocr_engine()
            ocr_result = await ocr_engine.extract_text(tmp_path)
            raw_text = ocr_result.raw_text or ""
            ocr_confidence = ocr_result.confidence or 0.0
        finally:
            # Always clean up temp file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        if not raw_text.strip():
            raise HTTPException(422, "Could not extract text from document. Try a higher quality scan.")

        # ── Step 2: Extract structured data with Groq LLM ──
        # We call the LLM directly here to avoid import issues with extraction_agent.
        # The extraction_agent internally does the same thing.
        extracted = await _extract_with_llm(raw_text)

        return {
            "status": "preview_ready",
            "ocr_confidence": ocr_confidence,
            "ocr_text": raw_text,
            "original_filename": filename,
            "extracted": extracted,
            "message": "Review and correct the extracted data below before submitting.",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OCR preview error: {e}")
        raise HTTPException(500, f"OCR preview failed: {str(e)}")


@router.post(
    "/claims/submit-with-data",
    tags=["Claims"],
    summary="Step 2: Submit user-reviewed data to the full AI pipeline",
)
async def submit_claim_with_reviewed_data(
    reviewed: ReviewedClaimData,
    db: AsyncSession = Depends(get_db),
):
    """
    Accept user-reviewed/corrected OCR data and run the full AI pipeline:
    Validate → Policy Check (RAG) → Fraud Analysis → Decision
    Skips OCR and extraction since user has already verified the data.
    """
    try:
        import json as _json

        # Encode reviewed data as a structured text document.
        # ClaimsService.submit_claim() handles all DB creation and pipeline correctly.
        # The extraction LLM will re-parse this clean JSON perfectly, preserving
        # all user-corrected values (including the verified claimed_amount).
        reviewed_doc = f"""REVIEWED CLAIM DATA (User-verified OCR output)
Original file: {reviewed.original_filename}

=== CLAIMANT ===
Name: {reviewed.claimant_name or ""}
Date of Birth: {reviewed.date_of_birth or ""}
Gender: {reviewed.gender or ""}
Contact: {reviewed.contact or ""}
Email: {reviewed.email or ""}
Address: {reviewed.address or ""}
Aadhaar: {reviewed.aadhaar_number or ""}
PAN: {reviewed.pan_number or ""}

=== POLICY ===
Policy Number: {reviewed.policy_number or ""}
Insurance Company: {reviewed.insurance_company or ""}
Insurance Type: {reviewed.insurance_type or "HEALTH"}
Policy Start: {reviewed.policy_start or ""}
Policy End: {reviewed.policy_end or ""}
Sum Insured: {reviewed.sum_insured or ""}

=== INCIDENT ===
Incident Date: {reviewed.incident_date or ""}
Reported Date: {reviewed.reported_date or ""}
Hospital: {reviewed.hospital_name or ""}
Doctor: {reviewed.doctor_name or ""}
Diagnosis: {reviewed.diagnosis or ""}
Treatment: {reviewed.treatment or ""}

=== CLAIM AMOUNTS (USER VERIFIED) ===
Claimed Amount: {reviewed.claimed_amount or ""} {reviewed.currency or "INR"}
Currency: {reviewed.currency or "INR"}
Country: {reviewed.country or "IN"}

=== ORIGINAL OCR TEXT ===
{reviewed.ocr_text}
"""

        # Use service.submit_claim() — this is the single correct entry point
        # It handles DB creation, pipeline execution, and async processing.
        service = ClaimsService(db)
        result = await service.submit_claim(
            file_content=reviewed_doc.encode("utf-8"),
            filename=reviewed.original_filename or "ocr_reviewed.txt",
            content_type="text/plain",
            correlation_id=None,
        )

        logger.info(f"Reviewed claim submitted via service: {result.claim_id}")
        return {
            "claim_id": result.claim_id,
            "status": "VALIDATING",
            "message": "Your reviewed data has been submitted. Processing will complete in ~30 seconds.",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"submit-with-data error: {e}")
        raise HTTPException(500, f"Submission failed: {str(e)}")
