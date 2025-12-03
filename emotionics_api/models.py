# models.py
from typing import Dict, Optional
from pydantic import BaseModel, Field


class AnalyzeOptions(BaseModel):
    top_k: int = Field(default=5, ge=1, le=64)
    return_trace: bool = False


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: str = Field(default="auto", description="BCP-47 code like 'ja', 'en'")
    elements_version: str = Field(default="3.0")
    options: AnalyzeOptions = Field(default_factory=AnalyzeOptions)


class ModeScores(BaseModel):
    feel: float = Field(ge=0.0, le=1.0)
    feign: float = Field(ge=0.0, le=1.0)


class MetaInfo(BaseModel):
    overall_confidence: float = Field(ge=0.0, le=1.0)
    tokens: Optional[int] = Field(default=None, ge=0)
    language: Optional[str] = None
    request_id: Optional[str] = None


class AnalyzeResponse(BaseModel):
    elements_version: str
    expressed_distribution: Dict[str, float]
    feel_distribution: Dict[str, float]
    feign_distribution: Dict[str, float]
    mode_scores: ModeScores
    meta: MetaInfo
    trace: Optional[dict] = None