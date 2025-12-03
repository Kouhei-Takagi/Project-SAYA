# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uuid

from .models import AnalyzeRequest, AnalyzeResponse, MetaInfo, ModeScores
from .core import (
    ElementTable,
    map_features_to_elements,
    estimate_mode_scores,
    build_emotion_vectors,
)
from .adapter_llm import extract_features_with_llm

app = FastAPI(title="Emotionics API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 必要なら制限
    allow_methods=["*"],
    allow_headers=["*"],
)

# ここで Emotionics Periodic Table 3.0 の JSON を読み込んで ElementTable を作る
import json
from pathlib import Path

TABLE_JSON_PATH = Path("EmotionalPeriodicTable3.0/periodic-table3.json")
table_data = json.loads(TABLE_JSON_PATH.read_text(encoding="utf-8"))

# alias -> canonical_name マップを構築
id_by_alias: dict[str, str] = {}

for emo in table_data["emotions"]:
    canonical = emo["name"]  # 例: "Confused"

    # 1. 正式名称
    id_by_alias[canonical] = canonical

    # 2. シンボル（J, Sh, Co など）も alias にしておく
    symbol = emo.get("symbol")
    if symbol:
        id_by_alias[symbol] = canonical

    # 3. シノニムも全部 alias として張る
    relations = emo.get("relations", {})
    for syn in relations.get("synonyms", []):
        # 大文字小文字のブレを吸収したければここで lower() してもOK
        id_by_alias[syn] = canonical

element_table = ElementTable(id_by_alias=id_by_alias)


@app.post("/v1/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    # 1. LLMで特徴抽出
    features = extract_features_with_llm(request.text)

    # 2. Feature -> expressed_distribution
    expressed = map_features_to_elements(features, element_table)

    # 3. Feel / Feign 比率推定
    mode_scores_core = estimate_mode_scores(features)

    # 4. Feel/Feign ベクトルを生成
    feel_vec, feign_vec = build_emotion_vectors(expressed, mode_scores_core)

    # 5. 必要なら top_k で切る
    def top_k(d: dict[str, float], k: int) -> dict[str, float]:
        items = sorted(d.items(), key=lambda kv: kv[1], reverse=True)
        return {k_: v for k_, v in items[:k]}

    k = request.options.top_k
    expressed_k = top_k(expressed, k)
    feel_k = top_k(feel_vec, k)
    feign_k = top_k(feign_vec, k)

    # 6. メタ情報
    request_id = str(uuid.uuid4())
    meta = MetaInfo(
        overall_confidence=0.73,  # v1では仮置き or ルールベースで計算
        tokens=None,              # OpenAIレスポンスから取れれば入れる
        language=request.language if request.language != "auto" else None,
        request_id=request_id,
    )

    trace = None
    if request.options.return_trace:
        trace = {
            "features": {
                "candidate_emotions": features.candidate_emotions,
                "intensity": features.intensity,
                "politeness": features.politeness,
                "sarcasm": features.sarcasm,
                "directness": features.directness,
                "honesty_cues": features.honesty_cues,
            }
        }

    return AnalyzeResponse(
        elements_version=request.elements_version,
        expressed_distribution=expressed_k,
        feel_distribution=feel_k,
        feign_distribution=feign_k,
        mode_scores=ModeScores(
            feel=mode_scores_core.feel,
            feign=mode_scores_core.feign,
        ),
        meta=meta,
        trace=trace,
    )