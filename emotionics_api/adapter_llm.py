# adapter_llm.py
import json
from typing import Any, Dict

from openai import OpenAI

# adapter_llm.py (APIキーない時のテスト用)

from .core import Features

def extract_features_with_llm(text: str) -> Features:
    """
    テスト用：OpenAI APIを呼ばずに固定の特徴量を返す。
    Emotionics Engine v1 の Core 部分が動くことを確認する目的。
    """

    # 超シンプルな判定：ネガ/ポジなどでダミー返す
    if "ごめんなさい" in text or "すみません" in text:
        candidate = (("Shame", 0.6), ("Confused", 0.4))
        politeness = 0.8
        honesty_cues = 0.7
        sarcasm = 0.0
        directness = 0.4
        intensity = 0.6
    else:
        candidate = (("Joy", 0.7), ("Neutral", 0.3))
        politeness = 0.5
        honesty_cues = 0.5
        sarcasm = 0.0
        directness = 0.5
        intensity = 0.7

    return Features(
        candidate_emotions=candidate,
        intensity=intensity,
        politeness=politeness,
        sarcasm=sarcasm,
        directness=directness,
        honesty_cues=honesty_cues,
    )


# from .core import Features

# client = OpenAI()  # OPENAI_API_KEY 環境変数前提


# FEATURES_SYSTEM_PROMPT = """You are an Emotionics feature extractor.
# Given a short text, you must respond with a strict JSON object having this schema:

# {
#   "candidate_emotions": [
#     { "label": "Confused", "score": 0.55 },
#     { "label": "Shame",    "score": 0.45 }
#   ],
#   "intensity": 0.7,
#   "politeness": 0.8,
#   "sarcasm": 0.1,
#   "directness": 0.6,
#   "honesty_cues": 0.7
# }

# Do not include any extra fields or commentary.
# Emotions must be high-level labels like Joy, Anger, Fear, Confused, Shame, etc.
# All scores must be between 0.0 and 1.0.
# """


# def extract_features_with_llm(text: str) -> Features:
#     resp = client.chat.completions.create(
#         model="gpt-5.1-mini",
#         messages=[
#             {"role": "system", "content": FEATURES_SYSTEM_PROMPT},
#             {"role": "user", "content": text},
#         ],
#         temperature=0.0,
#     )

#     content = resp.choices[0].message.content
#     data: Dict[str, Any] = json.loads(content)

#     cand = tuple(
#         (item["label"], float(item["score"]))
#         for item in data.get("candidate_emotions", [])
#     )

#     return Features(
#         candidate_emotions=cand,
#         intensity=float(data.get("intensity", 0.5)),
#         politeness=float(data.get("politeness", 0.5)),
#         sarcasm=float(data.get("sarcasm", 0.0)),
#         directness=float(data.get("directness", 0.5)),
#         honesty_cues=float(data.get("honesty_cues", 0.5)),
#     )

