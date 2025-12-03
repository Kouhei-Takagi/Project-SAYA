# core.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class Features:
    candidate_emotions: Tuple[Tuple[str, float], ...]
    intensity: float
    politeness: float
    sarcasm: float
    directness: float
    honesty_cues: float


@dataclass(frozen=True)
class ElementTable:
    """
    Emotionics元素周期表3.0のJSONを食わせて構成する想定。
    alias -> canonical_id のマップを持っておくと楽。
    """
    id_by_alias: Dict[str, str]


@dataclass(frozen=True)
class ModeScoresCore:
    feel: float
    feign: float


def _normalize_distribution(dist: Dict[str, float]) -> Dict[str, float]:
    s = sum(dist.values())
    if s <= 0:
        return {}
    return {k: v / s for k, v in dist.items() if v > 0}


# features-->expressed_distribution
def map_features_to_elements(
    features: Features,
    table: ElementTable
) -> Dict[str, float]:
    """
    features.candidate_emotions のラベルを Emotionics元素IDにマッピングし、
    正規化した expressed_distribution を返す。
    """
    raw: Dict[str, float] = {}

    for label, score in features.candidate_emotions:
        if score <= 0:
            continue
        # ラベルを正規化してから alias マップで引く
        key = label.strip()
        canonical = table.id_by_alias.get(key, key)  # 見つからない場合はそのまま
        raw[canonical] = raw.get(canonical, 0.0) + float(score)

    return _normalize_distribution(raw)

# Feel/Feign比率を推定
def estimate_mode_scores(features: Features) -> ModeScoresCore:
    """
    honesty_cues, sarcasm, politeness, directness から
    Feel / Feign の比率をざっくり推定する v1 規則ベース。
    """
    h = features.honesty_cues
    s = features.sarcasm
    p = features.politeness
    d = features.directness

    # 素朴なヒューリスティック（後で調整可）
    feel_score = 0.0
    feel_score += 0.6 * h
    feel_score += 0.2 * d
    feel_score -= 0.3 * s
    # 「高ポライトネス＋ネガティブ」は逆にFeign疑惑なので軽くマイナスに振る
    feel_score -= 0.2 * max(0.0, p - 0.7)

    # 0〜1 にクリップ
    feel_score = max(0.0, min(1.0, feel_score))
    feign_score = 1.0 - feel_score

    return ModeScoresCore(feel=feel_score, feign=feign_score)

# ベクトルを分解する
def build_emotion_vectors(
    expressed: Dict[str, float],
    mode_scores: ModeScoresCore
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    v1 では expressed_distribution と同じ元素構成で、
    重みだけ Feel/Feign 比率を掛けるシンプル設計。
    """
    feel_vec = {k: v * mode_scores.feel for k, v in expressed.items()}
    feign_vec = {k: v * mode_scores.feign for k, v in expressed.items()}

    return feel_vec, feign_vec