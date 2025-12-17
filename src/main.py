import os
import numpy as np
from tqdm import tqdm

from config import Config
from data import load_dataset
from embed import SBERTEmbedder
from features import llm_extract_scores
from llm_cache import JSONLCache
from train import train_eval_xgb

def build_features(E: np.ndarray, F: np.ndarray, mode: str) -> np.ndarray:
    base = F[:, :-1]
    risk = F[:, -1:]

    if mode == "embed_only":
        return E
    if mode == "llm_only":
        return F
    if mode == "fusion":
        return np.hstack([E, F])
    if mode == "fusion_no_risk":
        return np.hstack([E, base])
    if mode == "fusion_risk_only":
        return np.hstack([E, risk])
    raise ValueError(
        f"未知 feature_mode: {mode}（应为 embed_only/llm_only/fusion/fusion_no_risk/fusion_risk_only）"
    )

def main():
    cfg = Config()
    os.makedirs("cache", exist_ok=True)

    texts, y = load_dataset(
        cfg.dataset_path, cfg.sep, cfg.text_col, cfg.label_col, cfg.positive_if_not_zero
    )

    # 1. Embedding
    if os.path.exists(cfg.emb_cache_path):
        E = np.load(cfg.emb_cache_path)
        if len(E) != len(texts):
            raise ValueError("embeddings.npy 与数据行数不一致：请删除 cache/embeddings.npy 重新生成。")
        print(f"[OK] 读取 embedding 缓存: {cfg.emb_cache_path}, shape={E.shape}")
    else:
        embedder = SBERTEmbedder(cfg.sbert_model)
        E = embedder.encode(texts)
        np.save(cfg.emb_cache_path, E)
        print(f"[OK] 写入 embedding 缓存: {cfg.emb_cache_path}, shape={E.shape}")

    # 2. LLM features
    cache = JSONLCache(cfg.llm_cache_path)
    feats = []
    miss = 0

    for t in tqdm(texts, desc="LLM features"):
        v = cache.get(t)
        if v is None:
            miss += 1
            v = llm_extract_scores(t, cfg.llm_model)
            cache.set(t, v)
        feats.append([
            v["has_contact"], v["has_url"], v["has_money"], v["has_authority"],
            v["urgency"], v["money_lure"], v["action_push"], v["risk_overall"],
            ]
        )


    F = np.array(feats, dtype=float)
    print(f"[OK] LLM features shape={F.shape}, cache_miss={miss}/{len(texts)}")

    # 3. concat, train
    X = build_features(E, F, cfg.feature_mode)
    print(f"[OK] Feature mode={cfg.feature_mode}, X shape={X.shape}")
    _ = train_eval_xgb(X, y)

if __name__ == "__main__":
    main()
