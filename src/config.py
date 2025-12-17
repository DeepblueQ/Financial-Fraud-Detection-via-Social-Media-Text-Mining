# src/config.py
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:

    dataset_path: str = str(Path(__file__).resolve().parent.parent / "datasets" / "data.csv")
    sep: str = "\t"
    text_col: str = "Text"
    label_col: str = "Label_id"


    positive_if_not_zero: bool = True


    sbert_model: str = "shibing624/text2vec-base-chinese"


    llm_model: str = "deepseek-chat"


    feature_mode: str = "fusion"

    # 缓存路径, 便于多次测试
    llm_cache_path: str = "cache/llm_features.jsonl"
    emb_cache_path: str = "cache/embeddings.npy"

    # 可选api, in openai / qwen / deepseek
    llm_provider = "deepseek"
