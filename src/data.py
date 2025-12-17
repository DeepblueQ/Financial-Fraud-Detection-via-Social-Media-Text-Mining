import pandas as pd
import numpy as np

def load_dataset(path: str, sep: str, text_col: str, label_col: str, positive_if_not_zero: bool = True):
    df = pd.read_csv(path, sep=sep)

    # 列名校验
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"数据列名不匹配。需要列: {text_col}, {label_col}，实际列: {list(df.columns)}。"
            f"如果分隔符不对，请检查 sep（当前 sep={repr(sep)}）"
        )
    
    texts = df[text_col].astype(str).fillna("").tolist()
    raw = df[label_col].values

    if positive_if_not_zero:
        y = (raw != 0).astype(int)
    else:

        y = raw.astype(int)
    
    print("标签分布：", pd.Series(y).value_counts())


    uniq = set(y.tolist())
    if len(uniq) < 2:
        raise ValueError(f"标签只有一类：{uniq}，无法训练二分类器。")

    return texts, y
