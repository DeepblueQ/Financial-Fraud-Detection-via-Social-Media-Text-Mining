# src/llm_features.py
import json
from openai import OpenAI
from clients import build_client
from config import Config
from typing import Any

cfg = Config()
client = build_client(cfg.llm_provider)


SCHEMA = {
  "name": "fraud_feature_schema_v1",
  "strict": True,
  "schema": {
    "type": "object",
    "additionalProperties": False,
    "properties": {
      "has_contact": {"type": "integer", "enum": [0, 1]},
      "has_url": {"type": "integer", "enum": [0, 1]},
      "has_money": {"type": "integer", "enum": [0, 1]},
      "has_authority": {"type": "integer", "enum": [0, 1]},

      "urgency": {"type": "number", "minimum": 0, "maximum": 1},
      "money_lure": {"type": "number", "minimum": 0, "maximum": 1},
      "action_push": {"type": "number", "minimum": 0, "maximum": 1},
      "risk_overall": {"type": "number", "minimum": 0, "maximum": 1}
    },
    "required": [
      "has_contact", "has_url", "has_money", "has_authority",
      "urgency", "money_lure", "action_push", "risk_overall"
    ]
  }
}


def llm_extract_scores(text: str, model: str) -> dict:
    prompt = f"""
你是中文互联网文本的风控特征抽取器。你的任务不是生成解释，而是输出严格JSON特征，用于下游XGBoost分类。

请根据文本内容，抽取以下特征（所有数值必须在0~1，所有二值必须是0或1）：
1) has_contact：出现手机号/座机号/微信号/QQ号/群号/“加V/加微信/同号/联系我/私聊”等可联系线索 -> 1，否则0
2) has_url：出现网址/域名/链接形式（含www、.com、.cn、http等）-> 1，否则0
3) has_money：出现“代还/下款/额度/返利/收益/赚钱/投资/提现/贷款/刷单/彩金/充值/转账/支付”等金钱交易或金融相关 -> 1，否则0
4) has_authority：出现“银行/官方/客服/平台/公安/法院/征信/招商/京东白条/支付宝/微信支付”等权威或平台背书/冒充 -> 1，否则0

强度分数（0~1，0为无，1为极强）：
5) urgency：紧迫感（限时、马上、最后、立即处理、错过就没等）
6) money_lure：高收益/优惠诱导（高回报、返利、稳赚、免费提现、低门槛高收益等）
7) action_push：强引导用户执行动作（加微信/点击链接/下载APP/转账/充值/提供验证码/提交资料）
8) risk_overall：综合风险倾向（只基于文本，偏“诈骗/营销垃圾/灰产引流”越高）

要求：
- 输出必须是 JSON，不要输出任何多余文本。
- 如果信息不足，二值取0，强度取0.0~0.3的保守值，不要凭空臆测。
文本如下：
{text}
"""


    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "只输出 JSON 对象，不要额外文字。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    content = completion.choices[0].message.content
    obj: Any = json.loads(content)
    return obj
