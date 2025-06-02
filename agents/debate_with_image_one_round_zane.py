#!/usr/bin/env python3
import os
import csv
import base64
from mimetypes import guess_type
from typing import Any, List

from openai import OpenAI               # pip install openai
from autogen import ConversableAgent     # 可用 AssistantAgent 替代，但需保留 system_message

# ======================================
# === 常數：CSV/JSON 檔案名稱、輪數 ===
# ======================================
SUMMARIZER_CSV_PATH   = "SUMMARIZER.CSV"
DEBATE_HISTORY_PATH   = "DEBATE_HISTORY.json"
TOTAL_ROUNDS          = 1


# ========================================
# === 將本地圖片轉成 Data URL 的函式  ===
# ========================================
def local_image_to_data_url(path: str) -> str:
    """
    讀取本地圖片檔案，轉成 Data URL（Base64 編碼）。
    回傳形如 "data:image/png;base64,xxxxx..."。
    """
    mime, _ = guess_type(path)
    if not mime:
        mime = "application/octet-stream"
    with open(path, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"


# ===========================
# === LLM & Agent 設定     ===
# ===========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    raise RuntimeError("請先設定環境變數 OPENAI_API_KEY")

# 建立 OpenAI 客戶端（gpt-4o-mini 支援多模態）
client = OpenAI(api_key=OPENAI_API_KEY)

# 使用 ConversableAgent 主要是保留 system_message；不再用它呼叫 API
critique_agent = ConversableAgent(
    name="CritiqueAgent",
    system_message=(
        "You are an expert in evaluating style consistency between two images.\n"
        "You will receive exactly two image URLs in Data URL format.\n"
        "Your task:\n"
        "  1. Compare how closely the generated image’s style matches the reference image.\n"
        "  2. Consider color palette, brush strokes, lighting, composition, and overall aesthetic cohesion.\n"
        "  3. Assign a style consistency score from 0 to 10 and justify your reasoning.\n"
        "  4. If you find any issues with your own reasoning or score, challenge yourself and revise.\n"
    ),
    llm_config={"model": "gpt-4o-mini", "api_key": OPENAI_API_KEY}
)

analyzer_agent = ConversableAgent(
    name="AnalyzerAgent",
    system_message=(
        "You are an expert at assessing the content and context of a single image.\n"
        "You will receive exactly one image URL in Data URL format.\n"
        "Your task:\n"
        "  1. Evaluate whether the generated image is contextually consistent with its intended theme.\n"
        "  2. Assign a content/context score from 0 to 10 and justify your reasoning.\n"
        "  3. If you find any issues with your own reasoning or score, challenge yourself and revise.\n"
    ),
    llm_config={"model": "gpt-4o-mini", "api_key": OPENAI_API_KEY}
)

scorer_agent = ConversableAgent(
    name="ScorerAgent",
    system_message=(
        "You are responsible for synthesizing style and context evaluations into concrete improvement suggestions.\n"
        "Input will include:\n"
        "  - The CritiqueAgent’s output (style evaluation)\n"
        "  - The AnalyzerAgent’s output (context evaluation)\n"
        "Your task:\n"
        "  1. Assign a final style score (0–10) and a final context score (0–10).\n"
        "  2. Explain where and how style can be improved.\n"
        "  3. Explain where and how context can be improved.\n"
        "  4. Return a structured reply in JSON-like format:\n"
        "     {\n"
        "       \"Style Score\": \"<0–10>\",\n"
        "       \"Style Improvements\": \"<text>\",\n"
        "       \"Context Score\": \"<0–10>\",\n"
        "       \"Context Improvements\": \"<text>\"\n"
        "     }\n"
    ),
    llm_config={"model": "gpt-4o-mini", "api_key": OPENAI_API_KEY}
)

summarizer_agent = ConversableAgent(
    name="SummarizerAgent",
    system_message=(
        "You are a summarizer that takes structured style/context improvement suggestions\n"
        "and produces a single JSON output containing:\n"
        "  - \"New Prompt\": \"<a rewritten prompt that incorporates improvements>\"\n"
        "  - \"Style\": \"<concise style suggestion>\"\n"
        "  - \"Context\": \"<concise context suggestion>\"\n"
        "Your input will be the ScorerAgent’s JSON output.\n"
        "Return only valid JSON in this exact format:\n"
        "{\n"
        "  \"New Prompt\": \"<text>\",\n"
        "  \"Style\": \"<text>\",\n"
        "  \"Context\": \"<text>\"\n"
        "}\n"
    ),
    llm_config={"model": "gpt-4o-mini", "api_key": OPENAI_API_KEY}
)


# =====================================
# === Pipeline Functions with Logger ===
# =====================================

def run_critique_stage(
    generated_data_url: str,
    reference_data_url: str,
    local_gen: str,
    history_logger: Any = None
) -> str:
    """
    Critique 階段：將兩張 Data URL 圖片送給 GPT-4o-mini，多模態呼叫後，
    回傳 CritiqueAgent 的文字回覆，同時只把「本機路徑」+「回覆文字」存到 logger_dict["Style"]。
    """
    message_history: List[dict] = [
        {"role": "system", "content": critique_agent.system_message},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": generated_data_url}},
            {"type": "image_url", "image_url": {"url": reference_data_url}},
            {"type": "text", "text": "Above: the first is GENERATED; second is REFERENCE. Compare style, give a score 0–10 with reasoning."}
        ]}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=message_history,
        max_tokens=512
    )
    reply_text = response.choices[0].message.content.strip()

    if history_logger:
        entry = {
            "image_path": local_gen,
            "critique_reply": reply_text
        }
        import json
        history_logger.logger_dict["Style"] += json.dumps(entry, ensure_ascii=False) + "\n"

    return reply_text


def run_analyzer_stage(
    generated_data_url: str,
    local_gen: str,
    history_logger: Any = None
) -> str:
    """
    Analyzer 階段：將一張 Data URL 圖片送給 GPT-4o-mini，多模態呼叫後，
    回傳 AnalyzerAgent 的文字回覆，同時只把「本機路徑」+「回覆文字」存到 logger_dict["Context"]。
    """
    message_history: List[dict] = [
        {"role": "system", "content": analyzer_agent.system_message},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": generated_data_url}},
            {"type": "text", "text": "Above: this is GENERATED. Evaluate content/context, give a score 0–10 with reasoning."}
        ]}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=message_history,
        max_tokens=512
    )
    reply_text = response.choices[0].message.content.strip()

    if history_logger:
        entry = {
            "image_path": local_gen,
            "analyzer_reply": reply_text
        }
        import json
        history_logger.logger_dict["Context"] += json.dumps(entry, ensure_ascii=False) + "\n"

    return reply_text


def run_scorer_stage(
    critique_reply: str,
    analyzer_reply: str,
    history_logger: Any = None
) -> str:
    """
    Scorer 階段：純文字呼叫 GPT-4o-mini，將 Critique & Analyzer 回覆串成 prompt，
    回傳 JSON-like 字串，同時只把「critique_reply + analyzer_reply + scorer_reply」存到 logger_dict["Context"]。
    """
    combined_prompt = (
        f"Here is the CritiqueAgent’s output (style evaluation):\n"
        f"{critique_reply}\n\n"
        f"Here is the AnalyzerAgent’s output (context evaluation):\n"
        f"{analyzer_reply}\n\n"
        "Based on these two evaluations, provide:\n"
        "  1. A final style score (0–10) and how style can be improved.\n"
        "  2. A final context score (0–10) and how context can be improved.\n"
        "Return ONLY valid JSON in this format:\n"
        "{\n"
        "  \"Style Score\": \"<0–10>\",\n"
        "  \"Style Improvements\": \"<text>\",\n"
        "  \"Context Score\": \"<0–10>\",\n"
        "  \"Context Improvements\": \"<text>\"\n"
        "}\n"
    )
    message_history: List[dict] = [
        {"role": "system", "content": scorer_agent.system_message},
        {"role": "user", "content": combined_prompt}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=message_history,
        max_tokens=512
    )
    reply_text = response.choices[0].message.content.strip()

    if history_logger:
        entry = {
            "critique_reply": critique_reply,
            "analyzer_reply": analyzer_reply,
            "scorer_reply": reply_text
        }
        import json
        history_logger.logger_dict["Context"] += json.dumps(entry, ensure_ascii=False) + "\n"

    return reply_text


def run_summarizer_stage(
    scorer_reply: str,
    history_logger: Any = None
) -> str:
    """
    Summarizer 階段：純文字呼叫 GPT-4o-mini，將 Scorer 的 JSON-like 回覆包成 prompt，
    回傳最終的 JSON，同時只把「summarizer_reply」存到 logger_dict["Final Summary"]。
    """
    summarizer_prompt = (
        f"Here is the ScorerAgent’s structured feedback:\n"
        f"{scorer_reply}\n\n"
        "Your task:\n"
        "  1. Rewrite a single new prompt that incorporates the style and context improvements.\n"
        "  2. Include 'Style' and 'Context' fields as concise suggestions.\n"
        "Return ONLY valid JSON in this exact format:\n"
        "{\n"
        "  \"New Prompt\": \"<text>\",\n"
        "  \"Style\": \"<text>\",\n"
        "  \"Context\": \"<text>\"\n"
        "}\n"
    )
    message_history: List[dict] = [
        {"role": "system", "content": summarizer_agent.system_message},
        {"role": "user", "content": summarizer_prompt}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=message_history,
        max_tokens=512
    )
    reply_text = response.choices[0].message.content.strip()

    if history_logger:
        entry = {
            "summarizer_reply": reply_text
        }
        import json
        history_logger.logger_dict["Final Summary"] += json.dumps(entry, ensure_ascii=False) + "\n"

    return reply_text


def save_summarizer_output_to_csv(
    generated_path: str,
    reference_path: str,
    summarizer_json: str
) -> None:
    """
    將 SummarizerAgent 的 JSON-like 回覆解析後，附加寫到 CSV。
    只記錄「本地路徑」而非 Data URL。
    """
    from ast import literal_eval

    data = literal_eval(summarizer_json)
    file_exists = os.path.isfile(SUMMARIZER_CSV_PATH)
    with open(SUMMARIZER_CSV_PATH, mode="a", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow([
                "GeneratedImage",
                "ReferenceImage",
                "New Prompt",
                "Style Suggestion",
                "Context Suggestion"
            ])
        writer.writerow([
            generated_path,
            reference_path,
            data.get("New Prompt", ""),
            data.get("Style", ""),
            data.get("Context", "")
        ])


# ============================================
# === Main Pipeline: 呼叫各階段 & Logger ===
# ============================================
if __name__ == "__main__":
    print("Starting the multi-agent style/context pipeline…")

    from logger import HistoryLogger
    history_logger = HistoryLogger(DEBATE_HISTORY_PATH)

    # 1. 把本地路徑轉成 Data URL，供各階段多模態呼叫使用
    local_gen = "debate_image/gen/gen_1.png"
    local_ref = "debate_image/reference/reference.png"

    if not os.path.exists(local_gen):
        raise FileNotFoundError(f"找不到生成圖：{local_gen}")
    if not os.path.exists(local_ref):
        raise FileNotFoundError(f"找不到參考圖：{local_ref}")

    generated_data_url  = local_image_to_data_url(local_gen)
    reference_data_url  = local_image_to_data_url(local_ref)

    # 2. Critique 階段（兩張 Data URL）
    critique_reply = run_critique_stage(
        generated_data_url=generated_data_url,
        reference_data_url=reference_data_url,
        local_gen=local_gen,
        history_logger=history_logger
    )
    print(f"\n--- CritiqueAgent 回覆 ---\n{critique_reply}\n")

    # 3. Analyzer 階段（單張 Data URL）
    analyzer_reply = run_analyzer_stage(
        generated_data_url=generated_data_url,
        local_gen=local_gen,
        history_logger=history_logger
    )
    print(f"\n--- AnalyzerAgent 回覆 ---\n{analyzer_reply}\n")

    # 4. Scorer 階段（純文字）
    scorer_reply = run_scorer_stage(
        critique_reply=critique_reply,
        analyzer_reply=analyzer_reply,
        history_logger=history_logger
    )
    print(f"\n--- ScorerAgent 回覆 (JSON-like) ---\n{scorer_reply}\n")

    # 5. Summarizer 階段（純文字）
    summarizer_reply = run_summarizer_stage(
        scorer_reply=scorer_reply,
        history_logger=history_logger
    )
    print(f"\n--- SummarizerAgent 最終 JSON ---\n{summarizer_reply}\n")

    # 6. 儲存到 CSV，只記錄「本地路徑」而非 Data URL
    save_summarizer_output_to_csv(
        generated_path=local_gen,
        reference_path=local_ref,
        summarizer_json=summarizer_reply
    )
    print(f"Summarizer results appended to: {SUMMARIZER_CSV_PATH}")

    # 7. 把三個階段的簡化紀錄（不含 Data URL）寫入 Debate History JSON
    history_logger.log_history(
        target_image_url=local_gen,
        total_rounds=TOTAL_ROUNDS
    )
    print("History saved to", DEBATE_HISTORY_PATH)

    print("Pipeline completed.")
