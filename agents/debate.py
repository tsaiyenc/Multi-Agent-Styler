#!/usr/bin/env python3
import os
import csv
import base64
from mimetypes import guess_type
from typing import Any, Dict, List

from openai import OpenAI               # pip install openai
from autogen import ConversableAgent     # 如果你目前版本是 AssistantAgent，也請改用 ConversableAgent


SUMMARIZER_CSV_PATH = "SUMMARIZER.CSV"
DEBATE_HISTORY_PATH = "DEBATE_HISTORY.json"
TOTAL_ROUNDS          = 3

# ==========================
# === local image → Data URL ===
# ==========================
def local_image_to_data_url(path: str) -> str:
    """
    讀取本地圖片檔案，轉成 Data URL（Base64 編碼）。
    範例回傳： "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA..."
    """
    mime, _ = guess_type(path)
    if not mime:
        mime = "application/octet-stream"
    with open(path, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"


# ==========================
# === LLM & Agent 設定  ===
# ==========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    raise RuntimeError("請先設定環境變數 OPENAI_API_KEY")

# 啟用 OpenAI 客戶端（gpt-4o-mini 支援多模態）
client = OpenAI(api_key=OPENAI_API_KEY)

# 因為我們要直接呼叫 openai.chat.completions.create()，不再用 AssistantAgent 的 llm_config 驗證
# 但為了保留「system_message」的管理，我們用 ConversableAgent 來包裝 system_message
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

# ==========================
# === Pipeline Functions ===
# ==========================
def run_critique_stage(
    generated_data_url: str,
    reference_data_url: str,
    history_logger: Any = None
) -> str:
    """
    Critique 階段：將兩張 Data URL 圖片送給 GPT-4o-mini，回傳 CritiqueAgent 的文字回覆。
    """
    # 把兩張圖片和最後的文字說明，組成 entries
    entries: List[dict] = [
        {"type": "image_url", "image_url": {"url": generated_data_url}},
        {"type": "image_url", "image_url": {"url": reference_data_url}},
        {"type": "text",      "text": "Above: the first image is the GENERATED image; the second is the REFERENCE image. Compare the style and give a score 0–10 with reasoning."}
    ]

    messages = [
        {"role": "system", "content": critique_agent.system_message},
        {"role": "user",   "content": entries}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=512
    )
    reply_text = response.choices[0].message.content.strip()
    if history_logger:
        history_logger.logger_dict["Style"] += "\n" + reply_text
    return reply_text


def run_analyzer_stage(
    generated_data_url: str,
    history_logger: Any = None
) -> str:
    """
    Analyzer 階段：將一張 Data URL 圖片送給 GPT-4o-mini，回傳 AnalyzerAgent 的文字回覆。
    """
    entries: List[dict] = [
        {"type": "image_url", "image_url": {"url": generated_data_url}},
        {"type": "text",      "text": "Above: this is the GENERATED image. Evaluate its content/context, give a score 0–10 with reasoning."}
    ]

    messages = [
        {"role": "system", "content": analyzer_agent.system_message},
        {"role": "user",   "content": entries}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=512
    )
    reply_text = response.choices[0].message.content.strip()
    if history_logger:
        history_logger.logger_dict["Context"] += "\n" + reply_text
    return reply_text

def run_scorer_stage(
    critique_reply: str,
    analyzer_reply: str,
    history_logger: Any = None
) -> str:
    """
    Scorer 階段：以純文字方式把 Critique & Analyzer 的回覆送給 GPT-4o-mini，回傳 JSON-like 文字回覆。
    """
    combined = (
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

    messages = [
        {"role": "system", "content": scorer_agent.system_message},
        {"role": "user",   "content": combined}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=512
    )
    reply_text = response.choices[0].message.content.strip()
    if history_logger:
        history_logger.logger_dict["Context"] += "\n[SCORE] " + reply_text
    return reply_text


def run_summarizer_stage(
    scorer_reply: str,
    history_logger: Any = None
) -> str:
    """
    Summarizer 階段：以純文字方式把 Scorer 的 JSON-like 回覆送給 GPT-4o-mini，回傳最終 JSON。
    """
    prompt = (
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

    messages = [
        {"role": "system", "content": summarizer_agent.system_message},
        {"role": "user",   "content": prompt}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=512
    )
    reply_text = response.choices[0].message.content.strip()
    if history_logger:
        history_logger.logger_dict["Final Summary"] += "\n" + reply_text
    return reply_text


def save_summarizer_output_to_csv(
    generated_data_url: str,
    reference_data_url: str,
    summarizer_json: str
) -> None:
    """
    把 Summarizer 回覆的 JSON-like 字串解析後，附加寫進 summarizer_log.csv。
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
            generated_data_url,
            reference_data_url,
            data.get("New Prompt", ""),
            data.get("Style", ""),
            data.get("Context", "")
        ])


# ==========================
# === Main Pipeline 執行區 ===
# ==========================
if __name__ == "__main__":
    SUMMARIZER_CSV_PATH = "SUMMARIZER.CSV"
    DEBATE_HISTORY_PATH = "DEBATE_HISTORY.csv"

    print("Starting the multi-agent style/context pipeline...")

    from logger import HistoryLogger
    history_logger = HistoryLogger(DEBATE_HISTORY_PATH)

    # 1. 先把本地路徑轉成 Data URL
    local_gen = "debate_image/gen/gen_1.png"
    local_ref = "debate_image/reference/reference.png"

    if not os.path.exists(local_gen):
        raise FileNotFoundError(f"找不到生成圖：{local_gen}")
    if not os.path.exists(local_ref):
        raise FileNotFoundError(f"找不到參考圖：{local_ref}")

    generated_data_url = local_image_to_data_url(local_gen)
    reference_data_url = local_image_to_data_url(local_ref)

    # 2. Critique 階段（兩張圖片）
    critique_reply = run_critique_stage(
        generated_data_url=generated_data_url,
        reference_data_url=reference_data_url,
        history_logger=history_logger
    )
    print(f"\n--- CritiqueAgent 回覆 ---\n{critique_reply}\n")

    # 3. Analyzer 階段（單張圖片）
    analyzer_reply = run_analyzer_stage(
        generated_data_url=generated_data_url,
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

    # 6. 儲存到 CSV
    save_summarizer_output_to_csv(
        generated_data_url=local_gen,
        reference_data_url=local_ref,
        summarizer_json=summarizer_reply
    )
    print(f"Summarizer results appended to: {SUMMARIZER_CSV_PATH}")
    print("Pipeline completed.")
    
    history_logger.log_history(
        target_image_url=local_gen,
        total_rounds=TOTAL_ROUNDS
    )
