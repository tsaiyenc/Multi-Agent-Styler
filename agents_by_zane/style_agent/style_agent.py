#!/usr/bin/env python3
from __future__ import annotations
from autogen import ConversableAgent
from openai import OpenAI
import base64
import os
import sys
from mimetypes import guess_type
import argparse
from pathlib import Path
import textwrap

# ─── Configuration ─────────────────────────────────────────────────────────────

# Load API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    sys.exit("❗ ERROR: Please set the OPENAI_API_KEY environment variable first.")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)
LLM_CFG = {
    "config_list": [
        {"model": "gpt-4o-mini", "api_key": OPENAI_API_KEY}
    ]
}

# ─── Utility Functions ─────────────────────────────────────────────────────────

def local_image_to_data_url(path: str) -> str:
    """
    Read a local image file and convert it to a data URL (base64-encoded).
    """
    mime, _ = guess_type(path)
    if not mime:
        mime = "application/octet-stream"
    data = base64.b64encode(open(path, "rb").read()).decode("utf-8")
    return f"data:{mime};base64,{data}"

def build_style_agent() -> ConversableAgent:
    """
    Construct and return a ConversableAgent configured for style comparison.
    """
    system_message = textwrap.dedent("""
        You are a style comparison assistant.
        Your task: Given one original image and multiple candidate images,
        choose the candidate that matches the original's painting style,
        color palette, brushstroke feel, and overall composition most closely.
        Respond with exactly the filename of the best match.
    """).strip()
    return ConversableAgent(
        name="style_agent",
        system_message=system_message,
        llm_config=LLM_CFG
    )

def select_best_match(original: str, candidates: list[str], agent: ConversableAgent) -> str:
    """
    Send the original + candidates to the LLM and return the filename
    of the candidate that most closely matches the original's style.
    """
    # Prepare multimodal message entries
    entries: list[dict] = []
    # Original image entry
    entries.append({
        "type": "image_url",
        "image_url": {"url": local_image_to_data_url(original)}
    })
    # Candidate images entries
    for p in candidates:
        if not os.path.exists(p):
            sys.exit(f"❗ ERROR: Candidate file not found: {p}")
        entries.append({
            "type": "image_url",
            "image_url": {"url": local_image_to_data_url(p)}
        })
    # Add user instruction
    entries.append({
        "type": "text",
        "text": (
            "Above, the first image is the ORIGINAL; the next images are candidates. "
            "Reply with exactly the filename of the one that matches the original's style best."
        )
    })

    # Call the GPT-4o-mini chat API
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": agent.system_message},
                {"role": "user",   "content": entries}
            ],
            max_tokens=16
        )
    except Exception as e:
        sys.exit(f"❗ ERROR: OpenAI API request failed: {e}")

    return response.choices[0].message.content.strip()

# ─── Main Script ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare an ORIGINAL image against AI-generated candidates by style."
    )
    parser.add_argument(
        "-o", "--origin_folder",
        required=True,
        help="Path to folder containing exactly one ORIGINAL image"
    )
    parser.add_argument(
        "-g", "--gen_folder",
        required=True,
        help="Path to folder containing AI-generated candidate images"
    )
    args = parser.parse_args()

    origin_dir = Path(args.origin_folder)
    gen_dir    = Path(args.gen_folder)

    # Validate origin folder
    if not origin_dir.is_dir():
        sys.exit(f"❗ ERROR: {origin_dir} is not a directory.")
    original_images = [
        f for f in origin_dir.iterdir()
        if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
    ]
    if len(original_images) != 1:
        sys.exit(f"❗ ERROR: origin_folder must contain exactly one image; found {len(original_images)}.")

    # Validate generated-folder
    if not gen_dir.is_dir():
        sys.exit(f"❗ ERROR: {gen_dir} is not a directory.")
    candidate_images = sorted([
        str(f) for f in gen_dir.iterdir()
        if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
    ])
    if not candidate_images:
        sys.exit("❗ ERROR: gen_folder contains no supported image files.")

    # Build the agent and select best match
    style_agent = build_style_agent()
    best_filename = select_best_match(
        str(original_images[0]),
        candidate_images,
        style_agent
    )

    print(f"Best style match: {best_filename}")

if __name__ == "__main__":
    main()
