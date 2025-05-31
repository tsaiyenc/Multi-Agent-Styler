#!/usr/bin/env python3
"""
negative_prompt.py

Generate a negative prompt by comparing a target diffusion prompt
against an image's context description or a batch of contextual descriptions,
to identify unwanted content for negative prompting.
"""
from __future__ import annotations
from autogen import ConversableAgent
from openai import OpenAI
import os
import sys
import textwrap
import argparse
import re

# Import describe_image and IMAGE_AGENT from temp.py for single-image mode
from temp import describe_image, IMAGE_AGENT

# ─── Configuration ─────────────────────────────────────────────────────────────

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    sys.exit("❗ ERROR: Please set the OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=OPENAI_API_KEY)
LLM_CFG = {"config_list": [{"model": "gpt-4o-mini", "api_key": OPENAI_API_KEY}]}

# ─── Agent Builders ─────────────────────────────────────────────────────────────

def build_negative_agent() -> ConversableAgent:
    """
    Construct a ConversableAgent for generating negative prompts
    that exclude specific concrete objects and living beings.
    """
    system_message = textwrap.dedent("""
        You are a negative-prompt generator.
        You will receive two inputs:
        1) A target diffusion prompt that the user plans to use.
        2) One or more factual, objective image descriptions.

        Your task: From each description, extract only the concrete
        object nouns or living creatures that are irrelevant to the
        target diffusion prompt. Output each list as comma-separated terms.
    """).strip()

    return ConversableAgent(
        name="negative_agent",
        system_message=system_message,
        llm_config=LLM_CFG
    )

# ─── Negative Prompt Generation ─────────────────────────────────────────────────

def generate_negative_prompt(
    diffusion_prompt: str,
    description: str,
    agent: ConversableAgent
) -> str:
    """
    Compare the diffusion prompt and a single context description,
    returning a comma-separated list of unwanted phrases.
    """
    user_entries = [
        {"type": "text", "text": f"Target Prompt: {diffusion_prompt}"},
        {"type": "text", "text": f"Image Description: {description}"},
        {"type": "text", "text": "List the parts of the description that are NOT relevant to the target prompt."}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": agent.system_message},
                {"role": "user",   "content": user_entries}
            ],
            max_tokens=100
        )
    except Exception as e:
        sys.exit(f"❗ ERROR: OpenAI API request failed: {e}")

    return response.choices[0].message.content.strip()

# ─── CLI Entrypoint ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate negative prompts from one or multiple contextual descriptions."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "-i", "--image_path",
        help="Path to a single image file (generates one description then negative prompt)"
    )
    mode.add_argument(
        "-c", "--cc_file",
        help="Path to a text file containing batch contextual descriptions"
    )
    parser.add_argument(
        "-p", "--prompt",
        required=True,
        help="The diffusion prompt to compare against"
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to output txt file for results (optional)"
    )
    args = parser.parse_args()

    negative_agent = build_negative_agent()
    diffusion_prompt = args.prompt

    results: list[str] = []

    if args.image_path:
        # Single image flow
        context = describe_image(
            args.image_path,
            prompt="Please provide a factual, objective description of the image content."
        )
        unwanted = generate_negative_prompt(diffusion_prompt, context, negative_agent)
        results.append(f"{os.path.basename(args.image_path)}: {unwanted}")
    else:
        # Batch flow: parse each line in cc_file
        if not os.path.exists(args.cc_file):
            sys.exit(f"❗ ERROR: {args.cc_file} not found.")
        pattern = re.compile(r"^(?P<fname>[^,]+), (?P<desc>.*), in the style of \{\}$")
        with open(args.cc_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                m = pattern.match(line)
                if not m:
                    print(f"⚠️ Skipping unrecognized line: {line}")
                    continue
                fname = m.group("fname")
                desc  = m.group("desc")
                unwanted = generate_negative_prompt(diffusion_prompt, desc, negative_agent)
                results.append(f"{fname}: {unwanted}")

    # Output results either to stdout or to file
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f_out:
                for r in results:
                    f_out.write(r + "\n")
            print(f"✅ Results saved to {args.output}")
        except Exception as e:
            sys.exit(f"❗ ERROR: Failed to write output file: {e}")
    else:
        for r in results:
            print(r)

if __name__ == "__main__":
    main()
