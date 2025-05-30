from __future__ import annotations
from autogen import ConversableAgent
import os, sys, re
from openai import OpenAI
import base64
from mimetypes import guess_type
import argparse
from pathlib import Path

# OpenAI API key setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    sys.exit("❗ Set the OPENAI_API_KEY environment variable first.")

client = OpenAI(api_key=OPENAI_API_KEY)
LLM_CFG = {"config_list": [{"model": "gpt-4o-mini", "api_key": OPENAI_API_KEY}]}

def post_process(text: str, filename: str) -> str:
    core = text.strip().rstrip('.')  # 去掉尾部的句点
    return f"{filename}, A painting of \"{core}\", in the style of {{}}"

# Local image to data URL
def local_image_to_data_url(image_path):
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{base64_encoded_data}"

# Build image agent
def build_image_agent(name, msg):
    return ConversableAgent(name=name, system_message=msg, llm_config=LLM_CFG)  

IMAGE_AGENT = build_image_agent(
    "image_agent",
    """
    You are a context description assistant.
    Your task is to look at the image and produce a factual, objective description of:
    - all objects present in the scene
    - the quantity of each object
    - their relative positions and relationships
    - if any actions are occurring, briefly describe them
    You must NOT mention any artistic style, brushstrokes, color tones, art techniques, or any other aesthetic qualities.
    Please output exactly one line in this format:\n
    Ex. "a bustling street market, filled with people, stalls, a carriage, shops, and surrounding buildings."
    """
    )  

# Send image to agent and get description
def describe_image(image_source: str, prompt: str = "Please describe the spec of this image."):
    # Check if the image path exists
    if os.path.exists(image_source):
        image_url = local_image_to_data_url(image_source)
    else:
        image_url = image_source  # Assume it is a URL
    
    # Content supports multi-modal format: [{type: "text", text: ...}, {type: "image_url", image_url: {"url": ...}}]
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_url}}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": IMAGE_AGENT.system_message},
            {"role": "user", "content": content}
        ],
        max_tokens=300
    )
    # Get content from API response
    return response.choices[0].message.content


def process_file(path: Path, prompt: str) -> str:
    raw = describe_image(str(path), prompt)
    return post_process(raw, path.name)

def process_folder(folder: Path, prompt: str) -> list[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
    results: list[str] = []
    for img in folder.iterdir():
        if img.is_file() and img.suffix.lower() in exts:
            results.append(process_file(img, prompt))
    return results

# Test
if __name__ == "__main__":
    
    prompt = (
    "Please provide a detailed, factual, objective description of the image content:\n"
    "- List all visible objects and their quantities.\n"
    "- Describe each object's color and relative position.\n"
    "- If there are people or animals, state what actions they are performing.\n"
    "Do not mention any artistic style, brushstrokes, lighting effects, or aesthetic qualities.\n"
    "Please output line by line in this format, instead of using list format:\n"
    "Ex. a bustling street market, filled with people, stalls, a carriage, shops, and surrounding buildings."
    "IMPORTANT: output exactly one single sentence with NO newline or line breaks."
    )

    parser = argparse.ArgumentParser(description='Analyze image content')
    parser.add_argument('--image_path', '-i', type=str, required=False, help='Path to the image to analyze')
    parser.add_argument('--folder_path', '-f', type=str, required=False, help='Path to the folder containing images to analyze')
    parser.add_argument('-o', '--output', dest='out_file', required=False, help='If set, write all descriptions to this txt file')
    
    args = parser.parse_args()
    test_local_image = args.image_path
    test_local_folder = args.folder_path

    # if os.path.exists(test_local_image):
    #     result = describe_image(test_local_image, prompt=prompt)
    #     result = post_process(result)
    #     print("Description of the image:", result)
    # else:
    #     print("❗Error: The image does not exist.")
    
    # collect output
    lines: list[str] = []

    if args.image_path:
        img = Path(args.image_path)
        if not img.exists():
            sys.exit(f"❗ Error: {img} does not exist.")
        lines.append(process_file(img, prompt))
    else:
        folder = Path(args.folder_path)
        if not folder.is_dir():
            sys.exit(f"❗ Error: {folder} is not a directory.")
        lines.extend(process_folder(folder, prompt))

    # sort by filename
    lines.sort(key=lambda line: line.split(',', 1)[0])

    # write output into out_file
    if args.out_file:
        with open(args.out_file, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(line + "\n")
        print(f"✅ Descriptions saved to {args.out_file}")
    else:
        for line in lines:
            print(line)