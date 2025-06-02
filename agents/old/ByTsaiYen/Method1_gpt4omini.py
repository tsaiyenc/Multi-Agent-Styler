from __future__ import annotations
from autogen import ConversableAgent
import os, sys
from openai import OpenAI
import base64
from mimetypes import guess_type
import argparse

# OpenAI API key setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    sys.exit("❗ Set the OPENAI_API_KEY environment variable first.")

client = OpenAI(api_key=OPENAI_API_KEY)
LLM_CFG = {"config_list": [{"model": "gpt-4o-mini", "api_key": OPENAI_API_KEY}]}


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

IMAGE_AGENT = build_image_agent("image_agent", "You are a helpful assistant that can understand the content of an image and generate a description. You are also able to understand the user's request and provide the corresponding image.")


# Send image to agent and get description
def describe_image(image_source: str, prompt: str = "Please describe the content of this image."):
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

# Test
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze image content')
    parser.add_argument('--image_path', '-i', type=str, required=True, help='Path to the image to analyze')
    args = parser.parse_args()
    test_local_image = args.image_path
    if os.path.exists(test_local_image):
        result = describe_image(test_local_image)
        print("Description of the image:", result)
    else:
        print("❗Error: The image does not exist.")
