from __future__ import annotations
from autogen import ConversableAgent
import os, sys
from openai import OpenAI
import base64
from mimetypes import guess_type
import argparse
import csv
from datetime import datetime

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
def build_agent(name, msg):
    return ConversableAgent(name=name, system_message=msg, llm_config=LLM_CFG)

CONTENT_DESCRIPTER = build_agent(
    "image_content_descripter", 
    '''
    You are an expert image content analyzer. Your task is to provide detailed, factual descriptions of image contents.
    Focus on:
    - Main objects and their actions
    - Key objects and their relationships
    - Object arrangements and layout
    - Important object details
    Guidelines:
    - Be objective and precise
    - Only describe objects, no style or artistic elements
    - Use clear, concise language
    - Maintain a professional tone
    - Exclude any background descriptions
    '''
)
OBJECT_FETCHER = build_agent(
    "image_object_fetcher", 
    '''
    You are a precise object detection specialist. Your task is to identify and list all significant objects in the image.
    Requirements:
    - List each object with its basic description
    - Include object attributes (e.g., "a woman playing violin", "a red sports car")
    - Focus on distinct, identifiable objects
    - Exclude background elements and style descriptions
    - Maintain consistent format: [object1, object2, object3, ...]
    Example output:
    ["a young woman playing violin", "a wooden music stand", "a microphone"]
    '''
)
OBJECT_PROMPT_GENERATOR = build_agent(
    "image_object_prompt_generator", 
    '''
    You are a creative prompt engineer specializing in image-to-text generation. Your task is to create detailed, descriptive prompts based on image objects.
    Guidelines:
    - Focus on object relationships and interactions
    - Include spatial arrangements and positioning
    - Describe actions and activities
    - Exclude all style-related descriptions
    - Exclude background descriptions
    Example:
    Input: ["a woman playing violin", "a wooden music stand"]
    Output: "A woman playing a violin, with a wooden music stand positioned in front of her"
    '''
)
IMAGE_OBJECT_POSITION_ANALYZER = build_agent(
    "image_object_position_analyzer",
    '''
    You are an expert in analyzing object positions within images. Your task is to identify and describe the precise location of each object in the image.
    Requirements:
    - Describe each object's position using spatial terms (e.g., "top-left corner", "center", "bottom-right")
    - Include relative positions between objects (e.g., "above", "below", "to the left of")
    - Specify distance relationships when relevant (e.g., "close to", "far from")
    - Exclude background and style descriptions
    - Use consistent JSON format: [{"object": "object1", "position": "position1"}, {"object": "object2", "position": "position2"}, ...]
    Example output:
    [
        {"object": "a blue door", "position": "in the center of the image"},
        {"object": "a woman sitting on a chair", "position": "at the bottom-right corner"}
    ]
    '''
)

# Send image to agent and get description
def describe_image(image_source: str, agent: ConversableAgent, prompt: str = "Please describe only the objects in this image, excluding any background elements, style descriptions, or artistic elements."):
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
            {"role": "system", "content": agent.system_message},
            {"role": "user", "content": content}
        ],
        max_tokens=300
    )
    # Get content from API response
    return response.choices[0].message.content

def process_image_with_all_agents(image_path: str):
    agents = {
        "Content Descripter": CONTENT_DESCRIPTER,
        "Object Fetcher": OBJECT_FETCHER,
        "Object Prompt Generator": OBJECT_PROMPT_GENERATOR,
        "Object Position Analyzer": IMAGE_OBJECT_POSITION_ANALYZER
    }
    
    results = {}
    for agent_name, agent in agents.items():
        result = describe_image(image_path, agent)
        results[agent_name] = result
    
    return results

def save_results_to_csv(results_list, output_file):
    # 準備 CSV 的欄位名稱
    fieldnames = ['Image_Number'] + list(results_list[0][1].keys())
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for image_num, results in results_list:
            row = {'Image_Number': f"{image_num:02d}"}
            row.update(results)
            writer.writerow(row)

if __name__ == "__main__":
    all_results = []
    
    for i in range(1, 11):  # 處理 1 到 10 的圖片
        test_local_image = f"./images/{i:02d}.png"
        if os.path.exists(test_local_image):
            print(f"\n處理圖片 {i:02d}.png:")
            results = process_image_with_all_agents(test_local_image)
            all_results.append((i, results))
            
            # 同時在控制台顯示結果
            for agent_name, result in results.items():
                print(f"\n{agent_name} 的輸出結果:")
                print(result)
        else:
            print(f"❗Error: The image {i:02d}.png does not exist.")
    
    # 生成包含時間戳的檔案名稱
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"./test/image_analysis_results_{timestamp}.csv"
    
    # 將所有結果存成 CSV
    save_results_to_csv(all_results, output_file)
    print(f"\n所有結果已儲存至 {output_file}")
