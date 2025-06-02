import torch
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import numpy as np
import random
from PIL import Image
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

IMAGE_DIR = "images"
CAPTION_PATH = "prompt/qwen_caption-complicated.txt"
NEG_PATH = "prompt/qwen_neg-complicated.txt"

QWEN_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
QWEN_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    QWEN_MODEL
)
QWEN_model.to(DEVICE)

QWEN_processor = AutoProcessor.from_pretrained(QWEN_MODEL)

def qwen_generator(messages, images=None):
    text = QWEN_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = QWEN_processor(
        text=[text],
        images=images if images is not None else None,
        videos=None,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(DEVICE)
    generated_ids = QWEN_model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = QWEN_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0].strip()

def generate_caption_from_image(image: Image.Image):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Briefly describe the main content elements of this image for stable diffusion training. "
                        "Start with 'A painting of ...' and list the essential objects, people, actions, and scene elements. "
                        "For prominent objects or objects that appear multiple times, list them separately. "
                        "Keep it concise - use no more than 15-20 words total. "
                        "Exclude any style descriptions, adjectives, or artistic interpretations. "
                        "End with ', in the style of {}'. "
                        "Example: A painting of a large tree, mountains, two birds flying, a lake, in the style of {}"
                    ),
                },
                {"type": "image", "image": image},
            ],
        }
    ]
    return qwen_generator(messages, images=[image])

def generate_caption_from_prompt(prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Given the following image description: \"{prompt}\"\n"
                        "List all style-independent elements in this description. "
                        "Only return a valid Python list of objects, actions, or scene elements that are not related to artistic style. "
                        "Do not output anything except the Python list. Do not include explanations or extra text. "
                        "Example of correct output: ['cat', 'tree', 'run'] "
                    ),
                },
            ],
        }
    ]
    output = qwen_generator(messages)
    import ast
    try:
        result = ast.literal_eval(output)
        if isinstance(result, list):
            return result
        else:
            print("Output is not a valid Python list.")
            return []
    except Exception:
        return []

def get_caption_and_negative(image_dir, caption_path, neg_path):
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
    with open(caption_path, "w", encoding="utf-8") as fcap, \
         open(neg_path, "w", encoding="utf-8") as fneg:
        for fname in sorted(image_files):
            img_path = os.path.join(image_dir, fname)
            try:
                image = Image.open(img_path).convert("RGB")
                caption = generate_caption_from_image(image)
                print(f"Generated caption for {fname}: {caption}")
                if not caption.lower().startswith("a painting of"):
                    caption = "A painting of " + caption
                if not caption.strip().endswith(", in the style of {}"):
                    caption = caption.rstrip(".") + ", in the style of {}"
                fcap.write(f"{fname}, {caption}\n")
                lst = generate_caption_from_prompt(caption)
                print(f"Generated style-independent list for {fname}: {lst}")
                fneg.write(f"{fname}: {lst}\n")
            except Exception as e:
                print(f"Error processing {fname}: {e}")

if __name__ == "__main__":
    get_caption_and_negative(IMAGE_DIR, CAPTION_PATH, NEG_PATH)