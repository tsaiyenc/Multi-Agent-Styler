import os
from autogen import ConversableAgent, register_function
from mimetypes import guess_type
import base64
import json
import torch
import clip
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import vgg19
from torch.nn.functional import mse_loss
import math
import argparse

# === LLM Configuration ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-proj-").strip()
LLM_CFG = {
    "config_list": [
        {
            "model": "gpt-4o-mini",
            "api_key": OPENAI_API_KEY,
            "max_tokens": 2048
        }
    ]
}

# === Device ===
device = "cuda" if torch.cuda.is_available() else "cpu"

# === CLIP Model ===
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# === VGG Model ===
vgg = vgg19(pretrained=True).features[:21].to(device).eval()

# === VGG Transform ===
vgg_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# === Some helper functions ===
# === Image preprocess function ===
def preprocess_image(image_path, max_size=512):
    try:
        from PIL import Image
        import io
        
        img = Image.open(image_path)
        # Keep aspect ratio and scale
        img.thumbnail((max_size, max_size))
        
        # Convert to smaller format
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        return buffer.getvalue()
    except Exception as e:
        print(f"Error in preprocess_image: {str(e)}")
        raise

# === Image to data URL ===
def local_image_to_data_url(image_path):
    try:
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'
        
        # Preprocess image
        preprocessed_image = preprocess_image(image_path)
        base64_encoded_data = base64.b64encode(preprocessed_image).decode('utf-8')
        return f"data:{mime_type};base64,{base64_encoded_data}"
    except Exception as e:
        print(f"Error in local_image_to_data_url: {str(e)}")
        raise

# === Evaluate Functions ===
# === Style Score ===
def evaluate_style_score(image: str, style_image: str) -> float:
    """Evaluate the style similarity between two images.
    Args:
        image (str): The URL or path of the generated image
        style_image (str): The URL or path of the reference style image
    Returns:
        float: The style similarity score (0-10)
    """
    try:
        # If it's a URL, download the image first
        if image.startswith('data:'):
            import base64
            import io
            # Extract base64 data from the data URL
            image_data = image.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            gen_img = Image.open(io.BytesIO(image_bytes))
        else:
            gen_img = Image.open(image)
            
        if style_image.startswith('data:'):
            import base64
            import io
            # Extract base64 data from the data URL
            style_data = style_image.split(',')[1]
            style_bytes = base64.b64decode(style_data)
            style_img = Image.open(io.BytesIO(style_bytes))
        else:
            style_img = Image.open(style_image)

        gen_img = vgg_transform(gen_img).unsqueeze(0).to(device)
        style_img = vgg_transform(style_img).unsqueeze(0).to(device)

        with torch.no_grad():
            gen_feat = vgg(gen_img)  # Extract features from the generated image
            style_feat = vgg(style_img)  # Extract features from the style image
            mse = mse_loss(gen_feat, style_feat).item()  # Calculate the mean square error between the two features
            # Convert the MSE score to the 0-10 range
            # The smaller the MSE, the more similar the style, so use 1/(1+mse) to convert
            # When mse=0, the score is 10; when mse approaches infinity, the score approaches 0
            score = 10 * (1 / (1 + math.log(1 + mse)))  # Convert the score to the 0-10 range
        return score
    except Exception as e:
        print(f"Error in evaluate_style_score: {str(e)}")
        return 0.0  # Return the lowest score

# === Context Score ===
def evaluate_context_score(image: str, context: str) -> float:
    """Evaluate the context similarity between an image and a text description.
    Args:
        image (str): The URL or path of the image
        context (str): The text description    
    Returns:
        float: The context similarity score (0-10)
    """
    try:
        # If it's a URL, download the image first
        if image.startswith('data:'):
            import base64
            import io
            # Extract base64 data from the data URL
            image_data = image.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(image_bytes))
        else:
            img = Image.open(image)

        image = clip_preprocess(img).unsqueeze(0).to(device)
        context = clip.tokenize([context]).to(device)

        with torch.no_grad():
            image_feat = clip_model.encode_image(image)  # Extract image features
            context_feat = clip_model.encode_text(context)  # Extract text features
            image_feat /= image_feat.norm(dim=-1, keepdim=True)  # Normalize image features
            context_feat /= context_feat.norm(dim=-1, keepdim=True)  # Normalize text features
            similarity = (image_feat @ context_feat.T).item()  # Calculate the cosine similarity
            # Convert the cosine similarity (-1,1) to the 0-10 range
            # When the similarity is 1, the score is 10; when the similarity is -1, the score is 0
            score = 5 * (similarity + 1)  # Convert the score to the 0-10 range
        return score
    except Exception as e:
        print(f"Error in evaluate_context_score: {str(e)}")
        return 0.0  # Return the lowest score

# === Agent Definitions ===
# === StyleCritique Agent ===
style_critique = ConversableAgent(
    name="StyleCritiqueAgent",
    system_message=(
        "You are an art style critic. Your job is to evaluate how well a given image matches a specified target style "
        "(e.g., Studio Ghibli, watercolor, cyberpunk). "
        "You will critique the style score proposed by ScorerAgent and suggest corrections with reasoning."
    ),
    llm_config=LLM_CFG,
)

# === ContentAnalyzer Agent ===
content_analyzer = ConversableAgent(
    name="ContentAnalyzerAgent",
    system_message=(
        "You are a content analysis expert. Your role is to assess whether the image content is contextually consistent with the given theme, "
        "prompt, or story. You will challenge the context score proposed by ScorerAgent, and provide analytical feedback."
    ),
    llm_config=LLM_CFG,
)

# === Reviewer Agent ===
reviewer = ConversableAgent(
    name="ReviewerAgent",
    system_message=(
        "You are the final scoring agent. Given a visual description of an image, your job is to propose a style score "
        "and a context score (both on a scale from 0 to 10). You must defend your scores when challenged, and revise them if needed."
    ),
    llm_config=LLM_CFG,
)

# === Summarizer Agent ===
PROMPT_EXAMPLES = ''
summarizer = ConversableAgent(
    name="SummarizerAgent",
    system_message=(
        '''
        You are an image feedback summarizer. 
        Given a debate summary from the ReviewerAgent about the aesthetic, style, and context of a target image, 
        your task is to extract and propose:
        1. A concise target style description
        2. A concise target context description
        These will guide the next round of image generation to better match the intended aesthetic goals.
        RETURN ONLY IN JSON FORMAT:
        {
            "Overall Score": "<overall score>",
            "Style": "<style description>",
            "Context": "<context description>"
        }
        '''
    ),
    llm_config=LLM_CFG,
)

# === Pipeline runner ===
# === Debate ===
def run_debate(score_type, evaluator, reviewer, img_url, reference, rounds=3, history_logger=None):
    print(f"\n===== Starting {score_type} Debate =====")

    # Calculate initial score
    if score_type == "Style":
        score = evaluate_style_score(img_url, reference)
    elif score_type == "Context":
        score = evaluate_context_score(img_url, reference)

    initial_prompt = {
        "role": "user",
        "content": f"""Your task is to evaluate the **{score_type}** of the image on a scale from 0 to 10.
Begin by proposing a score and justification, then proceed with discussion and revisions if necessary.
FOCUS ONLY ON THE {score_type.upper()} SCORE, NOT THE OVERALL IMAGE QUALITY.
The objective {score_type.lower()} score is {score:.2f}/10. Please consider this score in your analysis and discussion."""
    }
    message_history = [initial_prompt]
    
    img_message = []
    if score_type == "Style":
        img_message = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": img_url}},
                {"type": "image_url", "image_url": {"url": reference}}
            ]}
        ]
    elif score_type == "Context":
        img_message = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": img_url}},
                {"type": "text", "text": f"The content of the image should be: {reference}"}
            ]}
        ]

    for i in range(rounds):
        print(f"\n--- Round {i + 1} ---")

        # Evaluator's turn
        message_history_add_img = message_history + img_message
        response_evaluator = evaluator.generate_reply(message_history_add_img, return_message=True)
        if isinstance(response_evaluator, str):
            response_evaluator = {"role": "assistant", "name": evaluator.name, "content": response_evaluator}
        message_history.append(response_evaluator)
        print(f"\n{evaluator.name}:\n{response_evaluator['content']}")

        # Reviewer's turn
        response_reviewer = reviewer.generate_reply(message_history, return_message=True)
        if isinstance(response_reviewer, str):
            response_reviewer = {"role": "assistant", "name": reviewer.name, "content": response_reviewer}
        message_history.append(response_reviewer)
        print(f"\n{reviewer.name}:\n{response_reviewer['content']}")

    print(f"\n===== End of {score_type} Debate =====")

    # Final summary
    final_summary = reviewer.generate_reply(message_history, return_message=True)
    if isinstance(final_summary, str):
        final_summary = {"role": "assistant", "name": reviewer.name, "content": final_summary}

    print(f"\nFinal {score_type} Score Summary from {reviewer.name}:\n{final_summary['content']}\n")
    
    # Put logger history
    if history_logger:
        history_logger.logger_dict[score_type] = message_history
        history_logger.logger_dict['Final Summary'] += final_summary['content']
    return final_summary

# === Summarize ===
def summarize_guidance_from_score(sytle_score_summary:dict, context_score_summary:dict):
    summary_prompt = [
        {"role": "user", "content": "Based on the following scoring summary, extract ideal descriptions for STYLE and CONTEXT."},
        {"role": "assistant", "name": "ReviewerAgent", "content": sytle_score_summary["content"]},
        {"role": "assistant", "name": "ReviewerAgent", "content": context_score_summary["content"]}
    ]

    summary_response = summarizer.generate_reply(summary_prompt, return_message=True)
    if isinstance(summary_response, str):
        summary_response = {"role": "assistant", "name": summarizer.name, "content": summary_response}

    # print("\n Summarized Target Guidance (from SummarizerAgent):")
    # print(summary_response["content"])
    return summary_response

# === Example usage ===
if __name__ == "__main__":
    try:
        ### Parse arguments
        parser = argparse.ArgumentParser(description='Image evaluation pipeline with debate rounds')
        parser.add_argument('--prompt-logger', type=str, default="./test/prompt_suggestion.csv",
                          help='Path to save prompt suggestions')
        parser.add_argument('--history-logger', type=str, default="./test/history_log.json",
                          help='Path to save debate history')
        parser.add_argument('--rounds', type=int, default=3,
                          help='Number of debate rounds')
        parser.add_argument('--generated-img', type=str, required=True,
                          help='Path to the generated image')
        parser.add_argument('--style-img', type=str, required=True,
                          help='Path to the style reference image')
        parser.add_argument('--prompt', type=str, required=True,
                          help='Original prompt used for image generation')
        
        args = parser.parse_args()
        
        ### Convert image to data URL
        try:
            GENERATED_IMG_URL = local_image_to_data_url(args.generated_img)
            STYLE_IMG_URL = local_image_to_data_url(args.style_img)
        except Exception as e:
            print(f"Error converting images to data URLs: {str(e)}")
            raise
        
        ### Running the pipeline
        print("Starting the image evaluation pipeline...")
        from logger import HistoryLogger
        history_logger = HistoryLogger(args.history_logger)

        try:
            style_score_summary = run_debate("Style", style_critique, reviewer, GENERATED_IMG_URL, STYLE_IMG_URL, rounds=args.rounds, history_logger = history_logger)
            context_score_summary = run_debate("Context", content_analyzer, reviewer, GENERATED_IMG_URL, args.prompt, rounds=args.rounds, history_logger = history_logger)
        except Exception as e:
            print(f"Error during debate rounds: {str(e)}")
            raise

        print("Style Score Content:\n", style_score_summary["content"])
        print("Context Score Content:\n", context_score_summary["content"])

        # Dump history logger
        try:
            history_logger.log_history(target_image_url=args.generated_img, total_rounds=args.rounds)
            print(f"History logged to {args.history_logger}")
        except Exception as e:
            print(f"Error logging history: {str(e)}")

        ### Summarizing Guidance
        print("\nSummarizing Guidance from Scores...")
        try:
            guidance_summary = summarize_guidance_from_score(style_score_summary, context_score_summary)
            # if guidance_summary is not a json, then call the summarizer again
            try:
                json.loads(guidance_summary["content"])
            except:
                guidance_summary = summarize_guidance_from_score(style_score_summary, context_score_summary)
            print("Guidance Summary Content:\n", guidance_summary["content"])
        except Exception as e:
            print(f"Error summarizing guidance: {str(e)}")
            raise
        
        ### Logging the results
        try:
            from logger import ImageEvalLogger
            from ast import literal_eval
                
            logger = ImageEvalLogger(args.prompt_logger)
            contents = literal_eval(guidance_summary["content"])
            logger.log(
                target_image_url=args.generated_img,
                overall_score=(contents["Overall Score"]),
                original_description=args.prompt,
                suggested_description=(contents["Style"] + " " + contents["Context"])    
            )

            print(f"Results logged to {args.prompt_logger}")
            print("Pipeline execution completed.")
        except Exception as e:
            print(f"Error logging results: {str(e)}")
            raise

    except Exception as e:
        print(f"Fatal error in main execution: {str(e)}")
        raise