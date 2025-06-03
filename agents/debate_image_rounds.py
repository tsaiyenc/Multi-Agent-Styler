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
# === Image preprocess function ==
# === Image to data URL ===
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
        
        # Define layers and weights to use
        layers = [0, 5, 10]  # Use first three convolutional layers
        weights = [0.5, 0.3, 0.2]  # Weight distribution
        
        with torch.no_grad():
            total_score = 0
            x_gen = gen_img
            x_style = style_img
            
            for i, layer_idx in enumerate(layers):
                # Calculate features up to current layer
                for j in range(layer_idx + 1):
                    x_gen = vgg[j](x_gen)
                    x_style = vgg[j](x_style)
                
                # Calculate style score for current layer
                layer_score = -mse_loss(x_gen, x_style).item()
                total_score += weights[i] * layer_score
                
                # Reset features for next layer calculation
                x_gen = gen_img
                x_style = style_img
        
        # Normalize the score to 0-10 range
        # Using normalization parameters specific to few-layers approach
        min_score = -8
        max_score = 0
        normalized_score = (total_score - min_score) / (max_score - min_score)
        final_score = max(0, min(normalized_score * 10, 10))
        
        return final_score
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
         '''
        You are an art style expert and professional painter. Your expertise lies in analyzing and comparing artistic styles.
        
        You will recieve two images, target image and reference image, respectively.

        Your primary responsibilities:
        1. Identify the Reference Image's style or movement, Ex: 
        - Academic Realism
        - Impressionism
        - Baroque
        - Rococo
        - Surrealism
        - Contemporary Digital
        - is it watercolor, ink painting, oil painting, pastel or anything else?

        2. Analyze the Reference Image's style by noting:
        - Color palette and harmony (muted vs. bold, warm vs. cool)
        - Brushstroke or texture (smooth blending vs. visible strokes, digital effects, etc.)
        - Lighting and shadow (soft gradations vs. dramatic contrast)
        - Composition and balance (figure poses, space, background detail)

        3. Compare Target image and Reference image:
        - Point out key similarities and differences in palette, brushwork/texture, lighting, and composition.
        - Cite specific areas where the Target aligns or diverges from the Reference's style.

        4. After receiving an objective style score, explain why that score is justified and offer concrete advice to improve the Target's style (for example: “Reduce saturation in highlights,” “Add more visible brush marks,” or “Simplify background details”).
                
        You can observe the image by the following simple question:
        1. Which explicit art style can this reference image belongs to? (e.g., Academic Realism, Impressionism, Abstract, Digital, etc.).
        2. Explicitly describe how the target image can be more closed to reference image?
        
        You will engage in a professional debate with the ReviewerAgent about your style analysis.
        Focus on providing actionable suggestions to improve the style matching in the next generation, do not use implicit adjective description like 'vibrant', 'elegant', 'dramatic',
        Remember: Your goal is to help create a better style description for the next image generation, so it is important to comunicate with review agent.
        
        Always keep your replies short and to the point:
        - Limit each answer to one or two sentences.
        - Use simple, direct language.
        - Provide only the most essential observation or question.
        '''
    ),
    llm_config=LLM_CFG,
)

# === ContentAnalyzer Agent ===
content_analyzer = ConversableAgent(
    name="ContentAnalyzerAgent",
    system_message=(
       '''
        You are a content and context analysis expert specializing in visual content evaluation. When you receive a Target Image and an original description C (e.g., “a painting of a dog”), perform these tasks:

        1. Analyze the content and context of the Target Image in detail:
           - Identify all objects, figures, and background elements present.
           - Note which objects occupy the most visual space and their relative importance.

        2. Compare the image content with the description C and provide:
           - A detailed analysis of alignment: which elements correctly match C, which are missing, and which are extraneous.
           - Specific observations about key elements: clarity, prominence, and relevance to C.

        3. Detect and remove or minimize irrelevant objects:
           - If an object appears but is not implied by C (for example, a woman in the painting when C is “a painting of a dog”), mark it as redundant and suggest removing or reducing its prominence.

        4. Identify missing but relevant objects or context that would strengthen alignment with C:
           - Suggest additions of objects, props, or environmental details that a viewer would expect given C. For “a painting of a dog,” you might propose adding a dog bowl, a favorite toy, a leash, or a park background—anything that reinforces “dog” as the subject.
           - Recommend ways to emphasize the intended object(s) from C: for example, repositioning, enlarging, or increasing contrast/lighting on the dog so it becomes the clear focal point.

        5. Provide concrete content improvement suggestions:
           - For each irrelevant or missing element, specify exactly what to remove, minimize, or add. Use phrasing like:
             • “Remove the human figure on the left margin; it distracts from the dog as the subject.”  
           - Advise on adjustments to ensure objects from C are clear and dominant: scale, placement, color contrast, or lighting changes.

        6. Engage in a professional debate with ReviewerAgent about your content analysis:
           - Be prepared to justify why certain objects are redundant or why suggested additions will improve alignment.
           - Focus on actionable, detailed recommendations that a future image generation prompt can use directly.

        You can observe the image by asking:
        1. Are there any objects not mentioned in C that occupy the majority of the scene? If so, they should be minimized or excluded.
        2. Are the objects described in C (the “dog”) clear, well-positioned, and prominent in the image? If not, they should be emphasized through size, contrast, or placement.

        Remember: Your ultimate goal is to supply precise content notes—removals and additions—so that the next image generation prompt can be more accurate and realistic. Keep your suggestions specific, using concrete object names and placement details.
        
        Always keep your replies short and to the point:
        - Limit each answer to one or two sentences.
        - Use simple, direct language.
        - Provide only the most essential observation or question.

        '''
    ),
    llm_config=LLM_CFG,
)

# === Reviewer Agent ===
reviewer = ConversableAgent(
    name="ReviewerAgent",
    system_message=(
         '''
        You are a critical reviewer and debate moderator for image analysis.

        You will not see the actual images, but you can talk to two agents who can truly see the images.
        
        If you are talking to StyleCritiqueAgent, your responsibilty is:
        Focus on the style.
        Analyze the style from StyleCritiqueAgent.
            You can ask StyleCritiqueAgent some questions to make it more clear.
            EX: if the painting is more about expressionism style, you can ask:
            1. "What is a typical expressionism style painting should be? Does the image match the style?"
            2. “What specific adjustments would make the Target Image's style more closely match the Reference Image's style?”

        If you are talking to ContentAnalyzerAgent, your responsibilty is:
        Focus on the content and objects.
        Analyze the description from ContentAnalyzerAgent.
            You can ask ContentAnalyzerAgent some questions to make it more clear.
            EX: if ContentAnalyzerAgent says there's something weird in the painting, you can ask:
            1. "Does the image content match with the image description?"
            2. "Is there anything redundant in the painting? Is there anything can be removed from the painting to make it more attach to the description C?"   
           
        If you are talking to ContentAnalyzerAgent, your responsibilty is:
        Analyze the content from ContentAnalyzerAgent.
        
        Your goal is to ensure a thorough, objective analysis and to produce a clear, actionable description
        that will guide the next image generation to better match both style and content requirements, so you should ask concise and accurate.
        
        
        
        '''
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
        1. A concise target style description.
        2. A concise target context description that we want in the painting, do not include redundant objets.
        These will guide the next round of image generation to better match the intended aesthetic goals.
        remember both of style description and context description should be limit into 35 words, respectively.

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
        "content": f"Evaluate the given image. The objective {score_type.lower()} score is {score:.2f}/10. Please consider this score in your analysis and discussion."
    }
    message_history = [initial_prompt]
    
    img_message = []
    if score_type == "Style":
        img_message = [
            {"role": "user", "content": [
                {"type": "text", "text": "This is target Image: " },
                {"type": "image_url", "image_url": {"url": img_url}},
                {"type": "text", "text": "This is reference image): " },
                {"type": "image_url", "image_url": {"url": reference}},
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
    conclusion_prompt = {
        "role": "user",
        "content": """
        Now you don't need to debate or communicate with the other two agents. Instead, please write a summary based on your entire debating history.

        The summary should include:
        1. How the painting's style should be improved.
        2. How the painting's content should be improved (e.g., what to add or remove).

        Example format:
        "The style should be improved by ..., and the content should include ... or remove ... .
        """
    }
    message_history.append(conclusion_prompt)
    
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
def summarize_guidance_from_score(original_prompt: str, style_score_summary:dict, context_score_summary:dict, max_retries=3, origin_prompt: str = ""):
    summary_prompt = [
        {"role": "user", "content": "Based on the following scoring summary and the original prompt, extract ideal descriptions for STYLE and CONTEXT."},
        {"role": "user", "content": f"The original image was generated with the prompt: \"{original_prompt}\"."},
        {"role": "assistant", "name": "ReviewerAgent", "content": style_score_summary["content"]},
        {"role": "assistant", "name": "ReviewerAgent", "content": context_score_summary["content"]},
        {"role": "user", "content": "Please output a new, improved prompt in JSON format with keys Overall Score, Style, and Context."},
        {"role": "user", "content": "Using simple words and keep the description concise."}
    ]

    for attempt in range(max_retries):
        try:
            summary_response = summarizer.generate_reply(summary_prompt, return_message=True)
            if isinstance(summary_response, str):
                summary_response = {"role": "assistant", "name": summarizer.name, "content": summary_response}
            
            # 驗證 JSON 格式
            content = summary_response["content"]
            # 如果內容不是以 { 開頭，嘗試提取 JSON 部分
            if not content.strip().startswith("{"):
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    content = json_match.group(0)
            
            # 驗證 JSON 格式是否正確
            parsed_json = json.loads(content)
            required_keys = ["Overall Score", "Style", "Context"]
            if all(key in parsed_json for key in required_keys):
                return summary_response
            
            # 如果缺少必要欄位，重試
            raise ValueError("Missing required JSON fields")
            
        except (json.JSONDecodeError, ValueError) as e:
            if attempt == max_retries - 1:
                # 最後一次嘗試失敗，返回預設值
                default_response = {
                    "role": "assistant",
                    "name": summarizer.name,
                    "content": json.dumps({
                        "Overall Score": "6.0",
                        "Style": "",
                        "Context": origin_prompt
                    })
                }
                return default_response
            continue
    
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
            guidance_summary = summarize_guidance_from_score(args.prompt, style_score_summary, context_score_summary, origin_prompt=args.prompt)
            # if guidance_summary is not a json, then call the summarizer again
            try:
                json.loads(guidance_summary["content"])
            except:
                guidance_summary = summarize_guidance_from_score(args.prompt, style_score_summary, context_score_summary, origin_prompt=args.prompt)
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

            if contents["Context"].endswith('.'):
                contents["Context"] = contents["Context"][:-1]

            logger.log(
                target_image_url=args.generated_img,
                overall_score=(contents["Overall Score"]),
                original_description=args.prompt,
                suggested_description=(contents["Context"] + ", in the style of {}. " + contents["Style"])    
                # suggested_description=(contents["Context"] + ", in the style of {}. ")    
            )

            print(f"Results logged to {args.prompt_logger}")
            print("Pipeline execution completed.")
        except Exception as e:
            print(f"Error logging results: {str(e)}")
            raise

    except Exception as e:
        print(f"Fatal error in main execution: {str(e)}")
        raise