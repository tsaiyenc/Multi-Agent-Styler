import os
from autogen import ConversableAgent
# from autogen import config_list_from_json
from mimetypes import guess_type
import base64
import json
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

# === Some helper functions ===
# === Image preprocess function ===
def preprocess_image(image_path, max_size=512):
    from PIL import Image
    import io
    
    img = Image.open(image_path)
    # Keep aspect ratio and scale
    img.thumbnail((max_size, max_size))
    
    # Convert to smaller format
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=85)
    return buffer.getvalue()

# === Image to data URL ===
def local_image_to_data_url(image_path):
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'
    
    # Preprocess image
    preprocessed_image = preprocess_image(image_path)
    base64_encoded_data = base64.b64encode(preprocessed_image).decode('utf-8')
    return f"data:{mime_type};base64,{base64_encoded_data}"

# === Agent Definitions ===
# === StyleCritique Agent ===
style_critique = ConversableAgent(
    name="StyleCritiqueAgent",
    system_message=(
        '''
        You are an art style expert and a expert painter.
        Your job is to evaluate how closely the style of the given image matches the style of the given style image.
        Consider color palette, brush strokes, lighting, composition, and overall aesthetic cohesion.
        Only focus on the style of the image, not the content or image quality.
        You will disccuss with ReviewerAgent about the score of style, if you disagree the score, you will provide your score and reasoning.
        score range: 0-10. If the style of two images are the same, the score should be 10. If the style of two images are totally different, the score should be 0.
        If you receive more than one score, you will tell the ReviewerAgent which one you prefer.
        '''
    ),
    llm_config=LLM_CFG,
)

# === ContentAnalyzer Agent ===
content_analyzer = ConversableAgent(
    name="ContentAnalyzerAgent",
    system_message=(
        '''
        You are a content analysis expert.
        Your role is to analyze whether the given image is the same as the given content.
        Only focus on the content of the image, not the style or image quality.
        You will disccuss with ReviewerAgent about the score of context, if you disagree the score, you will provide your score and reasoning.
        score range: 0-10. If the content of two images are the same, the score should be 10. If the content of two images are totally different, the score should be 0.
        If you receive more than one score, you will tell the ReviewerAgent which one you prefer.
        '''
    ),
    llm_config=LLM_CFG,
)

# === Reviewer Agent ===
reviewer = ConversableAgent(
    name="ReviewerAgent",
    system_message=(
        '''
        You are the helper on giving the score.
        You will receive a analysis of image:
        - if from StyleCritiqueAgent: The style analysis and how the image matches the given style.
        - if from ContentAnalyzerAgent: The content analysis and how the image matches the given content.
        Your job is to propose ONLY ONE score of 0-10 (0: worst match, 10: best match), and why you give this score.
        You will disccuss with StyleCritiqueAgent or ContentAnalyzerAgent about the score.
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

    initial_prompt = {
        "role": "user",
        "content": "Evaluate the given image."
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
    ### Variables
    PROMPT_LOGGER_PATH = "./test/prompt_suggestion.csv"
    HISTORY_LOGGER_PATH = "./test/history_log.json"
    TOTAL_ROUNDS = 3
    # IMG_URL = "https://example.com/path/to/your/image.jpg"
    GENERATED_IMG_PATH = "./outputs/list/list01/sample/A painting of a dog _2.png"
    GENERATED_IMG_URL = local_image_to_data_url(GENERATED_IMG_PATH)
    STYLE_IMG_PATH = "./images/01.png"
    STYLE_IMG_URL = local_image_to_data_url(STYLE_IMG_PATH)
    ORIGINAL_PROMPT = "A painting of a dog"
    # SOURCE_PROMPTS = ['blah', 'blah2', 'blah3']  # Example source prompts, replace with actual prompts
    # REFERENCE_STYLE = "Van Gogh"
    # REFERENCE_CONTEXT = "childhood connection with nature"
    # image_description = (
    #     "This is a watercolor illustration of a little girl playing with a fox in a lush forest. "
    #     "The target style is Studio Ghibli. The intended theme is 'childhood connection with nature'."
    # )
    
    ### Running the pipeline
    print("Starting the image evaluation pipeline...")
    from logger import HistoryLogger
    history_logger = HistoryLogger(HISTORY_LOGGER_PATH)

    style_score_summary = run_debate("Style", style_critique, reviewer, GENERATED_IMG_URL, STYLE_IMG_URL, rounds=TOTAL_ROUNDS, history_logger = history_logger)
    context_score_summary = run_debate("Context", content_analyzer, reviewer, GENERATED_IMG_URL, ORIGINAL_PROMPT, rounds=TOTAL_ROUNDS, history_logger = history_logger)

    print("Style Score Content:\n", style_score_summary["content"])
    print("Context Score Content:\n", context_score_summary["content"])

    # Dump history logger
    history_logger.log_history(target_image_url=GENERATED_IMG_PATH, total_rounds=TOTAL_ROUNDS)
    print(f"History logged to {HISTORY_LOGGER_PATH}")

    ### Summarizing Guidance
    print("\nSummarizing Guidance from Scores...")
    guidance_summary = summarize_guidance_from_score(style_score_summary, context_score_summary)
    # if guidance_summary is not a json, then call the summarizer again
    try:
        json.loads(guidance_summary["content"])
    except:
        guidance_summary = summarize_guidance_from_score(style_score_summary, context_score_summary)
    print("Guidance Summary Content:\n", guidance_summary["content"])
    

    ### Logging the results
    from logger import ImageEvalLogger
    from ast import literal_eval
        
    logger = ImageEvalLogger(PROMPT_LOGGER_PATH)
    contents = literal_eval(guidance_summary["content"])
    logger.log(
        target_image_url=GENERATED_IMG_PATH,
        overall_score=(contents["Overall Score"]),
        original_description=ORIGINAL_PROMPT,
        suggested_description=(contents["Style"] + " " + contents["Context"])    
    )

    print(f"Results logged to {PROMPT_LOGGER_PATH}")
    print("Pipeline execution completed.")