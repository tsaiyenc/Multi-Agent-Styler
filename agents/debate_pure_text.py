import os
from autogen import AssistantAgent
from autogen import config_list_from_json

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

# === Agent Definitions ===
# === Critique Agent ===
critique = AssistantAgent(
    name="CritiqueAgent",
    system_message=(
        "You are an art style critic. Your job is to evaluate how well a given image matches a specified target style "
        "(e.g., Studio Ghibli, watercolor, cyberpunk). "
        "You will critique the style score proposed by ScorerAgent and suggest corrections with reasoning."
    ),
    llm_config=LLM_CFG,
)

# === Analyzer Agent ===
analyzer = AssistantAgent(
    name="AnalyzerAgent",
    system_message=(
        "You are a content analysis expert. Your role is to assess whether the image content is contextually consistent with the given theme, "
        "prompt, or story. You will challenge the context score proposed by ScorerAgent, and provide analytical feedback."
    ),
    llm_config=LLM_CFG,
)

# === Scorer Agent ===
scorer = AssistantAgent(
    name="ScorerAgent",
    system_message=(
        "You are the final scoring agent. Given a visual description of an image, your job is to propose a style score "
        "and a context score (both on a scale from 0 to 10). You must defend your scores when challenged, and revise them if needed."
    ),
    llm_config=LLM_CFG,
)

# === Summarizer Agent ===
PROMPT_EXAMPLES = ''
summarizer = AssistantAgent(
    name="SummarizerAgent",
    system_message=(
        "You are an image feedback summarizer. Given a debate summary from the ScorerAgent "
        "about the aesthetic, style, and context of a target image, your task is to extract and propose:\n"
        "1. A concise target style description\n"
        "2. A concise target context description\n"
        "These will guide the next round of image generation to better match the intended aesthetic goals.\n"
        "RETURN ONY IN JSON FORMAT:\n"
        "{\"Overall Score\": \"<overall score>\", \"Style\": \"<style description>\", \"Context\": \"<context description>\"}"
        
    ),
    llm_config=LLM_CFG,
)

# === Pipeline runner ===
# === Debate ===
def run_debate(score_type, evaluator, scorer, description, rounds=3, history_logger=None):
    print(f"\n===== Starting {score_type} Debate =====")

    initial_prompt = {
        "role": "user",
        "content": f"""Please assess the following image:

        Image Description:
        \"\"\"{description}\"\"\"

        Your task is to evaluate the **{score_type}** of the image on a scale from 0 to 10.
        Begin by proposing a score and justification, then proceed with discussion and revisions if necessary.
        FOCUS ONLY ON THE {score_type.upper()} SCORE, NOT THE OVERALL IMAGE QUALITY.
        """
    }

    message_history = [initial_prompt]

    for i in range(rounds):
        print(f"\n--- Round {i + 1} ---")

        # Evaluator's turn
        response_evaluator = evaluator.generate_reply(message_history, return_message=True)
        if isinstance(response_evaluator, str):
            response_evaluator = {"role": "assistant", "name": evaluator.name, "content": response_evaluator}
        message_history.append(response_evaluator)
        print(f"\n{evaluator.name}:\n{response_evaluator['content']}")

        # Scorer's turn
        response_scorer = scorer.generate_reply(message_history, return_message=True)
        if isinstance(response_scorer, str):
            response_scorer = {"role": "assistant", "name": scorer.name, "content": response_scorer}
        message_history.append(response_scorer)
        print(f"\n{scorer.name}:\n{response_scorer['content']}")

    print(f"\n===== End of {score_type} Debate =====")

    # Final summary
    final_summary = scorer.generate_reply(message_history, return_message=True)
    if isinstance(final_summary, str):
        final_summary = {"role": "assistant", "name": scorer.name, "content": final_summary}

    print(f"\nFinal {score_type} Score Summary from {scorer.name}:\n{final_summary['content']}\n")
    
    # Put logger history
    if history_logger:
        history_logger.logger_dict[score_type] = message_history
        history_logger.logger_dict['Final Summary'] += final_summary['content']
    return final_summary

# === Summarize ===
def summarize_guidance_from_score(sytle_score_summary:dict, context_score_summary:dict):
    summary_prompt = [
        {"role": "user", "content": "Based on the following scoring summary, extract ideal descriptions for STYLE and CONTEXT."},
        {"role": "assistant", "name": "ScorerAgent", "content": sytle_score_summary["content"]},
        {"role": "assistant", "name": "ScorerAgent", "content": context_score_summary["content"]}
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
    PROMPT_LOGGER_PATH = "WHOS_YOUR_DADDY.csv"
    HISTORY_LOGGER_PATH = "WHOS_YOUR_MAMA.json"
    TOTAL_ROUNDS = 3
    IMG_URL = "https://example.com/path/to/your/image.jpg"
    SOURCE_PROMPTS = ['blah', 'blah2', 'blah3']  # Example source prompts, replace with actual prompts
    REFERENCE_STYLE = "Van Gogh"
    REFERENCE_CONTEXT = "childhood connection with nature"
    image_description = (
        "This is a watercolor illustration of a little girl playing with a fox in a lush forest. "
        "The target style is Studio Ghibli. The intended theme is 'childhood connection with nature'."
    )
    
    ### Running the pipeline
    print("Starting the image evaluation pipeline...")
    from logger import HistoryLogger
    history_logger = HistoryLogger(HISTORY_LOGGER_PATH)

    style_score_summary = run_debate("Style", critique, scorer, image_description, rounds=TOTAL_ROUNDS, history_logger = history_logger)
    context_score_summary = run_debate("Context", analyzer, scorer, image_description, rounds=TOTAL_ROUNDS, history_logger = history_logger)

    print("Style Score Content:\n", style_score_summary["content"])
    print("Context Score Content:\n", context_score_summary["content"])

    # Dump history logger
    history_logger.log_history(target_image_url=IMG_URL, total_rounds=TOTAL_ROUNDS)
    print(f"History logged to {HISTORY_LOGGER_PATH}")

    ### Summarizing Guidance
    print("\nSummarizing Guidance from Scores...")
    guidance_summary = summarize_guidance_from_score(style_score_summary, context_score_summary)
    print("Guidance Summary Content:\n", guidance_summary["content"])
    

    ### Logging the results
    from logger import ImageEvalLogger
    from ast import literal_eval
        
    logger = ImageEvalLogger(PROMPT_LOGGER_PATH)
    contents = literal_eval(guidance_summary["content"])
    logger.log(
        target_image_url=IMG_URL,
        overall_score=(contents["Overall Score"]),
        original_description=image_description,
        suggested_description=(contents["Style"] + " " + contents["Context"])    
    )

    print(f"Results logged to {PROMPT_LOGGER_PATH}")
    print("Pipeline execution completed.")