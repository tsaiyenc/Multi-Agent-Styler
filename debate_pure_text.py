import os
from autogen import AssistantAgent
from autogen import config_list_from_json

# === LLM Configuration ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-proj-").strip()
LLM_CFG = {
    "config_list": [
        {
            "model": "gpt-4o-mini",
            "api_key": OPENAI_API_KEY
        }
    ]
}

# === Agent Definitions ===
critique = AssistantAgent(
    name="CritiqueAgent",
    system_message=(
        "You are an art style critic. Your job is to evaluate how well a given image matches a specified target style "
        "(e.g., Studio Ghibli, watercolor, cyberpunk). "
        "You will critique the style score proposed by ScorerAgent and suggest corrections with reasoning."
    ),
    llm_config=LLM_CFG,
)

analyzer = AssistantAgent(
    name="AnalyzerAgent",
    system_message=(
        "You are a content analysis expert. Your role is to assess whether the image content is contextually consistent with the given theme, "
        "prompt, or story. You will challenge the context score proposed by ScorerAgent, and provide analytical feedback."
    ),
    llm_config=LLM_CFG,
)

scorer = AssistantAgent(
    name="ScorerAgent",
    system_message=(
        "You are the final scoring agent. Given a visual description of an image, your job is to propose a style score "
        "and a context score (both on a scale from 0 to 10). You must defend your scores when challenged, and revise them if needed."
    ),
    llm_config=LLM_CFG,
)

# === Debate Function with message format safety check ===
def run_debate(score_type, evaluator, scorer, description, rounds=3):
    print(f"\n===== Starting {score_type} Debate =====")

    initial_prompt = {
        "role": "user",
        "content": f"""Please assess the following image:

Image Description:
\"\"\"{description}\"\"\"

Your task is to evaluate the **{score_type}** of the image on a scale from 0 to 10.
Begin by proposing a score and justification, then proceed with discussion and revisions if necessary.
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

    print(f"\n🎯 Final {score_type} Score Summary from {scorer.name}:\n{final_summary['content']}\n")
    return final_summary

# === Example usage ===
if __name__ == "__main__":
    image_description = (
        "This is a watercolor illustration of a little girl playing with a fox in a lush forest. "
        "The target style is Studio Ghibli. The intended theme is 'childhood connection with nature'."
    )

    style_score_summary = run_debate("Style", critique, scorer, image_description, rounds=3)
    context_score_summary = run_debate("Context", analyzer, scorer, image_description, rounds=3)

    print("Style Score Content:\n", style_score_summary["content"])
    print("Context Score Content:\n", context_score_summary["content"])
