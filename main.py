# main.py

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from components import GeminiLLM, GroqLLM, GroqConfig, GeminiConfig, PromptGenerator

# ---------- Experiment helpers ----------

def parse_mcq_answer(raw_text: str) -> List[str]:
    """
    Parse responses like:
        - <answer>A,B</answer>
        - <answer>C</answer>
        - <answer>(none)</answer>
    Returns a list of option labels (e.g., ["A", "B"]) or [] for none.
    """
    try:
        answer_part = raw_text.split('<answer>')[1].split('</answer>')[0]
    except:
        raise ValueError("Error while parsing answer, returned text:", raw_text)

    # Split on commas and clean up
    labels = [token.strip().upper() for token in answer_part.split(",") if token.strip()]
    # Only keep valid options
    valid_labels = [x for x in labels if x in {"A", "B", "C", "D"}]
    return valid_labels


def parse_json_response(raw_text: str) -> Dict:
    """
    Very simple JSON parser.
    In real evaluations, you might need to:
    - strip extra text around the JSON
    - handle JSON-with-trailing-commas, etc.

    For teaching, we assume the model obeys the 'JSON only' instruction.
    """

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        # Fallback: try to extract the first {...} block (naive)
        json_match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        # If all fails, return empty
        return {}


@dataclass
class SimulatedParticipant:
    """
    Represents one 'user' in your human study.
    Each participant has their own conversation history,
    but uses the same underlying LLM.
    """
    id: str
    history: List[Tuple[str, str]] = field(default_factory=list)


# ---------- Core experiment logic ----------

def run_single_model_experiment(llm):
    """
    Runs the 3 questions for one 'participant',
    showing:
      1) MCQ-style responses
      2) Free-flow responses
      3) Structured JSON responses
    """
    prompt_gen = PromptGenerator()

    # Example employee descriptions for A–D
    employees = {
        "A": "Alice – consistently exceeds targets, mentors juniors, strong team player.",
        "B": "Bob – meets most targets, reliable but rarely takes initiative.",
        "C": "Carol – recently joined, very high potential but limited track record so far.",
        "D": "Dan – experienced but performance has been declining over the last year.",
    }

    # Conversation history for this participant
    history: List[Tuple[str, str]] = []

    print("=== Question 1: MCQ (who deserves a salary hike) ===")
    mcq_prompt = prompt_gen.mcq_salary_hike(employees)
    print("Prompt:\n", mcq_prompt)
    mcq_raw, thought = llm.generate(mcq_prompt, history=history)
    print("Raw model response:", mcq_raw)
    print("Thoughts: ", thought)
    mcq_parsed = parse_mcq_answer(mcq_raw)
    print("Parsed MCQ answer:", mcq_parsed)
    print()

    # Update history (user + model turn)
    history.append(("user", mcq_prompt))
    history.append(("model", mcq_raw))

    print("=== Question 2: Free-flow criteria ===")
    free_prompt = prompt_gen.free_flow_criteria()
    print("Prompt:\n", free_prompt)
    free_raw, thoughts = llm.generate(free_prompt, history=history)  # now history contains Q1 + A1
    print("Model response (free-flow):\n", free_raw)
    print("Thoughts", thoughts)
    print()

    # Update history again
    history.append(("user", free_prompt))
    history.append(("model", free_raw))

    print("=== Question 3: Structured JSON (who gets what raise) ===")
    json_prompt = prompt_gen.structured_json_raises(employees)
    json_raw, thoughts = llm.generate(json_prompt, history=history)
    json_parsed = parse_json_response(json_raw)

    print("Prompt:\n", json_prompt)
    print("Raw model response:", json_raw)
    print("Parsed JSON object:", json.dumps(json_parsed, indent=2))
    print("Thoughts", thoughts)
    print()

    # Here you could add evaluation logic, e.g.:
    # - check that parsed JSON uses only A–D
    # - compute fairness metrics
    # - compare to a gold standard, etc.


def run_chat_history_demo(llm, num_participants: int = 3):
    """
    Demonstrates chat history:
    - We simulate multiple 'participants' sharing the same underlying LLM.
    - Each participant answers the same questions, but we maintain a separate history.
    """

    prompt_gen = PromptGenerator()

    employees = {
        "A": "Alice – consistently exceeds targets, mentors juniors, strong team player.",
        "B": "Bob – meets most targets, reliable but rarely takes initiative.",
        "C": "Carol – recently joined, very high potential but limited track record so far.",
        "D": "Dan – experienced but performance has been declining over the last year.",
    }

    participants = [
        SimulatedParticipant(id=f"worker_{i+1}") for i in range(num_participants)
    ]

    for participant in participants:
        print(f"\n#############################")
        print(f"# Conversation with {participant.id}")
        print(f"#############################\n")

        # Q1: MCQ
        mcq_prompt = prompt_gen.mcq_salary_hike(employees)
        mcq_raw, thoughts = llm.generate(mcq_prompt, history=participant.history)
        participant.history.append(("user", mcq_prompt))
        participant.history.append(("model", mcq_raw))

        print("[Q1] MCQ raw:", mcq_raw)
        print(f"Thoughts: {thoughts}")

        # Q2: Free-flow, conditioned on Q1 history
        free_prompt = prompt_gen.free_flow_criteria()
        free_raw, thoughts = llm.generate(free_prompt, history=participant.history)
        participant.history.append(("user", free_prompt))
        participant.history.append(("model", free_raw))
        print(f"Thoughts: {thoughts}")

        print("[Q2] Criteria answer (short):", free_raw[:200], "...\n")

        # Q3: JSON, still with full history
        json_prompt = prompt_gen.structured_json_raises(employees)
        json_raw, thoughts = llm.generate(json_prompt, history=participant.history)
        participant.history.append(("user", json_prompt))
        participant.history.append(("model", json_raw))

        json_parsed = parse_json_response(json_raw)
        print("[Q3] JSON parsed:", json.dumps(json_parsed, indent=2))
        print(f"Thoughts: {thoughts}")


if __name__ == "__main__":
    config = GeminiConfig(
        model_name="gemini-2.5-flash",  # or another Gemini model
        temperature=0.4,
        max_output_tokens=None,
        thinking_budget=4000,
        include_thoughts=True
    )

    file_path = "Auth_keys/gemini.txt"  # Replace with your file name/path

    with open(file_path, 'r') as file:
        GEMINI_KEY = file.read()

    llm = GeminiLLM(config=config, api_key=GEMINI_KEY)


    # For teaching, you can toggle these:
    # print("Running single-model experiment:\n")
    # run_single_model_experiment(llm)

    config = GroqConfig(
        model_name="qwen/qwen3-32b",  # or another Gemini model
        temperature=0.4,
        max_output_tokens=None,
        reasoning_format="parsed",
        reasoning_effort=None,
        # include_reasoning=True
    )

    file_path = "Auth_keys/groq.txt"  # Replace with your file name/path

    with open(file_path, 'r') as file:
        GROQ_KEY = file.read()

    llm = GroqLLM(config, api_key=GROQ_KEY)
    

    # For teaching, you can toggle these:
    print("Running single-model experiment:\n")
    run_single_model_experiment(llm)

    # print("\n\nRunning multi-participant chat-history demo:\n")
    # run_chat_history_demo(llm, num_participants=2)
