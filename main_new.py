import json
from typing import List, Tuple, Dict
import pandas as pd
import re
from dataclasses import dataclass, field
from components import GeminiLLM, GroqLLM, GroqConfig, GeminiConfig, PromptGenerator

# -------------------------------------------------------------------
# CSV loader
# -------------------------------------------------------------------

def load_employee_examples(csv_path: str) -> List[Dict[str, str]]:
    """
    Reads the CSV file containing employee comparison examples and
    returns a list of dicts of the form:
        {"A": "...", "B": "...", "C": "...", "D": "..."}
    suitable for PromptGenerator.mcq_salary_hike / structured_json_raises.
    Expected columns: example_id, employee_A, employee_B, employee_C, employee_D
    """

    def _clean_cell(text: str, label: str) -> str:
        """
        Handles cells like 'A: Alice – ...' or just 'Alice – ...'.
        Strips an 'A:'/'B:' prefix if present.
        """
        if not isinstance(text, str):
            return str(text)
        text = text.strip()
        prefix = f"{label}:"
        if text.startswith(prefix):
            return text[len(prefix):].strip()
        return text

    df = pd.read_csv(csv_path)

    examples: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        example = {
            "A": _clean_cell(row["employee_A"], "A"),
            "B": _clean_cell(row["employee_B"], "B"),
            "C": _clean_cell(row["employee_C"], "C"),
            "D": _clean_cell(row["employee_D"], "D"),
        }
        examples.append(example)
    return examples


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


# -------------------------------------------------------------------
# Single-request helper functions
# -------------------------------------------------------------------

def call_free_flow_once(
    llm,
    prompt_gen: "PromptGenerator",
    history: List[Tuple[str, str]] | None = None,
):
    """
    Single free-flow request.
    - Builds the free-flow prompt.
    - Calls the LLM (optionally with history).
    - If history is provided, appends (user, prompt) and (model, response).
    Returns: (prompt, raw_response, thoughts)
    """
    free_prompt = prompt_gen.free_flow_criteria()
    for i in range(3):
        try:
            # raw, thoughts = llm.generate(free_prompt, history=history)
            response = llm.generate(free_prompt, history=history)
            print(response)
            break
        except Exception as e:
            if i == 2:
                raise e
            print(f"Retrying free-flow call due to error: {e}")

    if history is not None:
        history.append(("user", free_prompt))
        history.append(("model", raw))

    return free_prompt, raw, thoughts


def call_mcq_once(
    llm,
    prompt_gen: "PromptGenerator",
    employees: Dict[str, str],
    history: List[Tuple[str, str]] | None = None,
):
    """
    Single MCQ request for one example (A–D employees).
    - Builds the MCQ prompt using the given employees.
    - Calls the LLM (optionally with history).
    - Updates history if provided.
    - Parses the MCQ answer into a list of options.
    Returns: (prompt, raw_response, thoughts, parsed_answer)
    """
    mcq_prompt = prompt_gen.mcq_salary_hike(employees)
    for i in range(3):
        try:
            raw, thoughts = llm.generate(mcq_prompt, history=history)
            break
        except Exception as e:
            if i == 2:
                raise e
            print(f"Retrying free-flow call due to error: {e}")

    if history is not None:
        history.append(("user", mcq_prompt))
        history.append(("model", raw))

    parsed = parse_mcq_answer(raw)
    return mcq_prompt, raw, thoughts, parsed


def call_json_once(
    llm,
    prompt_gen: "PromptGenerator",
    employees: Dict[str, str],
    history: List[Tuple[str, str]] | None = None,
):
    """
    Single JSON-structured request for one example (A–D employees).
    - Builds the JSON prompt using the given employees.
    - Calls the LLM (optionally with history).
    - Updates history if provided.
    - Parses the JSON response into a Python dict.
    Returns: (prompt, raw_response, thoughts, parsed_json)
    """
    json_prompt = prompt_gen.structured_json_raises(employees)
    for i in range(3):
        try:
            raw, thoughts = llm.generate(json_prompt, history=history)
            break
        except Exception as e:
            if i == 2:
                raise e
            print(f"Retrying free-flow call due to error: {e}")

    if history is not None:
        history.append(("user", json_prompt))
        history.append(("model", raw))

    parsed = parse_json_response(raw)
    return json_prompt, raw, thoughts, parsed


# -------------------------------------------------------------------
# Updated experiment functions
# -------------------------------------------------------------------

def run_independent_sampling(llm, examples: List[Dict[str, str]]):
    """
    Independent sampling:
      - Free-flow question is asked once (no history).
      - For EACH example:
          * Ask MCQ (no history).
          * Ask JSON (no history).
      - No chat history is used anywhere.
    """
    prompt_gen = PromptGenerator()

    # 1) Free-flow criteria ONCE, with no history
    print("\n==============================")
    print("Free-flow criteria (no history)")
    print("==============================\n")

    try:
        free_prompt, free_raw, free_thoughts = call_free_flow_once(
            llm, prompt_gen, history=None
        )
    except Exception as e:
        print(f"Error during free-flow call: {e}")
        return

    print("Free-flow prompt:\n", free_prompt)
    print("Model response (free-flow):\n", free_raw)
    print("Thoughts:", free_thoughts)
    print()

    # 2) For each example, MCQ + JSON with NO history
    for idx, employees in enumerate(examples, start=1):
        print("\n========================================")
        print(f"Example {idx} – Independent MCQ + JSON")
        print("========================================\n")

        # MCQ (no history)
        try:
            mcq_prompt, mcq_raw, mcq_thoughts, mcq_parsed = call_mcq_once(
                llm, prompt_gen, employees, history=None
            )
        except Exception as e:
            print(f"Error during MCQ call for example {idx}: {e}")
            continue
        print("MCQ prompt:\n", mcq_prompt)
        print("MCQ raw response:", mcq_raw)
        print("MCQ thoughts:", mcq_thoughts)
        print("MCQ parsed answer:", mcq_parsed)
        print()

        # JSON (no history)
        try:
            json_prompt, json_raw, json_thoughts, json_parsed = call_json_once(
                llm, prompt_gen, employees, history=None
            )
        except Exception as e:
            print(f"Error during JSON call for example {idx}: {e}")
            continue
        print("JSON prompt:\n", json_prompt)
        print("JSON raw response:", json_raw)
        print("JSON thoughts:", json_thoughts)
        print("JSON parsed object:", json.dumps(json_parsed, indent=2))
        print()


def run_sampling_with_memory(llm, examples: List[Dict[str, str]], num_participants: int = 3):
    """
    Sampling with memory:
      - Simulate `num_participants` users.
      - Each participant has their own conversation history.
      - For EACH example, for EACH participant:
          * Ask MCQ (with that participant's history).
          * Ask free-flow (with history).
          * Ask JSON (with history).
      - The history accumulates across BOTH examples and questions.
    """
    prompt_gen = PromptGenerator()

    participants = [
        SimulatedParticipant(id=f"worker_{i+1}") for i in range(num_participants)
    ]

    for participant in participants:
        print("\n########################################")
        print(f"# Conversation with {participant.id}")
        print("########################################\n")

        for idx, employees in enumerate(examples, start=1):
            print(f"\n----------- {participant.id} – Example {idx} -----------\n")

            # Q1: MCQ with memory
            try:
                mcq_prompt, mcq_raw, mcq_thoughts, mcq_parsed = call_mcq_once(
                    llm, prompt_gen, employees, history=participant.history
                )
            except Exception as e:
                print(f"Error during MCQ call for {participant.id}, example {idx}: {e}")
                continue
            print("[Q1] MCQ prompt:\n", mcq_prompt)
            print("[Q1] MCQ raw:", mcq_raw)
            print("[Q1] MCQ thoughts:", mcq_thoughts)
            print("[Q1] MCQ parsed:", mcq_parsed)
            print()

            # Q2: Free-flow with memory
            try:
                free_prompt, free_raw, free_thoughts = call_free_flow_once(
                    llm, prompt_gen, history=participant.history
                )
            except Exception as e:
                print(f"Error during free-flow call for {participant.id}, example {idx}: {e}")
                continue
            print("[Q2] Free-flow prompt:\n", free_prompt)
            print("[Q2] Free-flow answer (truncated):", free_raw[:200], "...")
            print("[Q2] Free-flow thoughts:", free_thoughts)
            print()

            # Q3: JSON with full accumulated history
            try:
                json_prompt, json_raw, json_thoughts, json_parsed = call_json_once(
                    llm, prompt_gen, employees, history=participant.history
                )
            except Exception as e:
                print(f"Error during JSON call for {participant.id}, example {idx}: {e}")
                continue
            print("[Q3] JSON prompt:\n", json_prompt)
            print("[Q3] JSON raw:", json_raw)
            print("[Q3] JSON parsed:", json.dumps(json_parsed, indent=2))
            print("[Q3] JSON thoughts:", json_thoughts)
            print()


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------

if __name__ == "__main__":
    # ----------------------------------------------------------------
    # Load all examples from CSV
    # ----------------------------------------------------------------
    EXAMPLES_CSV = "employee_comparison_examples.csv"  # adjust path if needed
    all_examples = load_employee_examples(EXAMPLES_CSV)

    # Variable specifying how many rows/examples to consider
    NUM_EXAMPLES_TO_RUN = 5  # <-- change this for your experiments

    # Slice the first N examples (you could random.sample instead if you prefer)
    examples_to_use = all_examples[:NUM_EXAMPLES_TO_RUN]

    # ----------------------------------------------------------------
    # GEMINI RUN (optional)
    # ----------------------------------------------------------------
    config = GeminiConfig(
        model_name="gemini-2.5-flash",  # or another Gemini model
        temperature=0.4,
        max_output_tokens=None,
        thinking_budget=1000,
        include_thoughts=True,
    )

    file_path = "Auth_keys/gemini.txt"

    with open(file_path, "r") as file:
        GEMINI_KEY = file.read().strip()

    gemini_llm = GeminiLLM(config=config, api_key=GEMINI_KEY)

    print("\n================ GEMINI: Independent sampling ================\n")
    run_independent_sampling(gemini_llm, examples_to_use)

    # print("\n================ GEMINI: Sampling with memory ================\n")
    # run_sampling_with_memory(gemini_llm, examples_to_use, num_participants=2)

    # ----------------------------------------------------------------
    # GROQ RUN (optional)
    # ----------------------------------------------------------------
    config = GroqConfig(
        model_name="qwen/qwen3-32b",
        temperature=0.4,
        max_completion_tokens=1024,
        reasoning_format="parsed",
        reasoning_effort=None,
        # include_reasoning=True,  # if you prefer that mode
    )

    file_path = "Auth_keys/groq.txt"

    with open(file_path, "r") as file:
        GROQ_KEY = file.read().strip()

    groq_llm = GroqLLM(config, api_key=GROQ_KEY)

    # print("\n================ GROQ: Independent sampling ==================\n")
    # run_independent_sampling(groq_llm, examples_to_use)

    # print("\n================ GROQ: Sampling with memory ==================\n")
    # run_sampling_with_memory(groq_llm, examples_to_use, num_participants=2)
