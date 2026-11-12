
import argparse
import re
import json
import os
import urllib.error
import urllib.request
from datasets import load_dataset
import transformers


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate TRT on AIME 2025 using Ollama."
    )

    parser.add_argument("--model_name", type=str, default="qwen3:4b-thinking-2507-q4_K_M")
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--ollama_url", type=str, default="http://localhost:11434")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--num_ctx", type=int, default=8192)
    parser.add_argument("--request_timeout", type=int, default=3600)
    parser.add_argument("--split", type=str, default="train", choices=["train"])
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--demo_size", type=int, default=0)
    parser.add_argument("--reflex_size", type=int, default=64)

    return parser.parse_args()


def get_last_integer(text):
    i = len(text) - 1

    while i >= 0:
        if text[i].isdigit():
            end = i

            while i >= 0 and text[i].isdigit():
                i -= 1

            if i >= 0 and text[i] == "-":
                return text[i:end + 1]

            return text[i + 1:end + 1]

        i -= 1

    return None


def extract_answer_integer(generated_text):
    matches = re.findall(
        r"\\boxed\{(-?\d+)\}",
        generated_text,
        flags=re.IGNORECASE
    )

    if matches:
        try:
            return int(matches[-1])
        except ValueError:
            pass

    last_int = get_last_integer(generated_text)

    if last_int is not None:
        try:
            return int(last_int)
        except ValueError:
            pass

    return None


def remove_first_colon(text):
    colon_index = text.find(":")

    if colon_index != -1:
        return text[:colon_index] + text[colon_index + 1:].lstrip()

    return text


def generate_with_ollama(prompt, args):
    request_body = {
        "model": args.model_name,
        "prompt": prompt,
        "raw": True,
        "stream": False,
        "options": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "num_predict": int(0.9 * args.max_new_tokens),
            "num_ctx": args.num_ctx
        }
    }

    request = urllib.request.Request(
        args.ollama_url.rstrip("/") + "/api/generate",
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=args.request_timeout
        ) as response:
            response_data = json.loads(
                response.read().decode("utf-8")
            )

    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Ollama generation request failed: {exc}"
        ) from exc

    generation = response_data.get("response", "")
    thinking = response_data.get("thinking", "")

    if thinking and "</think>" not in generation:
        generation = "<think>" + thinking + "</think>" + generation

    if not generation.strip():
        raise RuntimeError("Ollama returned an empty generation.")

    return generation


def save_results(path, results):
    temp_path = path + ".tmp"

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )
        f.flush()
        os.fsync(f.fileno())

    os.replace(temp_path, path)


def create_initial_state(dataset_size):
    return {
        "conversations": [[] for _ in range(dataset_size)],
        "summaries": [[] for _ in range(dataset_size)],
        "knowledges": [[] for _ in range(dataset_size)],
        "answers_tracker": [[] for _ in range(dataset_size)],
        "drop_decisions": [[] for _ in range(dataset_size)]
    }


def main():
    args = parse_arguments()

    if args.limit < 1:
        raise ValueError("--limit must be at least 1")

    if args.reflex_size < 1:
        raise ValueError("--reflex_size must be at least 1")

    if args.max_new_tokens < 1:
        raise ValueError("--max_new_tokens must be positive")

    dataset = load_dataset("yentinglin/aime_2025")["train"]

    dataset = dataset.remove_columns([
        col for col in dataset.column_names
        if col not in ["problem", "answer"]
    ])

    dataset = dataset.select(
        range(min(args.limit, len(dataset)))
    )

    print(f"Dataset split: {args.split}")
    print(f"Total questions: {len(dataset)}")
    print(f"TRT rounds: {args.reflex_size}")
    print(f"Model: {args.model_name}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        trust_remote_code=True
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))

    results_dir = os.path.abspath(
        os.path.join(
            script_dir,
            "..",
            "results",
            "aime2025_ollama"
        )
    )

    os.makedirs(results_dir, exist_ok=True)

    output_path = os.path.join(results_dir, "results.json")

    config = {
        "model": args.model_name,
        "tokenizer": args.tokenizer_name,
        "dataset": "yentinglin/aime_2025",
        "split": args.split,
        "limit": len(dataset),
        "reflex_size": args.reflex_size,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "num_ctx": args.num_ctx
    }

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        if results["config"] != config:
            raise ValueError(
                "Existing results.json uses a different configuration. "
                "Do not overwrite it. Restore the original settings "
                "or move the old results.json before starting a new run."
            )

        state = results["state"]
        completed_rounds = len(results["rounds"])

        print(f"Resuming from round {completed_rounds + 1}.")

    else:
        state = create_initial_state(len(dataset))

        results = {
            "config": config,
            "rounds": [],
            "state": state
        }

        completed_rounds = 0

        save_results(output_path, results)

        print("Starting a new TRT run.")

    if completed_rounds >= args.reflex_size:
        print("All requested TRT rounds are already completed.")
        return

    conversations = state["conversations"]
    summaries = state["summaries"]
    knowledges = state["knowledges"]
    answers_tracker = state["answers_tracker"]
    drop_decisions = state["drop_decisions"]

    init_prompt = """
\n\nGuideline: Let's solve this problem. Be thorough.

## Output format (Use exact headers including square brackets):
[Summary]: A paragraph of detailed step-by-step summary of your solution, note down every reasoning step and calculation you did, where you might have gone wrong, and what was the final answer you got.
[Answer]: Therefore, final answer is \\boxed{<integer>}.

Let's think step by step. Follow the output format strictly.
"""

    cmd_prompt = """
\n\nLet's solve this problem. Be thorough. Let's solve from scratch first. Try to avoid any heuristics, use solid logic and math. Then check against the Wrong Answer List and the reference solution, see which one is correct.

## Reference Solution
{reference_solution}

{knowledge_text}

### Output format (Use exact headers including square brackets):
[Error Report]: If you get a different solution than your recalled solutions, explain why the old answer is wrong in details here, start with "The Answer cannot be <integer>, because..." end with "The correct approach is ..." without listing out the answer you got, explain in details in a standalone manner (or write "N/A" if you agree with the recalled solution)
[Summary]: A paragraph of detailed step-by-step summary of your solution, write thoroughly and in details, note down every steps of reasoning you did, and what was the final answer you got. Write it in a standalone manner.
[Answer]: Therefore, final answer is \\boxed{<integer>}.
"""

    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{%- if messages[0]['role'] == 'system' %}"
            "{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}"
            "{%- else %}"
            "{{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}"
            "{%- endif %}"
            "{%- for message in messages %}"
            "{%- if message['role'] == 'user' or (message['role'] == 'assistant' and not message.get('tool_calls')) %}"
            "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
            "{%- endif %}"
            "{%- endfor %}"
            "{%- if add_generation_prompt %}"
            "{{- '<|im_start|>assistant\\n<|begin_of_thought|>' }}"
            "{%- endif %}"
        )

    for reflex_ctr in range(completed_rounds, args.reflex_size):

        all_prompts = []
        gold_answers = []

        print(
            f"\n========== TRT ROUND "
            f"{reflex_ctr + 1}/{args.reflex_size} =========="
        )

        for example_ctr, example in enumerate(dataset):

            question = example["problem"]
            gold_answer = example["answer"]

            gold_answers.append(gold_answer)

            if reflex_ctr == 0:

                tmp_conv = [{
                    "role": "user",
                    "content": question + init_prompt
                }]

                conversations[example_ctr].append(tmp_conv[0])

                prompt = tokenizer.apply_chat_template(
                    tmp_conv,
                    tokenize=False,
                    add_generation_prompt=True
                )

            else:

                last_conv = conversations[example_ctr][-1]

                x = last_conv["content"].split("</think>")[-1]

                summary_text = (
                    x.split("[Summary]")[-1]
                    .strip()
                    .split("[Answer]")[0]
                    .strip()
                )

                summary_text = remove_first_colon(summary_text)

                if summary_text:
                    summary_text = (
                        summary_text[0].lower() + summary_text[1:]
                    )

                summaries[example_ctr].append(summary_text)

                this_answer = extract_answer_integer(
                    x.split("[Answer]")[-1].strip()
                )

                if len(answers_tracker[example_ctr]) > 1:

                    previous_answer = answers_tracker[example_ctr][-2]

                    if (
                        this_answer is not None
                        and previous_answer is not None
                        and this_answer != previous_answer
                        and "[Error Report]" in x
                    ):

                        new_knowledge = (
                            x.split("[Error Report]")[-1]
                            .split("[Summary]")[0]
                            .strip()
                        )

                        new_knowledge = remove_first_colon(
                            new_knowledge
                        )

                        if new_knowledge:
                            new_knowledge = (
                                new_knowledge[0].lower()
                                + new_knowledge[1:]
                            )

                            drop_decisions[example_ctr].append(
                                previous_answer
                            )

                            knowledges[example_ctr].append(
                                new_knowledge
                            )

                knowledge_items = []

                for wrong_answer, knowledge in zip(
                    drop_decisions[example_ctr],
                    knowledges[example_ctr]
                ):
                    knowledge_items.append(
                        (wrong_answer, knowledge)
                    )

                wrong_set = {}
                unique_knowledges = []

                for wrong_answer, knowledge in reversed(knowledge_items):

                    if wrong_set.get(wrong_answer, 0) >= 2:
                        continue

                    wrong_set[wrong_answer] = (
                        wrong_set.get(wrong_answer, 0) + 1
                    )

                    unique_knowledges.append(knowledge)

                unique_knowledges.reverse()

                if unique_knowledges:
                    knowledge_text = (
                        "## Wrong Answer List\n"
                        + "\n".join([
                            f"**Wrong Answer {i + 1}**: {item}"
                            for i, item in enumerate(unique_knowledges)
                        ])
                    )
                else:
                    knowledge_text = "## Wrong Answer List\nN/A"

                reference_solution = (
                    "Note: No reference solution available."
                )

                if summaries[example_ctr]:
                    reference_solution = (
                        "Reference Solution "
                        "(check after you first solve independently):\n"
                        + summaries[example_ctr][-1]
                    )

                tmp_conv = [{
                    "role": "user",
                    "content": (
                        question
                        + cmd_prompt.replace(
                            "{knowledge_text}",
                            knowledge_text
                        ).replace(
                            "{reference_solution}",
                            reference_solution
                        )
                    )
                }]

                eps_text = (
                    "This is a complex problem. I sense a mistake "
                    "in the reference solution, I will solve it "
                    "from scratch first without referencing it. "
                    "Then I will check against the Wrong Answer "
                    "List and the Reference Solution to write "
                    "an improved solution."
                )

                prompt = tokenizer.apply_chat_template(
                    tmp_conv,
                    tokenize=False,
                    add_generation_prompt=True
                ) + eps_text

                conversations[example_ctr].append(tmp_conv[0])

            all_prompts.append(prompt)

        generations = []

        for question_idx, prompt in enumerate(all_prompts):

            print(
                f"Generating question "
                f"{question_idx + 1}/{len(all_prompts)}..."
            )

            generation = generate_with_ollama(prompt, args)
            generations.append(generation)

        predictions = []
        predicted_solutions = []
        correct = 0

        for example_ctr, (generation, gold) in enumerate(
            zip(generations, gold_answers)
        ):

            conversations[example_ctr].append({
                "role": "assistant",
                "content": generation
            })

            predicted_solutions.append(generation)

            pred_answer = extract_answer_integer(generation)

            answers_tracker[example_ctr].append(pred_answer)
            predictions.append(pred_answer)

            if pred_answer is not None and int(pred_answer) == int(gold):
                correct += 1

        total = len(gold_answers)
        accuracy = correct / total if total else 0

        print(f"Total questions: {total}")
        print(f"Correct answers: {correct}")
        print(f"Accuracy: {accuracy:.2%}")

        round_result = {
            "round": reflex_ctr + 1,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "average_prompt_length": (
                sum(len(tokenizer.encode(p)) for p in all_prompts)
                / len(all_prompts)
                if all_prompts else 0
            ),
            "predictions": [
                {
                    "question": example["problem"],
                    "gold": gold,
                    "predicted": pred,
                    "pred_solution": solution
                }
                for example, gold, pred, solution in zip(
                    dataset,
                    gold_answers,
                    predictions,
                    predicted_solutions
                )
            ]
        }

        results["rounds"].append(round_result)

        results["state"] = {
            "conversations": conversations,
            "summaries": summaries,
            "knowledges": knowledges,
            "answers_tracker": answers_tracker,
            "drop_decisions": drop_decisions
        }

        save_results(output_path, results)

        print(f"Saved checkpoint: {output_path}")

    print("\nTRT evaluation completed.")
    print(f"Results: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Completed rounds are saved in results.json.")
        raise SystemExit(130)
