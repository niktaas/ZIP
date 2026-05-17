import os
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from openai import OpenAI
from tqdm.auto import tqdm


def get_openai_client(api_key: Optional[str] = None) -> OpenAI:
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY in the notebook or pass api_key.")
    return OpenAI(api_key=api_key)


def query_gpt_chat(
    prompt: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.5,
    max_retries: int = 3,
) -> str:
    client = get_openai_client(api_key)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2 * (attempt + 1))


def format_options(options: str) -> str:
    options = str(options)

    replacements = {
        "'": "",
        "[": "",
        "]": "",
        "A)": "[A] ",
        "B)": "[B] ",
        "C)": "[C] ",
        "D)": "[D] ",
        "E)": "[E] ",
    }

    for old, new in replacements.items():
        options = options.replace(old, new)

    return options


def build_aqua_prompt(question: str, options: str, zeroshot_prompt: str) -> str:
    return (
        f"Multiple-Choice Question: {question}\n"
        f"Answer Choices: {format_options(options)}\n\n"
        f"{zeroshot_prompt}\n\n"
        'Write down your final answer in the format: "The correct answer is [X].", '
        "where X is the letter of the correct answer choice (A, B, C, D, or E)."
    )


def parse_final_answer(text: str) -> str:
    if not isinstance(text, str):
        return "N"

    text = text.upper()

    patterns = [
        r"THE CORRECT ANSWER IS\s*\[?([A-E])\]?",
        r"CORRECT ANSWER\s*(?:IS|:)?\s*\[?([A-E])\]?",
        r"ANSWER\s*(?:IS|:)?\s*\[?([A-E])\]?",
        r"\[([A-E])\]",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)

    return "N"


def generate_answers(
    api_key: Optional[str],
    zeroshot_prompt: str,
    sample_file: str,
    output_dir: str,
    repetitions: int = 3,
    model: str = "gpt-4o-mini",
    temperature: float = 0.5,
) -> list[pd.DataFrame]:

    sample_df = pd.read_csv(sample_file)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_outputs = []

    for rep in tqdm(range(repetitions), desc="Original prompt repetitions"):
        df = sample_df.copy()
        answers = []

        question_iterator = tqdm(
            df.iterrows(),
            total=len(df),
            desc=f"Repetition {rep + 1}/{repetitions}",
            leave=False,
        )

        for _, row in question_iterator:
            prompt = build_aqua_prompt(
                question=row["question"],
                options=row["options"],
                zeroshot_prompt=zeroshot_prompt,
            )

            ans = query_gpt_chat(
                prompt=prompt,
                api_key=api_key,
                model=model,
                temperature=temperature,
            )
            answers.append(ans)

        df["gpt_answers"] = answers
        df["final_answer"] = [parse_final_answer(ans) for ans in answers]

        output_file = output_dir / f"sample_df150_{rep}.csv"
        df.to_csv(output_file, index=False)

        all_outputs.append(df)

    return all_outputs