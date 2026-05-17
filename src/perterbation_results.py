import os
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from openai import OpenAI
from tqdm.auto import tqdm


STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "on", "in", "at", "to",
    "of", "for", "with", "by", "from", "about", "this", "that", "these",
    "those", "is", "are", "was", "were", "be", "been", "being", "it",
}


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


def valid_perturbation_value(value) -> bool:
    if pd.isna(value):
        return False
    value = str(value).strip()
    if not value:
        return False
    if value.lower() in {"none", "nan", "null"}:
        return False
    return True


def get_alternative_columns(perturbations: pd.DataFrame) -> list[str]:
    return [
        col for col in perturbations.columns
        if col not in {"token_index", "token"}
    ]


def generate_answers(
    api_key: Optional[str],
    sample_file: str,
    perturbations_file: str,
    output_dir: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.5,
    skip_stopwords: bool = False,
) -> list[pd.DataFrame]:

    sample_df = pd.read_csv(sample_file)
    perturbations = pd.read_csv(perturbations_file)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    alt_cols = get_alternative_columns(perturbations)

    # Count total valid perturbation runs for the progress bar
    total_runs = 0
    for _, row in perturbations.iterrows():
        token = str(row["token"])

        if skip_stopwords and token.lower().strip(".,!?;:") in STOP_WORDS:
            continue

        for alt_col in alt_cols:
            if valid_perturbation_value(row.get(alt_col)):
                total_runs += 1

    perturbation_iterator = tqdm(
        perturbations.iterrows(),
        total=len(perturbations),
        desc="Perturbed tokens",
    )

    run_bar = tqdm(
        total=total_runs,
        desc="Perturbed prompt variants",
        leave=True,
    )

    for _, row in perturbation_iterator:
        token_index = int(row["token_index"])
        token = str(row["token"])

        if skip_stopwords and token.lower().strip(".,!?;:") in STOP_WORDS:
            continue

        for alt_col in alt_cols:
            perturbation = row.get(alt_col)

            if not valid_perturbation_value(perturbation):
                continue

            perturbation = str(perturbation).strip()

            answers = []
            df_out = pd.DataFrame()

            question_iterator = tqdm(
                sample_df.iterrows(),
                total=len(sample_df),
                desc=f"T{token_index} A{alt_col}",
                leave=False,
            )

            for _, sample_row in question_iterator:
                prompt = build_aqua_prompt(
                    question=sample_row["question"],
                    options=sample_row["options"],
                    zeroshot_prompt=perturbation,
                )

                ans = query_gpt_chat(
                    prompt=prompt,
                    api_key=api_key,
                    model=model,
                    temperature=temperature,
                )
                answers.append(ans)

            df_out["gpt_answers"] = answers
            df_out["final_answer"] = [parse_final_answer(ans) for ans in answers]
            df_out["token_index"] = token_index
            df_out["token"] = token
            df_out["alternative_index"] = alt_col
            df_out["perturbation"] = perturbation

            output_file = output_dir / f"dataframe_T_{token_index}_A_{alt_col}.csv"
            df_out.to_csv(output_file, index=False)

            outputs.append(df_out)

            run_bar.update(1)

    run_bar.close()

    return outputs