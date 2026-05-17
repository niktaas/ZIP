import os
import re
from pathlib import Path

import pandas as pd


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


def ensure_final_answer_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "final_answer" not in df.columns:
        if "gpt_answers" in df.columns:
            df["final_answer"] = [parse_final_answer(x) for x in df["gpt_answers"]]
        elif "0" in df.columns:
            df["final_answer"] = [parse_final_answer(x) for x in df["0"]]
        else:
            raise ValueError("No final_answer, gpt_answers, or 0 column found.")

    return df


def disagreement_score(answers1, answers2) -> int:
    return sum(a != b for a, b in zip(answers1, answers2))


def load_original_samples(samples_dir: str, repetitions: int) -> list[pd.DataFrame]:
    samples = []

    for rep in range(repetitions):
        file_path = Path(samples_dir) / f"sample_df150_{rep}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing original sample file: {file_path}")

        df = pd.read_csv(file_path)
        df = ensure_final_answer_column(df)
        samples.append(df)

    return samples


def get_alternative_columns(perturbations: pd.DataFrame) -> list[str]:
    return [
        col for col in perturbations.columns
        if col not in {"token_index", "token"}
    ]


def valid_perturbation_value(value) -> bool:
    if pd.isna(value):
        return False
    value = str(value).strip()
    if not value:
        return False
    if value.lower() in {"none", "nan", "null"}:
        return False
    return True


def compute_original_disagreements(samples: list[pd.DataFrame]) -> list[int]:
    base = samples[0]["final_answer"].tolist()
    disagreements = []

    for sample in samples[1:]:
        disagreements.append(
            disagreement_score(base, sample["final_answer"].tolist())
        )

    return disagreements


def compute_zip_scores(
    samples_dir: str,
    perturbed_dir: str,
    perturbations_file: str,
    output_dir: str,
    repetitions: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    perturbations = pd.read_csv(perturbations_file)
    samples = load_original_samples(samples_dir, repetitions=repetitions)

    base_answers = samples[0]["final_answer"].tolist()
    n_examples = len(base_answers)

    alt_cols = get_alternative_columns(perturbations)

    score_rows = []
    detail_rows = []

    for _, row in perturbations.iterrows():
        token_index = int(row["token_index"])
        token = str(row["token"])

        disagreements = []

        for alt_col in alt_cols:
            perturbation = row.get(alt_col)

            if not valid_perturbation_value(perturbation):
                continue

            file_path = Path(perturbed_dir) / f"dataframe_T_{token_index}_A_{alt_col}.csv"

            if not file_path.exists():
                print(f"Missing perturbed output: {file_path}")
                continue

            df = pd.read_csv(file_path)
            df = ensure_final_answer_column(df)

            dis = disagreement_score(base_answers, df["final_answer"].tolist())
            disagreements.append(dis)

            detail_rows.append({
                "token_index": token_index,
                "token": token,
                "alternative_index": alt_col,
                "perturbation": perturbation,
                "disagreement": dis,
                "disagreement_percent": (dis / n_examples) * 100,
            })

        if disagreements:
            zip_score = (sum(disagreements) / len(disagreements) / n_examples) * 100
        else:
            zip_score = 0.0

        score_rows.append({
            "token_index": token_index,
            "token": token,
            "zip_score": zip_score,
            "n_alternatives": len(disagreements),
        })

    scores_df = pd.DataFrame(score_rows)
    details_df = pd.DataFrame(detail_rows)

    scores_df.to_csv(output_dir / "zip_scores.csv", index=False)
    details_df.to_csv(output_dir / "zip_details.csv", index=False)

    original_disagreements = compute_original_disagreements(samples)
    pd.DataFrame({
        "original_disagreement": original_disagreements
    }).to_csv(output_dir / "original_disagreements.csv", index=False)

    return scores_df, details_df