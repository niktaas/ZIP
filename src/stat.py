from pathlib import Path

import pandas as pd
from scipy import stats


def compute_significance(
    zip_details_file: str,
    original_disagreements_file: str,
    output_dir: str,
    alpha: float = 0.05,
) -> pd.DataFrame:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    details = pd.read_csv(zip_details_file)
    original_df = pd.read_csv(original_disagreements_file)

    original_disagreements = original_df["original_disagreement"].dropna().tolist()

    rows = []

    for token_index, group in details.groupby("token_index"):
        token = group["token"].iloc[0]
        perturbed_disagreements = group["disagreement"].dropna().tolist()

        if len(perturbed_disagreements) == 0:
            statistic = None
            p_value = None
            significant = False
        elif len(original_disagreements) == 0:
            statistic = None
            p_value = None
            significant = False
        else:
            statistic, p_value = stats.ranksums(
                perturbed_disagreements,
                original_disagreements,
            )
            significant = p_value < alpha

        rows.append({
            "token_index": int(token_index),
            "token": token,
            "mean_perturbed_disagreement": (
                sum(perturbed_disagreements) / len(perturbed_disagreements)
                if perturbed_disagreements else 0
            ),
            "mean_original_disagreement": (
                sum(original_disagreements) / len(original_disagreements)
                if original_disagreements else 0
            ),
            "statistic": statistic,
            "p_value": p_value,
            "significant": significant,
        })

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(output_dir / "stats_summary.csv", index=False)

    significant_tokens = stats_df.loc[
        stats_df["significant"] == True,
        "token"
    ].tolist()

    with open(output_dir / "significant_tokens.txt", "w") as f:
        for token in significant_tokens:
            f.write(f"{token}\n")

    return stats_df