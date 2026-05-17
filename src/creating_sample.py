from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import load_dataset


def create_aqua_sample(
    output_path: str,
    sample_size: int = 5,
    local_sample_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create a small AQUA-RAT sample.

    If local_sample_path is given, load from that file.
    Otherwise, download deepmind/aqua_rat from Hugging Face.
    """
    if local_sample_path is not None and Path(local_sample_path).exists():
        df = pd.read_csv(local_sample_path)
    else:
        ds = load_dataset("deepmind/aqua_rat", "raw")

        if "validation" in ds:
            split = ds["validation"]
        elif "test" in ds:
            split = ds["test"]
        else:
            split = ds[list(ds.keys())[0]]

        df = pd.DataFrame(split)

    required_cols = {"question", "options"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in AQUA-RAT data: {missing}")

    sample_df = df.head(sample_size).copy()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(output_path, index=False)

    return sample_df