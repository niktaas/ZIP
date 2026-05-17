import os
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow_hub as hub
from openai import OpenAI


embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def get_openai_client(api_key: Optional[str] = None) -> OpenAI:
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY in the notebook or pass api_key.")
    return OpenAI(api_key=api_key)


def query_gpt_4_chat(
    prompt: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.5,
    max_retries: int = 3,
) -> str:
    """
    Send a query to GPT and return the response.
    """
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


def clean_text(text: str) -> str:
    text = str(text).strip()
    text = text.strip('"').strip("'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_altered_sentences_and_replacements(output: str) -> List[Dict[str, str]]:
    parsed = []

    chunks = re.split(r"\d+\.", str(output))

    for chunk in chunks:
        if "Replaced word:" not in chunk:
            continue

        sentence_part = chunk.split("Replaced word:")[0]
        replaced_part = chunk.split("Replaced word:")[1]

        altered_sentence = clean_text(sentence_part)
        replaced_word = clean_text(replaced_part)

        # In case GPT adds extra text after the replaced word
        replaced_word = replaced_word.split("\n")[0].strip()
        replaced_word = replaced_word.strip(".").strip()

        if altered_sentence and replaced_word:
            parsed.append({
                "altered_sentence": altered_sentence,
                "replaced_word": replaced_word,
            })

    return parsed


def deduplicate_keep_order(values: List[str]) -> List[str]:
    seen = set()
    cleaned = []

    for value in values:
        value = clean_text(value)

        if not value:
            continue
        if value.lower() in {"none", "nan", "null"}:
            continue
        if value in seen:
            continue

        seen.add(value)
        cleaned.append(value)

    return cleaned


def perturb_by_deletion(
    prompt: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> List[Tuple[int, str, List[str]]]:

    tokens = prompt.split()
    deleted_sentences = []

    for i, token in enumerate(tokens):
        removed = " ".join(tokens[:i] + tokens[i + 1:])

        removed_prompt = f"Is this sentence meaningful and grammatically correct? \"{removed}\" Answer only with Yes or NO."

        output = query_gpt_4_chat(
            removed_prompt,
            api_key=api_key,
            model=model,
            temperature=0.0,
        )

        deleted_sentences.append(
            (i, token, [removed] if "Yes" in str(output) else [])
        )

    return deleted_sentences

def generate_synonyms(
    prompt: str,
    n_per_method: int = 1,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> List[Tuple[int, str, List[Dict[str, str]]]]:

    tokens = prompt.split()
    accepted_synonyms = []

    for token_index, token in enumerate(tokens):

        altering_prompt = """
        Example1:
        Original Sentence: " Let’s think about it step-by-step. "
        Target word: " think "
        Task: Please provide""" + str(n_per_method) + """different meaningful alterations of the original sentence, each time replacing the word "think" with a different synonym. Ensure that the rest of the sentence remains unchanged and you only change the target word. Write down the altered sentence and the replaced word as the output.
        Output:
        1. Let’s "wonder" about it step-by-step. Replaced word: "wonder"
        2. Let's "ponder" about it step-by-step. Replaced word: "ponder"
        3. Let's "reason" about it step by step. Replaced word: "reason"

        Example2:

        Original Sentence: " Let’s think about it step-by-step. "
        Target word: " Let’s "
        Task: Please provide""" + str(n_per_method) + """different meaningful alterations of the original sentence, each time replacing the word "Let’s" with a different synonym. Ensure that the rest of the sentence remains unchanged. Write down the altered sentence and the replaced word as the output.
        Output:
        1. "We should" think about it step-by-step. Replaced word: "We should"
        2. "Allow us to" think about it step-by-step. Replaced word: "Allow us to"
        3. "Advise us to" think about it step-by-step. Replaced word: "Advise us to"

        Example3:

        Original Sentence: " Let’s think about it step-by-step. "
        Target word: " it "
        Task: Please provide""" + str(n_per_method) + """different meaningful alterations of the original sentence, each time replacing the word "it" with a different synonym. Ensure that the rest of the sentence remains unchanged. Write down the altered sentence and the replaced word as the output.
        Output:
        1. Let’s think about "this" step-by-step. Replaced word: "this"
        2. Let’s think about "that" step-by-step. Replaced word: "that"
        3. Let’s think about "them" step-by-step. Replaced word: "them"

        Input:
        Original Sentence: """ + """ " """ + prompt + """ " """ + """Target word:""" + """ " """ + str(token) + """ " """  + """Task: Please provide""" + str(n_per_method) + """different meaningful alterations of the original sentence, each time replacing the word""" + """ " """ + token + """ " """ + """with a different synonym. Ensure that the rest of the sentence remains unchanged. Write down the altered sentence and the replaced word as the output.
        What is the output?
        """

        output = query_gpt_4_chat(
            altering_prompt,
            api_key=api_key,
            model=model,
            temperature=0.5,
        )

        parsed = parse_altered_sentences_and_replacements(output)
        accepted_synonyms.append((token_index, token, parsed[:n_per_method]))

    return accepted_synonyms


def generate_cohyponyms(
    prompt: str,
    n_per_method: int = 1,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> List[Tuple[int, str, List[Dict[str, str]]]]:

    tokens = prompt.split()
    accepted_cohyponyms = []

    for token_index, token in enumerate(tokens):

        altering_prompt = f"""
        Example:
        Original Sentence: " Let’s think about it step-by-step. "
        Target word: " think "
        Task: Please provide""" + str(n_per_method) + """different meaningful co-hyponyms of the original sentence, each time replacing the word "think" with a different co-hyponym. Ensure that the rest of the sentence remains unchanged and you only change the target word. Write down the altered sentence and the replaced word as the output.
        Output:
        1. Let’s "accept" about it step-by-step. Replaced word: "accept"
        2. Let's "fail" about it step-by-step. Replaced word: "fail"
        3. Let's "disapprove" about it step by step. Replaced word: "disapprove"
        4. Let's "measure" about it step by step. Replaced word: "measure"

        Input:
        Original Sentence: """ + """ " """ + prompt + """ " """ + """Target word:""" + """ " """ + str(token) + """ " """  + """Task: Please provide""" + str(n_per_method) + """different meaningful co-hyponyms of the original sentence, each time replacing the word""" + """ " """ + token + """ " """ + """with a different co-hyponym. Ensure that the rest of the sentence remains unchanged. Write down the altered sentence and the replaced word as the output.
        What is the output?
        """

        output = query_gpt_4_chat(
            altering_prompt,
            api_key=api_key,
            model=model,
            temperature=0.5,
        )

        parsed = parse_altered_sentences_and_replacements(output)
        accepted_cohyponyms.append((token_index, token, parsed[:n_per_method]))

    return accepted_cohyponyms


def validate_alternatives(
    accepted_alternatives: List[Tuple[int, str, List[Dict[str, str]]]],
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    similarity_threshold: float = 0.30,
) -> List[Tuple[int, str, List[str]]]:

    validated_alternatives = []

    for token_index, token, alternatives in accepted_alternatives:
        validated = []

        for alt_dict in alternatives:
            altered_sentence = clean_text(alt_dict["altered_sentence"])
            replaced_word = clean_text(alt_dict["replaced_word"])

            phrase_embeddings = embed([token, replaced_word])
            similarity = np.inner(phrase_embeddings[0], phrase_embeddings[1])

            if similarity >= similarity_threshold:
                validation_prompt = f"Is this sentence meaningful and grammatically correct? \"{altered_sentence}\" Answer only with Yes or NO."

                validation_response = query_gpt_4_chat(
                    validation_prompt,
                    api_key=api_key,
                    model=model,
                    temperature=0.0,
                )

                if "yes" in validation_response.lower():
                    validated.append(altered_sentence)

        validated_alternatives.append(
            (token_index, token, deduplicate_keep_order(validated))
        )

    return validated_alternatives


def save_perturbations(
    prompt: Optional[str] = None,
    output_path: str = "perturbations.csv",
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    n_per_method: int = 1,
    validate: bool = True,
    zeroshot_prompt: Optional[str] = None,
    similarity_threshold: float = 0.30,
) -> pd.DataFrame:

    if prompt is None:
        prompt = zeroshot_prompt

    if prompt is None:
        raise ValueError("Please pass prompt=... from the notebook.")

    if n_per_method < 1:
        raise ValueError("n_per_method must be at least 1.")

    tokens = prompt.split()

    # 1. Deletion
    deletion_rows = perturb_by_deletion(
        prompt=prompt,
        api_key=api_key,
        model=model,
    )

    deletion_map = {
        token_index: alternatives[:1]
        for token_index, token, alternatives in deletion_rows
    }

    # 2. Synonyms
    synonym_rows = generate_synonyms(
        prompt=prompt,
        n_per_method=n_per_method,
        api_key=api_key,
        model=model,
    )

    if validate:
        synonym_rows = validate_alternatives(
            synonym_rows,
            api_key=api_key,
            model=model,
            similarity_threshold=similarity_threshold,
        )
    else:
        synonym_rows = [
            (
                token_index,
                token,
                [x["altered_sentence"] for x in alternatives],
            )
            for token_index, token, alternatives in synonym_rows
        ]

    synonym_map = {
        token_index: alternatives[:n_per_method]
        for token_index, token, alternatives in synonym_rows
    }

    # 3. Co-hyponyms
    cohyponym_rows = generate_cohyponyms(
        prompt=prompt,
        n_per_method=n_per_method,
        api_key=api_key,
        model=model,
    )

    if validate:
        cohyponym_rows = validate_alternatives(
            cohyponym_rows,
            api_key=api_key,
            model=model,
            similarity_threshold=similarity_threshold,
        )
    else:
        cohyponym_rows = [
            (
                token_index,
                token,
                [x["altered_sentence"] for x in alternatives],
            )
            for token_index, token, alternatives in cohyponym_rows
        ]

    cohyponym_map = {
        token_index: alternatives[:n_per_method]
        for token_index, token, alternatives in cohyponym_rows
    }

    # 4. Combine
    rows = []

    for token_index, token in enumerate(tokens):
        alternatives = []

        alternatives.extend(deletion_map.get(token_index, []))
        alternatives.extend(synonym_map.get(token_index, []))
        alternatives.extend(cohyponym_map.get(token_index, []))

        alternatives = deduplicate_keep_order(alternatives)

        if not alternatives:
            continue

        row = {
            "token_index": token_index,
            "token": token,
        }

        for alt_index, alt in enumerate(alternatives):
            row[str(alt_index)] = alt

        rows.append(row)

    perturbations_df = pd.DataFrame(rows)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    perturbations_df.to_csv(output_path, index=False)

    return perturbations_df