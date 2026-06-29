# ZIP Reproducibility Code

This repository provides an implementation of ZIP (Zero-shot Importance of Perturbation), a model-agnostic method for measuring how much each prompt word influences an LLM’s output. By perturbing prompt words and comparing responses, ZIP identifies influential words in zero-shot prompts.


## Repository structure

```text
.
├── data/
│   └── aqua-rat_questions.csv        # small AQUA-RAT sample for the demo
├── notebooks/
│   ├── zip_demo.ipynb                # end-to-end ZIP demo on AQUA-RAT
│   └── validation_prompt.ipynb       # validation prompt experiment using the same ZIP process
├── src/
│   ├── creating_sample.py            # loads/creates the AQUA-RAT sample
│   ├── creating_perturbations.py     # creates deletion, synonym, and co-hyponym perturbations
│   ├── original_reprompting.py       # runs the original prompt multiple times
│   ├── perterbation_results.py       # runs perturbed prompts on the same questions
│   ├── ZIP.py                        # computes ZIP scores for task outputs
│   ├── stat.py                       # runs significance testing
│   └── visualization.py              # creates the word-level ZIP visualization
└── requirements.txt
```

## Setup

Create and activate an environment, then install the requirements:

```bash
pip install -r requirements.txt
```

## OpenAI API key

The demo uses OpenAI models for both perturbation generation and answer generation. Set your key inside the notebook before running the pipeline:

```python
import os
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
```

## Citation

If you use this code, please cite our paper:

```bibtex
@article{sadr2025think,
  title={ZIP: Quantifying Which Words Matter in Zero-Shot Instructional Prompts},
  author={Gohari Sadr, Nikta  and Madhusudan, Sangmitra and Asgari, Arash and Sajjad, Hassan and Seyyed-Kalantari, Laleh and Emami, Ali},
  journal={arXiv preprint arXiv:2502.03418},
  year={2025}
}
