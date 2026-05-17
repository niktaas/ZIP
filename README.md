# ZIP (Zero-shot Importance of Perturbation)

This repository provides an implementation of ZIP, a model-agnostic method for measuring how much each word in a prompt influences an LLM's output. By perturbing prompt words and comparing responses, ZIP helps analyze zero-shot instructional prompts and identify influential words. The picture below demonstrates the ZIP score generation process for the Chain-of-Thought prompt using GPT-4o on a AQUA-RAT dataset instance. Perturbed prompts are compared to the original, generating word-level ZIP scores. The red box highlights
“step-by-step” as significantly important based on statistical analysis.

<p align="center">
  <img src="overview.png" width="500" />
</p>



## Repository structure

```text
data/
aqua-rat_questions.csv       # small AQUA-RAT sample for the demo
notebooks/
zip_demo.ipynb                 # end-to-end ZIP demo
src/
creating_sample.py             # loads/creates the AQUA-RAT sample
creating_perturbations.py      # creates deletion, synonym, and co-hyponym perturbations
original_reprompting.py        # runs the original prompt multiple times
perterbation_results.py        # runs perturbed prompts on the same questions
ZIP.py                         # computes ZIP scores
stat.py                        # runs significance testing
visualization.py               # creates the word-level ZIP visualization
requirements.txt
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
  title={Think or Step-by-Step? UnZIPping the Black Box in Zero-Shot Prompts},
  author={Gohari Sadr, Nikta  and Madhusudan, Sangmitra and Emami, Ali},
  journal={arXiv preprint arXiv:2502.03418},
  year={2025}
}
