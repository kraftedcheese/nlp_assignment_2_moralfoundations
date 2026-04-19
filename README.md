# Simulating Persona-Based Moral Foundation Annotation with LLMs

Can an LLM annotate moral content the way a human would, and does giving it a
demographic persona make it more human-like? This project investigates those
questions using **Llama-3-8B** and the
[Moral Foundations Reddit Corpus (MFRC)](https://arxiv.org/abs/2208.05545).

## Overview

We study LLM annotation behavior across three moral foundations: **Care**,
**Authority**, and **Purity**, across three research questions:

- **RQ1:** How well does the LLM agree with human annotators overall?
- **RQ2:** Does sociodemographic persona prompting (Christian/Atheist,
  Conservative/Progressive) shift annotation behavior?
- **RQ3:** Does giving the LLM an individual annotator's demographic profile
  bring it closer to that person's annotations?

## Key Findings

- The LLM agrees with humans only slightly better than chance overall
  (MASI α ≈ 0.07), but performs meaningfully better on **Care** (α = 0.31)
  and **Non-Moral** (α = 0.13).
- The LLM is **more sensitive to political than religious framing**, contrary
  to theoretical predictions. Care was the most polarized foundation between
  Conservative and Progressive personas.
- Providing an individual annotator's full demographic profile nudges the LLM
  in the right direction, but **none of the human-LLM pairs reached
  inter-human agreement** (α = 0.41).
- For Purity, the LLM appears to rely on surface-level lexical cues (e.g.
  "corrupt", "creepy") rather than genuine semantic understanding.

## Dataset

We use the [MFRC](https://arxiv.org/abs/2208.05545) (17,457 Reddit comments,
annotated by 4 trained human annotators across 6 moral foundations). Analysis
subsets:

| Experiment | N comments |
|---|---|
| RQ1 baseline | 12,963 |
| RQ2 persona prompting | 1,000 (stratified) |
| RQ3 individual profiles | 55 (manually annotated by authors) |

## Methodology

- **Model:** Llama-3-8B with few-shot prompting based on the MFRC annotation guide
- **Metric:** Krippendorff's α (MASI distance for multi-label sets; binary
  nominal α per foundation)
- **Persona prompting:** Sentence-level system prompt injection for 4
  sociodemographic personas; full profiles for 2 individual annotators

## Repository Structure
├── data/ # MFRC dataset files
├── prompts/ # Prompt templates used for annotation
├── scripts/ # Inference and evaluation code
├── results/ # Output annotations and α scores
└── report/ # Full paper (LaTeX source)
