# Patent Search

This folder contains a small family of patent-focused RL environments built with Verifiers.

The environments train search, retrieval, and technical reasoning over patent-style documents.
Each package uses the v1 `Taskset` plus default `Harness` pattern: the taskset owns dataset loading, Chroma-backed patent tools, system prompt, and LLM-judge rewards.

## Environments in this folder

1. **Basic Patent Q&A**
   - Path: `cookbook/recipes/patent_search/basic_patent_q_and_a/`
   - Package id: `prime/basic-patent-q-and-a`
   - Focus: straightforward patent question answering with agentic retrieval

2. **Advanced Patent Q&A**
   - Path: `cookbook/recipes/patent_search/advanced_patent_q_and_a/`
   - Package id: `prime/advanced-patent-q-and-a`
   - Focus: harder multi-step retrieval and comparison-style patent questions

3. **Patent Technical Analysis**
   - Path: `cookbook/recipes/patent_search/patent_technical_analysis/`
   - Package id: `prime/patent-technical-analysis`
   - Focus: technical analysis of claims, innovations, and differentiators

## Quick start

Evaluate published environments:

```bash
prime eval run primeintellect/basic-patent-q-and-a --model openai/gpt-oss-120b
prime eval run primeintellect/advanced-patent-q-and-a --model openai/gpt-oss-120b
prime eval run primeintellect/patent-technical-analysis --model openai/gpt-oss-120b
```

Train from the included configs:

```bash
prime rl run cookbook/recipes/patent_search/basic_patent_q_and_a/config.toml
prime rl run cookbook/recipes/patent_search/advanced_patent_q_and_a/config.toml
prime rl run cookbook/recipes/patent_search/patent_technical_analysis/config.toml
```
