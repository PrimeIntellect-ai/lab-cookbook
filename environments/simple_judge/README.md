# simple-judge

Toy single-turn environment for learning LLM judge wiring. Each task asks for a short response; an LLM judge checks one plain-language criterion stored in `info`.

## Required Environment Variables

- `PRIME_API_KEY` — judge model API key (default `judge_api_key_var`)

## Usage

```bash
prime env install simple-judge
prime eval run simple-judge -m openai/gpt-4.1-mini -n 6 -r 2
```
