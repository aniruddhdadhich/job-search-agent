# India Job Search Agent (OpenAI + Playwright)

This repo automates India-focused job discovery and ranking using OpenAI (ChatGPT paid tiers) and Playwright. It targets Indian job boards such as LinkedIn, Naukri, Instahyre, and Cutshort, then scores results against a candidate profile.

> **Note**: This script focuses on discovery and ranking. It does **not** submit applications by default. You can extend it with per-board application flows, but be mindful of each site's terms of service.

## Features

- Searches multiple Indian job boards with the same query/locations.
- Uses OpenAI to score job relevance and extract matching skills.
- Saves structured results to a timestamped JSON file.
- Supports logged-in sessions using Playwright storage state (required for LinkedIn).

## Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) installed (`pipx install uv` or see uv docs)
- An OpenAI API key (paid ChatGPT plan supported via API billing)
- Playwright browsers installed: `python -m playwright install`

## Setup

1. Create a virtual environment and install dependencies with uv:

```bash
uv venv
source .venv/bin/activate
uv pip install openai playwright
```

2. Create a Playwright storage state after logging in to job boards (especially LinkedIn):

```bash
python -m playwright codegen --save-storage=storage_state.json
```

3. Copy and edit the example config:

```bash
cp config.example.json config.json
```

4. Export your OpenAI API key:

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini
```

## Run

```bash
python job_search_agent.py --config config.json
```

Results are saved in `./outputs` with a timestamp.

## Notes on Reliability

- **LinkedIn / Naukri require login**. Use storage state to avoid expiring sessions.
- If a board changes DOM structure, update selectors in the relevant `JobBoard` class.
- Keep API usage bounded by lowering `max_results_per_board` and `max_applications`.

## Extending to Applications

The current script is safe by default and does not submit applications. To add application automation:

1. Add a board-specific `apply()` method that navigates to `job.url`.
2. Use OpenAI to draft answers and fill form fields.
3. Keep `dry_run` enabled until you are confident in the flow.

## Legal / Compliance

Always review the terms of each job board and comply with their automation policies.
