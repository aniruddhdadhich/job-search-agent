import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

from openai import AsyncOpenAI
from playwright.async_api import async_playwright


@dataclass
class CandidateProfile:
    name: str
    email: str
    phone: str
    current_location: str
    experience_years: float
    skills: list[str]
    preferred_titles: list[str]
    portfolio_url: str
    linkedin_url: str
    resume_path: str
    cover_letter: str


@dataclass
class JobPosting:
    title: str
    company: str
    location: str
    url: str
    source: str
    description: str | None = None


@dataclass
class SearchConfig:
    search_queries: list[str]
    locations: list[str]
    max_results_per_board: int
    max_applications: int
    dry_run: bool
    headless: bool
    storage_state_path: str


class JobBoard:
    def __init__(self, name: str) -> None:
        self.name = name

    async def search(self, page, query: str, location: str, max_results: int) -> list[JobPosting]:
        raise NotImplementedError


class LinkedInJobBoard(JobBoard):
    def __init__(self) -> None:
        super().__init__("LinkedIn")

    async def search(self, page, query: str, location: str, max_results: int) -> list[JobPosting]:
        search_url = (
            "https://www.linkedin.com/jobs/search/?keywords="
            f"{quote(query)}&location={quote(location)}"
        )
        await page.goto(search_url, wait_until="domcontentloaded")
        await page.wait_for_timeout(2000)

        if "login" in page.url:
            raise RuntimeError("LinkedIn requires login. Provide a storage state file.")

        await page.wait_for_selector(".jobs-search__results-list li", timeout=10000)
        cards = await page.locator(".jobs-search__results-list li").all()

        postings: list[JobPosting] = []
        for card in cards[:max_results]:
            title = await card.locator("h3").first.inner_text()
            company = await card.locator("h4").first.inner_text()
            location_text = await card.locator(".job-search-card__location").first.inner_text()
            link = await card.locator("a").first.get_attribute("href")
            if not link:
                continue
            postings.append(
                JobPosting(
                    title=title.strip(),
                    company=company.strip(),
                    location=location_text.strip(),
                    url=link,
                    source=self.name,
                )
            )
        return postings


class NaukriJobBoard(JobBoard):
    def __init__(self) -> None:
        super().__init__("Naukri")

    async def search(self, page, query: str, location: str, max_results: int) -> list[JobPosting]:
        slug_query = quote(query.replace(" ", "-"))
        slug_location = quote(location.replace(" ", "-"))
        search_url = f"https://www.naukri.com/{slug_query}-jobs-in-{slug_location}"
        await page.goto(search_url, wait_until="domcontentloaded")
        await page.wait_for_timeout(2000)

        await page.wait_for_selector(".list article", timeout=15000)
        cards = await page.locator(".list article").all()

        postings: list[JobPosting] = []
        for card in cards[:max_results]:
            title = await card.locator("a.title").first.inner_text()
            company = await card.locator("a.subTitle").first.inner_text()
            location_text = await card.locator(".locWdth").first.inner_text()
            link = await card.locator("a.title").first.get_attribute("href")
            if not link:
                continue
            postings.append(
                JobPosting(
                    title=title.strip(),
                    company=company.strip(),
                    location=location_text.strip(),
                    url=link,
                    source=self.name,
                )
            )
        return postings


class InstahyreJobBoard(JobBoard):
    def __init__(self) -> None:
        super().__init__("Instahyre")

    async def search(self, page, query: str, location: str, max_results: int) -> list[JobPosting]:
        search_url = (
            "https://www.instahyre.com/search?query="
            f"{quote(query)}&locations={quote(location)}"
        )
        await page.goto(search_url, wait_until="domcontentloaded")
        await page.wait_for_timeout(2000)

        cards = await page.locator(".listing-row").all()
        postings: list[JobPosting] = []
        for card in cards[:max_results]:
            title = await card.locator(".title").first.inner_text()
            company = await card.locator(".company-name").first.inner_text()
            location_text = await card.locator(".locations").first.inner_text()
            link = await card.locator("a").first.get_attribute("href")
            if not link:
                continue
            postings.append(
                JobPosting(
                    title=title.strip(),
                    company=company.strip(),
                    location=location_text.strip(),
                    url=f"https://www.instahyre.com{link}",
                    source=self.name,
                )
            )
        return postings


class CutshortJobBoard(JobBoard):
    def __init__(self) -> None:
        super().__init__("Cutshort")

    async def search(self, page, query: str, location: str, max_results: int) -> list[JobPosting]:
        search_url = (
            "https://cutshort.io/jobs?query="
            f"{quote(query)}&locations={quote(location)}"
        )
        await page.goto(search_url, wait_until="domcontentloaded")
        await page.wait_for_timeout(2000)

        cards = await page.locator("[data-test=job-card]").all()
        postings: list[JobPosting] = []
        for card in cards[:max_results]:
            title = await card.locator("[data-test=job-title]").first.inner_text()
            company = await card.locator("[data-test=company-name]").first.inner_text()
            location_text = await card.locator("[data-test=job-location]").first.inner_text()
            link = await card.locator("a").first.get_attribute("href")
            if not link:
                continue
            postings.append(
                JobPosting(
                    title=title.strip(),
                    company=company.strip(),
                    location=location_text.strip(),
                    url=f"https://cutshort.io{link}",
                    source=self.name,
                )
            )
        return postings


class JobRanker:
    def __init__(self, client: AsyncOpenAI, model: str, candidate: CandidateProfile) -> None:
        self.client = client
        self.model = model
        self.candidate = candidate

    async def score(self, job: JobPosting) -> dict[str, Any]:
        prompt = (
            "You are an assistant ranking job relevance for a candidate. "
            "Return JSON with keys: score (0-100), summary, matching_skills.\n\n"
            f"Candidate:\n{self.candidate}\n\nJob:\n{job}"
        )
        response = await self.client.responses.create(
            model=self.model,
            input=prompt,
            temperature=0.2,
        )
        content = response.output_text.strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"score": 0, "summary": content, "matching_skills": []}


class JobSearchAgent:
    def __init__(self, candidate: CandidateProfile, config: SearchConfig) -> None:
        self.candidate = candidate
        self.config = config
        self.client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.boards: list[JobBoard] = [
            LinkedInJobBoard(),
            NaukriJobBoard(),
            InstahyreJobBoard(),
            CutshortJobBoard(),
        ]

    async def run(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        ranker = JobRanker(self.client, self.model, self.candidate)

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.config.headless)
            context = await browser.new_context(storage_state=self.config.storage_state_path)
            page = await context.new_page()

            for query in self.config.search_queries:
                for location in self.config.locations:
                    for board in self.boards:
                        postings = await board.search(
                            page,
                            query,
                            location,
                            self.config.max_results_per_board,
                        )
                        for job in postings:
                            score_payload = await ranker.score(job)
                            results.append(
                                {
                                    "job": job.__dict__,
                                    "score": score_payload.get("score", 0),
                                    "summary": score_payload.get("summary", ""),
                                    "matching_skills": score_payload.get("matching_skills", []),
                                }
                            )
                            if len(results) >= self.config.max_applications:
                                await browser.close()
                                return results

            await browser.close()

        return results


def load_candidate(path: Path) -> CandidateProfile:
    data = json.loads(path.read_text())
    return CandidateProfile(**data["candidate"])


def load_config(path: Path) -> SearchConfig:
    data = json.loads(path.read_text())
    config = data["search"]
    return SearchConfig(
        search_queries=config["search_queries"],
        locations=config["locations"],
        max_results_per_board=config["max_results_per_board"],
        max_applications=config["max_applications"],
        dry_run=config.get("dry_run", True),
        headless=config.get("headless", True),
        storage_state_path=config["storage_state_path"],
    )


def persist_results(results: Iterable[dict[str, Any]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"job_results_{timestamp}.json"
    output_path.write_text(json.dumps(list(results), indent=2))
    return output_path


async def main() -> None:
    parser = argparse.ArgumentParser(description="India-focused job search automation")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    parser.add_argument("--output", default="./outputs", help="Directory for results")
    args = parser.parse_args()

    config_path = Path(args.config)
    output_dir = Path(args.output)

    config = load_config(config_path)
    candidate = load_candidate(config_path)

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required.")

    agent = JobSearchAgent(candidate, config)
    results = await agent.run()
    output_path = persist_results(results, output_dir)

    print(f"Saved {len(results)} scored jobs to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
