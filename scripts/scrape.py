"""Scrape California benefits websites to raw_pages.jsonl"""
import json
import time
import urllib.robotparser
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

OUTPUT = Path(__file__).parent / "data" / "raw_pages.jsonl"
OUTPUT.parent.mkdir(exist_ok=True)

TARGETS = [
    ("https://www.cdss.ca.gov/calfresh", "CalFresh", "CalFresh"),
    ("https://www.cdss.ca.gov/calworks", "CalWORKs", "CalWORKs"),
    ("https://edd.ca.gov/en/unemployment/", "Unemployment Insurance", "UI"),
    ("https://edd.ca.gov/en/unemployment/eligibility/", "Unemployment Insurance", "UI Eligibility"),
    ("https://edd.ca.gov/en/unemployment/filing_a_ui_claim/", "Unemployment Insurance", "UI How to Apply"),
    ("https://www.cdss.ca.gov/inforesources/cdss-programs/wic", "WIC", "WIC"),
    ("https://www.dhcs.ca.gov/services/medi-cal/Pages/whatismedi-cal.aspx", "Medi-Cal", "Medi-Cal"),
    ("https://www.dhcs.ca.gov/services/medi-cal/Pages/medi-caleligibility.aspx", "Medi-Cal", "Medi-Cal Eligibility"),
    ("https://www.coveredca.com/learn/how-it-works/", "Health", "Covered California"),
    ("https://www.coveredca.com/support/before-you-buy/who-can-enroll/", "Health", "Covered CA Eligibility"),
    ("https://www.cpuc.ca.gov/industries-and-topics/electrical-energy/electric-costs/care-fera-program", "Utilities", "CARE/FERA"),
    ("https://edd.ca.gov/en/jobs_and_training/", "Employment", "Job Search"),
    ("https://www.cdss.ca.gov/food-nutrition", "CalFresh", "Food Nutrition"),
    ("https://www.cdss.ca.gov/cash-assistance", "CalWORKs", "Cash Assistance"),
    ("https://www.211ca.org/", "General", "211 CA Services"),
    ("https://edd.ca.gov/en/disability/", "Disability Insurance", "SDI"),
    ("https://edd.ca.gov/en/disability/paid-family-leave/", "Paid Family Leave", "PFL"),
    ("https://www.cdss.ca.gov/inforesources/cdss-programs/general-assistance-general-relief", "General Assistance", "General Assistance"),
]

HEADERS = {"User-Agent": "BenefitsFlow Research Bot (educational project)"}
_rp_cache: dict[str, urllib.robotparser.RobotFileParser] = {}


def robots_ok(url: str) -> bool:
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    if base not in _rp_cache:
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(f"{base}/robots.txt")
        try:
            rp.read()
        except Exception:
            return True
        _rp_cache[base] = rp
    return _rp_cache[base].can_fetch(HEADERS["User-Agent"], url)


def extract_text(html: str, url: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["nav", "footer", "script", "style", "header", "aside"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.find(id="content") or soup.find(class_="content")
    root = main if main else soup.body or soup

    sections = []
    current_section = "General"
    current_text = []

    for el in root.find_all(["h1", "h2", "h3", "p", "li", "td"]):
        if el.name in ("h1", "h2", "h3"):
            if current_text:
                sections.append((current_section, " ".join(current_text).strip()))
                current_text = []
            current_section = el.get_text(" ", strip=True)
        else:
            t = el.get_text(" ", strip=True)
            if t:
                current_text.append(t)

    if current_text:
        sections.append((current_section, " ".join(current_text).strip()))

    return sections


def main():
    written = 0
    with OUTPUT.open("w") as f:
        for url, program_name, section_prefix in TARGETS:
            if not robots_ok(url):
                print(f"  Skipped (robots.txt): {url}")
                continue
            try:
                resp = requests.get(url, headers=HEADERS, timeout=15)
                resp.raise_for_status()
            except Exception as e:
                print(f"  Error fetching {url}: {e}")
                continue

            sections = extract_text(resp.text, url)
            for section, text in sections:
                if len(text) < 50:
                    continue
                record = {
                    "url": url,
                    "program_name": program_name,
                    "section": f"{section_prefix} — {section}",
                    "text": text,
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                }
                f.write(json.dumps(record) + "\n")
                written += 1

            print(f"  Scraped {len(sections)} sections from {url}")
            time.sleep(1)

    print(f"\nDone. Wrote {written} records to {OUTPUT}")


if __name__ == "__main__":
    main()
