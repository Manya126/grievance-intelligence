"""
pib_scraper.py  — v3 (Indian News RSS feeds)
---------------------------------------------
PIB blocks all automated access (Playwright, requests, BeautifulSoup).
Their headless Chrome detection returns 0 bytes — nothing works.

REPLACEMENT: 3 major Indian news RSS feeds that:
  ✓ Require zero authentication
  ✓ Update daily with fresh India content
  ✓ Cover government grievances, ministry news, scheme updates
  ✓ Work on any machine with internet access

SOURCES:
  1. The Hindu    — thehindu.com/news/national/
  2. NDTV India   — ndtvnews-india-news RSS
  3. Times of India — India news RSS
  4. Hindustan Times — india-news RSS

No setup needed. Just run:
  python pib_scraper.py
  python pib_scraper.py --test
"""

import os, re, time, logging, argparse
import xml.etree.ElementTree as ET
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_PATH = "../../data/raw"   # relative to src/ingestion/

RSS_FEEDS = [
    ("The Hindu - National",       "https://www.thehindu.com/news/national/feeder/default.rss"),
    ("The Hindu - Economy",        "https://www.thehindu.com/business/Economy/feeder/default.rss"),
    ("NDTV India News",            "https://feeds.feedburner.com/ndtvnews-india-news"),
    ("Times of India - India",     "https://timesofindia.indiatimes.com/rssfeeds/296589292.cms"),
    ("Hindustan Times - India",    "https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml"),
    ("Indian Express - India",     "https://indianexpress.com/section/india/feed/"),
    ("Financial Express - Economy","https://www.financialexpress.com/feed/"),
    ("Economic Times - Economy",   "https://economictimes.indiatimes.com/rssfeedstopstories.cms"),
]

GRIEVANCE_KEYWORDS = [
    "grievance", "complaint", "cpgrams", "redress", "ministry",
    "scheme", "benefit", "pension", "subsidy", "ration", "portal",
    "government", "railway", "electricity", "hospital", "farmer",
    "ayushman", "kisan", "scholarship", "epfo", "income tax",
    "delay", "corruption", "accountability", "citizen", "public",
    "modi", "central government", "state government", "collector",
    "district", "welfare", "relief", "camp", "helpline",
]

DEPT_MAP = {
    "Ministry of Railways":            ["railway", "train", "irctc", "station", "pnr"],
    "Ministry of Finance":             ["income tax", "gst", "epfo", "pan", "tax", "budget", "rbi", "sebi"],
    "Ministry of Health":              ["health", "hospital", "ayushman", "cghs", "medicine", "aiims", "pmjay"],
    "Ministry of Agriculture":         ["farmer", "agriculture", "kisan", "crop", "fertiliser", "pmkisan", "msp"],
    "Ministry of Education":           ["education", "scholarship", "school", "university", "nta", "ugc", "neet"],
    "Ministry of Home Affairs":        ["police", "passport", "aadhaar", "home", "security", "nrc"],
    "Ministry of Labour":              ["labour", "worker", "wage", "mnrega", "esic", "epf", "employment"],
    "Ministry of Power":               ["electricity", "power", "discoms", "meter", "solar", "energy"],
    "Ministry of Telecommunications":  ["telecom", "5g", "broadband", "trai", "jio", "airtel", "spectrum"],
    "Ministry of Housing":             ["housing", "pmay", "smart city", "urban", "dda", "flat"],
    "Department of Posts":             ["post office", "postal", "india post", "ppf"],
    "Ministry of Water Resources":     ["water", "jal jeevan", "irrigation", "dam", "river"],
    "Ministry of Transport":           ["highway", "toll", "nhai", "road", "driving", "fastag"],
    "Ministry of Petroleum":           ["lpg", "petrol", "diesel", "gas", "cylinder", "ujjwala", "oil"],
    "Ministry of Panchayati Raj":      ["panchayat", "gram", "village", "rural"],
    "Ministry of Social Justice":      ["disability", "sc", "st", "obc", "tribal", "dalit"],
    "Ministry of Commerce":            ["export", "import", "customs", "trade", "startup", "msme"],
    "DARPG / Grievances":              ["grievance", "cpgrams", "redress", "complaint", "citizen service"],
}

STATES = [
    "Uttar Pradesh", "Maharashtra", "Bihar", "West Bengal", "Madhya Pradesh",
    "Rajasthan", "Tamil Nadu", "Karnataka", "Gujarat", "Andhra Pradesh",
    "Odisha", "Telangana", "Punjab", "Haryana", "Delhi", "Assam",
    "Jharkhand", "Chhattisgarh", "Kerala", "Uttarakhand",
]

def _dept(text: str) -> str:
    t = text.lower()
    for dept, kws in DEPT_MAP.items():
        if any(k in t for k in kws):
            return dept
    return "Ministry of Home Affairs"

def _state(text: str) -> str:
    t = text.lower()
    for s in STATES:
        if s.lower() in t:
            return s
    return "national"

def _is_relevant(text: str) -> bool:
    """Keep article if it mentions India govt/grievance topics."""
    t = text.lower()
    return any(kw in t for kw in GRIEVANCE_KEYWORDS)

def _clean(text: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&[a-z]+;', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def fetch_rss(feed_name: str, feed_url: str, max_items: int = 50) -> list[dict]:
    """Fetch and parse a single RSS feed."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept":     "application/rss+xml, application/xml, text/xml, */*",
    }
    try:
        r = requests.get(feed_url, headers=headers, timeout=20)
        if r.status_code != 200:
            logger.warning(f"  {feed_name}: HTTP {r.status_code}")
            return []

        # Parse RSS XML
        try:
            root = ET.fromstring(r.content)
        except ET.ParseError as e:
            # Some feeds have encoding issues — try stripping BOM
            content = r.content.lstrip(b'\xef\xbb\xbf')
            try:
                root = ET.fromstring(content)
            except ET.ParseError:
                logger.warning(f"  {feed_name}: XML parse failed — {e}")
                return []

        items = root.findall(".//item")
        records = []

        for item in items[:max_items]:
            title   = _clean(item.findtext("title")       or "")
            desc    = _clean(item.findtext("description") or "")
            link    = (item.findtext("link")              or "").strip()
            pub     = (item.findtext("pubDate")           or datetime.now().isoformat()).strip()
            text    = f"{title}. {desc}".strip(". ")

            if len(text) < 20:
                continue
            if not _is_relevant(text):
                continue

            records.append({
                "grievance_id":    f"rss_{abs(hash(link or text)):010d}",
                "text":            text[:1200],
                "department":      _dept(text),
                "state":           _state(text),
                "status":          "open",
                "date_filed":      pub,
                "date_resolved":   "",
                "feedback_rating": "",
                "source":          "india_news_rss",
                "query_used":      feed_name,
                "lang":            "en",
                "likes":           0,
                "retweets":        0,
                "scraped_at":      datetime.now().isoformat(),
            })

        return records

    except requests.exceptions.ConnectionError:
        logger.warning(f"  {feed_name}: Connection failed")
        return []
    except requests.exceptions.Timeout:
        logger.warning(f"  {feed_name}: Timed out")
        return []
    except Exception as e:
        logger.error(f"  {feed_name}: Unexpected error — {e}")
        return []


def run(n_feeds: int = None, save: bool = True) -> pd.DataFrame:
    """
    Fetch from all Indian news RSS feeds.
    Called by live_collector.py --source pib
    """
    logger.info("━━ SOURCE: Indian News RSS feeds (live, no auth) ━━")
    feeds = RSS_FEEDS[:n_feeds] if n_feeds else RSS_FEEDS

    all_records = []
    for feed_name, feed_url in feeds:
        records = fetch_rss(feed_name, feed_url)
        logger.info(f"  {feed_name}: {len(records)} relevant articles")
        all_records.extend(records)
        time.sleep(0.5)   # polite delay

    if not all_records:
        logger.warning("  RSS: 0 articles. Check internet connection.")
        return pd.DataFrame()

    df = pd.DataFrame(all_records).drop_duplicates(subset=["grievance_id"])
    logger.info(f"  RSS total: {len(df)} unique articles ✓")

    if save:
        os.makedirs(RAW_PATH, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(RAW_PATH, f"rss_{ts}.csv")
        df.to_csv(path, index=False, encoding="utf-8")
        logger.info(f"  Saved → {path}")

        # Save to DB (only when run from project root)
        try:
            from db_writer import save_to_db
            inserted = save_to_db(df)
            logger.info(f"  Inserted {inserted} new records into DB")
        except ImportError:
            logger.info("  (db_writer not in path — run from project root to save to DB)")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Indian news RSS feeds")
    parser.add_argument("--test",   action="store_true", help="Print results, don't save to DB")
    parser.add_argument("--feeds",  type=int, default=None, help="Number of feeds to use (default: all)")
    args = parser.parse_args()

    df = run(n_feeds=args.feeds, save=not args.test)

    if not df.empty:
        print(f"\n✓ {len(df)} articles collected")
        print(f"\nSources breakdown:")
        print(df["query_used"].value_counts().to_string())
        print(f"\nTop departments:")
        print(df["department"].value_counts().head(6).to_string())
        print(f"\nSample articles:")
        for _, r in df.sample(min(5, len(df))).iterrows():
            print(f"  [{r['department'][:28]}] {r['text'][:85]}...")
    else:
        print("\n✗ 0 articles. Check internet connection.")
        print("  All 8 RSS feeds tried: The Hindu, NDTV, TOI, HT, IE, FE, ET")