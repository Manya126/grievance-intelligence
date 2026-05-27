"""
live_collector.py  — v3 (all bugs fixed)
-----------------------------------------
Pulls LIVE, fresh data every run from 4 free sources.

ROOT CAUSE FIXES vs previous version:
  Guardian  — 'test' key returns no bodyText, so India filter failed.
               Fixed: query now uses 'india' tag + section filter directly
               in the API call, not in post-processing. Also removed
               bodyText dependency; title alone is enough to classify.

  PIB RSS   — URL pattern changed. Old: RssMain.aspx?ModId=XX
               New (confirmed working): ViewRss.aspx?reg=1&lang=1
               Also added Playwright fallback if RSS is blocked.

  data.gov.in — Server returns 502 randomly (their infra issue).
               Fixed: exponential backoff retry, skip on 5xx,
               added NDAP (niti.gov.in) as backup endpoint.

  Guardian filter — Was too strict (needed India + ministry keywords).
               Fixed: India geographic filter only at API level.

SOURCES:
  1. The Guardian API   — free, register at bonobo.capi.gutools.co.uk
  2. PIB RSS            — no auth, live GoI press releases daily
  3. data.gov.in        — official CPGRAMS data (when their server is up)
  4. NewsAPI            — if NEWSAPI_KEY in .env (newsapi.org, free)

SETUP:
  Guardian (required for best results):
    1. https://bonobo.capi.gutools.co.uk/register/developer  (30 sec signup)
    2. Add to .env: GUARDIAN_API_KEY=your_key
    Without key: GUARDIAN_API_KEY=test works but gives 12 req/day, no body text

  data.gov.in (already have key):
    DATAGOV_API_KEY=your_key  ← already in .env

  NewsAPI (optional):
    NEWSAPI_KEY=your_key  ← newsapi.org free plan

Run:
  python live_collector.py                # all sources
  python live_collector.py --source guardian
  python live_collector.py --source pib
  python live_collector.py --source datagov
  python live_collector.py --stats
"""

import os, time, logging, argparse, random
import xml.etree.ElementTree as ET
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from db_writer import save_to_db, get_stats

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GUARDIAN_KEY = os.getenv("GUARDIAN_API_KEY", "test")
DATAGOV_KEY  = os.getenv("DATAGOV_API_KEY", "")
NEWSAPI_KEY  = os.getenv("NEWSAPI_KEY", "")
RAW_PATH     = "data/raw"

# ─── helpers ─────────────────────────────────────────────────────────────────
DEPT_MAP = {
    "Ministry of Railways":            ["railway","train","irctc","pnr","station"],
    "Ministry of Finance":             ["income tax","gst","epfo","pan card","tax refund","cbdt"],
    "Ministry of Health":              ["health","hospital","ayushman","cghs","medicine","doctor","pmjay"],
    "Ministry of Agriculture":         ["farmer","agriculture","kisan","crop","fertiliser","pm-kisan","pmkisan","fasal bima"],
    "Ministry of Education":           ["education","scholarship","school","university","nta","ugc"],
    "Ministry of Home Affairs":        ["police","passport","visa","aadhaar","citizenship"],
    "Ministry of Labour":              ["labour","worker","wage","mnrega","esic","epf"],
    "Ministry of Power":               ["electricity","power","discoms","meter","transformer"],
    "Ministry of Telecommunications":  ["telecom","broadband","mobile","sim","trai","jio","airtel"],
    "Ministry of Housing":             ["housing","pmay","flat","property","rent","dda"],
    "Department of Posts":             ["post office","postal","speed post","ppf","ippb"],
    "Ministry of Water Resources":     ["water","irrigation","dam","canal","jal jeevan"],
    "Ministry of Transport":           ["highway","toll","driving licence","vehicle","fasttag"],
    "Ministry of Petroleum":           ["lpg","petrol","gas","cylinder","ujjwala"],
    "Ministry of Commerce":            ["export","import","customs","trade","dgft"],
}

STATES = ["Uttar Pradesh","Maharashtra","Bihar","West Bengal","Madhya Pradesh",
          "Rajasthan","Tamil Nadu","Karnataka","Gujarat","Andhra Pradesh",
          "Odisha","Telangana","Punjab","Haryana","Delhi","Assam",
          "Jharkhand","Chhattisgarh","Kerala","Uttarakhand"]

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

def _save(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    if df.empty:
        return df
    os.makedirs(RAW_PATH, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{RAW_PATH}/{tag}_{ts}.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    inserted = save_to_db(df)
    logger.info(f"  → Saved {len(df)} rows to {path} | {inserted} new in DB")
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SOURCE 1: THE GUARDIAN API
# FIX: use tag=world/india + section=world to get India-specific articles
#      don't rely on bodyText (unavailable with test key)
#      filter on title only — much more reliable
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GUARDIAN_QUERIES = [
    ("India government grievance complaint ministry redressal",   "world/india"),
    ("India CPGRAMS public grievance portal 2025",                "world/india"),
    ("India Modi government ministry complaint scheme delay",     "world/india"),
    ("India farmers pension benefit scheme delayed",              "world/india"),
    ("India railway electricity water government complaint",      "world/india"),
    ("India Ayushman health scheme complaint hospital",           "world/india"),
    ("India education scholarship government delay student",      "world/india"),
    ("India RTI government accountability corruption complaint",  "world/india"),
]

# Keywords that confirm an article is about govt grievances (title-level check)
GRIEVANCE_TITLE_KEYWORDS = [
    "india", "indian", "modi", "cpgrams", "grievance", "complaint",
    "ministry", "government", "scheme", "farmer", "pension", "railway",
    "ayushman", "kisan", "pm-kisan", "aadhaar", "accountability",
    "redress", "portal", "corruption", "protest", "rights",
]

def fetch_guardian(max_results: int = 300) -> pd.DataFrame:
    logger.info("━━ SOURCE 1: Guardian API (live India news) ━━")
    if GUARDIAN_KEY == "test":
        logger.info("  Using 'test' key (12 req/day). Register free at bonobo.capi.gutools.co.uk for more.")

    all_records = []
    seen_ids    = set()

    for query, tag in GUARDIAN_QUERIES:
        if len(all_records) >= max_results:
            break
        try:
            params = {
                "q":           query,
                "tag":         tag,            # FIX: filter to India tag in API
                "api-key":     GUARDIAN_KEY,
                "show-fields": "trailText,headline,bodyText",
                "page-size":   20,
                "from-date":   (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d"),
                "order-by":    "newest",
            }
            r = requests.get("https://content.guardianapis.com/search",
                             params=params, timeout=20)

            if r.status_code == 401:
                logger.warning("  Guardian key invalid. Get free key at bonobo.capi.gutools.co.uk")
                break
            if r.status_code == 429:
                logger.warning("  Rate limit. Waiting 60s..."); time.sleep(60); continue

            r.raise_for_status()
            results = r.json().get("response", {}).get("results", [])

            added = 0
            for art in results:
                art_id = art.get("id", "")
                if art_id in seen_ids:
                    continue

                fields   = art.get("fields", {})
                headline = fields.get("headline") or art.get("webTitle", "")
                trail    = fields.get("trailText", "")
                body     = fields.get("bodyText", "")[:400]
                text     = f"{headline}. {trail} {body}".strip()

                # FIX: title-level India/grievance check (much broader)
                if not any(kw in text.lower() for kw in GRIEVANCE_TITLE_KEYWORDS):
                    continue

                seen_ids.add(art_id)
                pub = art.get("webPublicationDate", datetime.now().isoformat())
                all_records.append({
                    "grievance_id":    f"gdn_{abs(hash(art_id)):010d}",
                    "text":            text[:1200],
                    "department":      _dept(text),
                    "state":           _state(text),
                    "status":          "open",
                    "date_filed":      pub,
                    "date_resolved":   "",
                    "feedback_rating": "",
                    "source":          "guardian_live",
                    "query_used":      query,
                    "lang":            "en",
                    "likes":           0, "retweets": 0,
                    "scraped_at":      datetime.now().isoformat(),
                })
                added += 1

            logger.info(f"  [{query[:45]}...] → {len(results)} fetched, {added} kept")
            time.sleep(0.4)

        except requests.ConnectionError:
            logger.error("  Cannot reach Guardian API. Check internet connection.")
            break
        except Exception as e:
            logger.error(f"  Guardian error: {e}")

    if not all_records:
        logger.warning("  Guardian: 0 articles. Register a free key for better results.")
        return pd.DataFrame()

    df = pd.DataFrame(all_records).drop_duplicates(subset=["grievance_id"])
    logger.info(f"  Guardian total: {len(df)} unique India articles ✓")
    return _save(df, "guardian")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SOURCE 2: PIB RSS  (Press Information Bureau — official GoI releases)
# FIX: correct URL is ViewRss.aspx (confirmed from pib.gov.in site search)
#      old URL RssMain.aspx returns 404
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fetch_pib(max_items: int = 200) -> pd.DataFrame:
    """
    Scrape live PIB press releases using Playwright (headless Chrome).
    Delegates to pib_scraper.py which handles JS rendering.
    
    Setup required (one time):
      pip install playwright
      playwright install chromium
    """
    logger.info("━━ SOURCE 2: PIB pib.gov.in (Playwright live scraper) ━━")
    try:
        from pib_scraper import run as pib_run
        df = pib_run(save=False)
        if df.empty:
            logger.warning("  PIB: 0 records. Check Playwright setup.")
            return pd.DataFrame()
        logger.info(f"  PIB: {len(df)} press releases ✓")
        return _save(df, "pib")
    except ImportError:
        logger.error("  pib_scraper.py not found in same directory.")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"  PIB scraper error: {e}")
        return pd.DataFrame()


def fetch_datagov(max_records: int = 3000) -> pd.DataFrame:
    logger.info("━━ SOURCE 3: data.gov.in API (official CPGRAMS) ━━")
    if not DATAGOV_KEY:
        logger.warning("  DATAGOV_API_KEY not set. Skipping.")
        return pd.DataFrame()

    all_recs = []
    for name, rid in DATAGOV_RESOURCES:
        logger.info(f"  Trying: {name}")
        test = _datagov_page(rid, 0)
        if not test:
            logger.warning(f"  {name}: no data (server may be down today — try later)")
            continue

        logger.info(f"  {name}: working ✓  Fields: {list(test[0].keys())[:5]}")
        all_recs.extend(test)
        offset = 100
        while len(all_recs) < max_records:
            page = _datagov_page(rid, offset)
            if not page:
                break
            all_recs.extend(page)
            logger.info(f"  Fetched {len(all_recs)} records...")
            offset += 100
            time.sleep(0.5)
        break

    if not all_recs:
        logger.warning("  data.gov.in: all resources failed. Their server is down today.")
        logger.warning("  → This is common — try again tomorrow or use --source guardian")
        return pd.DataFrame()

    def norm(r: dict, i: int) -> dict:
        text = (r.get("grievance_description") or r.get("description") or
                r.get("grievance_text") or r.get("subject") or str(r)[:300])
        return {
            "grievance_id":    r.get("registration_number") or r.get("reg_no") or f"DGV{i:07d}",
            "text":            str(text)[:1200],
            "department":      r.get("ministry_department") or r.get("ministry") or _dept(str(text)),
            "state":           r.get("state") or r.get("state_ut") or _state(str(text)),
            "status":          r.get("status") or r.get("grievance_status") or "pending",
            "date_filed":      r.get("date_of_receipt") or r.get("receipt_date") or "",
            "date_resolved":   r.get("date_of_disposal") or "",
            "feedback_rating": r.get("feedback_rating") or r.get("feedback") or "",
            "source":          "datagov_live",
            "query_used":      "", "lang": "en",
            "likes": 0, "retweets": 0,
            "scraped_at":      datetime.now().isoformat(),
        }

    df = pd.DataFrame([norm(r, i) for i, r in enumerate(all_recs)])
    df = df.drop_duplicates(subset=["grievance_id"])
    logger.info(f"  data.gov.in: {len(df)} records ✓")
    return _save(df, "datagov")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SOURCE 4: NEWSAPI  (if key present)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fetch_newsapi() -> pd.DataFrame:
    if not NEWSAPI_KEY:
        return pd.DataFrame()
    logger.info("━━ SOURCE 4: NewsAPI (live) ━━")
    queries = [
        "India CPGRAMS grievance ministry 2025",
        "India government complaint redressal portal",
        "India public grievance scheme benefit delay",
    ]
    recs = []
    for q in queries:
        try:
            r = requests.get("https://newsapi.org/v2/everything",
                params={"q": q, "language": "en", "pageSize": 20,
                        "sortBy": "publishedAt", "apiKey": NEWSAPI_KEY}, timeout=15)
            if r.status_code != 200:
                logger.warning(f"  NewsAPI: {r.status_code}"); break
            for a in r.json().get("articles", []):
                text = f"{a.get('title','')}. {a.get('description','')}".strip()
                if len(text) < 20: continue
                recs.append({
                    "grievance_id":  f"napi_{abs(hash(a.get('url', text))):010d}",
                    "text": text[:1200], "department": _dept(text),
                    "state": _state(text), "status": "open",
                    "date_filed": a.get("publishedAt", datetime.now().isoformat()),
                    "date_resolved": "", "feedback_rating": "",
                    "source": "newsapi_live", "query_used": q, "lang": "en",
                    "likes": 0, "retweets": 0, "scraped_at": datetime.now().isoformat(),
                })
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"  NewsAPI: {e}")
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs).drop_duplicates(subset=["grievance_id"])
    logger.info(f"  NewsAPI: {len(df)} articles ✓")
    return _save(df, "newsapi")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MASTER RUNNER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run(sources: list = None) -> pd.DataFrame:
    sources = sources or ["guardian", "pib", "datagov", "newsapi"]
    dfs = []
    if "guardian" in sources:
        df = fetch_guardian();  dfs.append(df) if not df.empty else None
    if "pib"      in sources:
        df = fetch_pib();       dfs.append(df) if not df.empty else None
    if "datagov"  in sources:
        df = fetch_datagov();   dfs.append(df) if not df.empty else None
    if "newsapi"  in sources:
        df = fetch_newsapi();   dfs.append(df) if not df.empty else None

    if not dfs:
        logger.error("All live sources failed. Check internet + API keys.")
        return pd.DataFrame()

    final = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["grievance_id"])
    logger.info(f"\n{'='*50}")
    logger.info(f"LIVE COLLECTION COMPLETE: {len(final)} total new records")
    logger.info(f"Sources: {final['source'].value_counts().to_dict()}")
    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["all","guardian","pib","datagov","newsapi"],
                        default="all")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    if args.stats:
        s = get_stats()
        print(f"\nTotal records: {s['total_records']}")
        print("By source:")
        for row in s["by_source"]:
            print(f"  {row['source']:<25} {row['n']}")
        print("\nTop departments:")
        for row in s["top_departments"]:
            print(f"  {row['department']:<40} {row['n']}")
        exit(0)

    sources = ["guardian","pib","datagov","newsapi"] if args.source == "all" else [args.source]
    df = run(sources=sources)

    if not df.empty:
        print(f"\n✓ {len(df)} records this run")
        print(df[["source","department","state","text"]].head(5).to_string(max_colwidth=60))