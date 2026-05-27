"""
pib_debug.py  — run this ONCE to find the correct CSS selectors
----------------------------------------------------------------
Saves the fully-rendered PIB page HTML to data/raw/pib_debug.html
so we can inspect exactly what the DOM looks like after JS runs.

Run:
  cd src/ingestion
  python pib_debug.py
  
Then open:  data/raw/pib_debug.html  in your browser
And check:  data/raw/pib_links.txt   for all <a> tags found
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from playwright.sync_api import sync_playwright
import time

URL = "https://www.pib.gov.in/allRel.aspx"
OUT_HTML  = "../../data/raw/pib_debug.html"
OUT_LINKS = "../../data/raw/pib_links.txt"

print(f"Opening {URL} with Playwright...")

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
    ctx = browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        viewport={"width": 1280, "height": 900},
    )
    page = ctx.new_page()

    # Step 1: go to page
    print("Navigating...")
    page.goto(URL, timeout=60000)

    # Step 2: wait different durations and check what loads
    for wait_sec in [3, 5, 8, 12]:
        page.wait_for_timeout(wait_sec * 1000)
        links = page.query_selector_all("a")
        print(f"  After {wait_sec}s wait: {len(links)} <a> tags total")

        # Find links with press-release-like hrefs
        pr_links = [l for l in links if "PRID" in (l.get_attribute("href") or "")
                    or "PressRelease" in (l.get_attribute("href") or "")]
        print(f"    → Press release links (PRID/PressRelease in href): {len(pr_links)}")
        if pr_links:
            print(f"    → First link: {pr_links[0].get_attribute('href')} | {pr_links[0].inner_text()[:60]}")
            break

    # Step 3: try scrolling to trigger lazy load
    print("Scrolling page...")
    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    page.wait_for_timeout(3000)

    # Step 4: dump full rendered HTML
    html = page.content()
    os.makedirs("../../data/raw", exist_ok=True)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Full HTML saved → {OUT_HTML}  ({len(html)//1024} KB)")

    # Step 5: dump all unique hrefs and their text
    all_links = page.query_selector_all("a")
    with open(OUT_LINKS, "w", encoding="utf-8") as f:
        for lnk in all_links:
            href = lnk.get_attribute("href") or ""
            text = lnk.inner_text().strip().replace("\n", " ")[:80]
            if href or text:
                f.write(f"{href}  |  {text}\n")
    print(f"All {len(all_links)} links saved → {OUT_LINKS}")

    # Step 6: print all unique href patterns
    hrefs = set()
    for lnk in all_links:
        h = lnk.get_attribute("href") or ""
        if h and not h.startswith("javascript") and not h.startswith("#"):
            hrefs.add(h[:80])

    print(f"\nUnique non-JS hrefs found ({len(hrefs)}):")
    for h in sorted(hrefs)[:30]:
        print(f"  {h}")

    # Step 7: check all div/ul class names (to find content containers)
    divs = page.query_selector_all("div[class], ul[class], section[class]")
    classes = set()
    for d in divs:
        cls = d.get_attribute("class") or ""
        for c in cls.split():
            classes.add(c)
    print(f"\nCSS classes on divs/uls/sections ({len(classes)}):")
    for c in sorted(classes)[:40]:
        print(f"  .{c}")

    browser.close()

print("\nDone. Open data/raw/pib_debug.html in browser to inspect the page.")
print("Share the 'Unique non-JS hrefs' and 'CSS classes' output here.")