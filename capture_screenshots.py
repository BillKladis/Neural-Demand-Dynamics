"""
Capture real screenshots of the Pricing Strategy Simulator with Playwright.
No API key required - the app runs entirely on local trained models.
"""
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

PORT = 8503
BASE = f"http://localhost:{PORT}"
SHOTS = Path("screenshots")
SHOTS.mkdir(exist_ok=True)

VIEWS = [
    ("scenario=Promotion", "01_promotion.png"),
    ("scenario=Rapid+repricing", "02_rapid_repricing.png"),
    ("scenario=Gradual+ramp", "03_gradual_ramp.png"),
]


def wait_up(timeout=90):
    for _ in range(timeout):
        try:
            urllib.request.urlopen(BASE, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def settle(page, timeout=60000):
    try:
        page.wait_for_selector("[data-testid='stSpinner']", state="hidden", timeout=timeout)
    except PlaywrightTimeout:
        pass
    page.wait_for_timeout(2500)


def main():
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py",
         "--server.port", str(PORT), "--server.headless", "true",
         "--server.fileWatcherType", "none"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        if not wait_up():
            print("App did not start"); sys.exit(1)
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_context(viewport={"width": 1400, "height": 1000}).new_page()
            for query, fname in VIEWS:
                page.goto(f"{BASE}/?{query}", wait_until="networkidle", timeout=30000)
                settle(page)
                page.screenshot(path=str(SHOTS / fname), full_page=True)
                print("saved", fname)
            browser.close()
    finally:
        proc.terminate(); proc.wait()

    ok = True
    print("\n--- verification ---")
    for f in sorted(SHOTS.glob("*.png")):
        kb = f.stat().st_size / 1024
        flag = "OK" if kb > 10 else "FAIL"
        ok = ok and kb > 10
        print(f"  {f.name}: {kb:.1f} KB [{flag}]")
    print("ALL OK" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
