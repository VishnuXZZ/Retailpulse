"""Headless Streamlit: launch the app and capture each of the 6 pages as PNG."""
import subprocess, time, sys, os
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path("/home/user/RetailPulse").resolve()
SCREEN = ROOT / "screenshots"
SCREEN.mkdir(exist_ok=True, parents=True)

os.environ["STREAMLIT_GLOBAL_SUPPRESS_DEPRECATION_WARNINGS"] = "true"
env = os.environ.copy()
env["PATH"] = env.get("PATH", "") + ":" + "/home/user/.local/bin"

print("Starting Streamlit …")
proc = subprocess.Popen(
    ["streamlit", "run", "app.py",
     "--server.headless=true", "--server.port=8765",
     "--server.address=127.0.0.1", "--browser.gatherUsageStats=false"],
    cwd=str(ROOT), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    time.sleep(15)
    pages = [
        ("01_home.png", "Sales Dashboard"),
        ("02_sales.png", "Sales Dashboard"),
        ("03_customers.png", "Customer Analytics"),
        ("04_forecast.png", "Demand Forecasting"),
        ("05_churn.png", "Churn Risk"),
        ("06_inventory.png", "Inventory Optimization"),
        ("07_monitoring.png", "Model Monitoring & MLOps"),
    ]
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(viewport={"width": 1500, "height": 1100},
                                       device_scale_factor=2)
        page = context.new_page()
        url = "http://127.0.0.1:8765/"
        page.goto(url, wait_until="networkidle", timeout=90_000)
        time.sleep(5)
        page.screenshot(path=str(SCREEN / "01_home.png"), full_page=True)
        for name, sel in pages[1:]:
            page.evaluate(f"""() => {{
                const els = [...document.querySelectorAll('label, span, div, p')];
                const target = els.find(e => e.innerText && e.innerText.trim() === '{sel}');
                if (target) target.click();
            }}""")
            time.sleep(6)
            page.screenshot(path=str(SCREEN / name), full_page=True)
            print(f"captured {name}")
        browser.close()
finally:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
print("done.")
