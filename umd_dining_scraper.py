#!/usr/bin/env python3
"""
UMD Dining Scraper: Breakfast/Lunch/Dinner -> CSV

Requirements:
    pip install playwright beautifulsoup4 requests
    playwright install

What this script does:
  - For a given hall ('south', 'yahentamitsi', '251') and date ('12/1/2025')
  - Uses Playwright to click Breakfast, Lunch, Dinner
  - Collects all food items for each meal with their label.aspx URLs
  - Fetches nutrition info for each item (with a JSON cache)
  - Writes everything into a CSV file
"""

import csv
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, quote

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

# -----------------------------
# Config
# -----------------------------

BASE_URL = "https://nutrition.umd.edu"

HALL_LOCATION = {
    "south": 16,          # South Campus Dining
    "yahentamitsi": 19,   # Yahentamitsi
    "The Y": 19,
    "251": 51,            # 251 North
    "251 north": 51,
}

CACHE_FILE = "label_cache.json"


# -----------------------------
# Cache helpers
# -----------------------------

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


# -----------------------------
# Core helpers
# -----------------------------

def get_location_num(hall_name: str) -> int:
    key = hall_name.strip().lower()
    if key not in HALL_LOCATION:
        raise ValueError(f"Unknown dining hall: {hall_name}")
    return HALL_LOCATION[key]


def get_items_grouped_by_meal(location_num: int, date_str: str):
    meals_dict = {"breakfast": [], "lunch": [], "dinner": []}

    url = f"{BASE_URL}/?locationNum={location_num}&dtdate={quote(date_str)}"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # 🔧 CHANGED THIS LINE
        page.goto(url, wait_until="load", timeout=60000)

        for meal in ["Breakfast", "Lunch", "Dinner"]:
            page.click(f"text={meal}")
            page.wait_for_timeout(800)

            anchors = page.query_selector_all("a[href^='label.aspx']")
            seen = set()
            items = []
            for a in anchors:
                href = a.get_attribute("href")
                if not href:
                    continue
                if not href.lower().startswith("label.aspx"):
                    continue
                full_url = urljoin(BASE_URL + "/", href)
                if full_url in seen:
                    continue
                seen.add(full_url)
                name = a.inner_text().strip()
                if not name:
                    continue
                items.append({"name": name, "label_url": full_url})

            meals_dict[meal.lower()] = items
            print(f"{meal}: {len(items)} items")

        browser.close()

    return meals_dict


def parse_label_page(label_url: str):
    """
    Fetch and parse one label.aspx page.

    Returns:
      {
        "name": ...,
        "serving_size": ...,
        "calories": ...,
        "total_fat": ...,
        "saturated_fat": ...,
        "carbs": ...,
        "protein": ...,
        "label_url": ...
      }
    """
    resp = requests.get(label_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    title_tag = soup.find(["h1", "h2"])
    title = title_tag.get_text(strip=True) if title_tag else None
    text = soup.get_text(" ", strip=True)

    def grab(pattern, default=None):
        m = re.search(pattern, text)
        return m.group(1).strip() if m else default

    return {
        "name": title,
        "serving_size": grab(r"Serving size\s+(.+?)\s+Calories per serving"),
        "calories": grab(r"Calories per serving\s+(\d+)"),
        "total_fat": grab(r"Total Fat\s*([\d\.]+g)"),
        "saturated_fat": grab(r"Saturated Fat\s*([\d\.]+g)"),
        "carbs": grab(r"Total Carbohydrate\.\s*([\d\.]+g)"),
        "protein": grab(r"Protein\s*([\d\.]+g)"),
        "label_url": label_url,
    }


def enrich_meals_with_nutrition(hall_name: str, date_str: str, max_workers: int = 8):
    """
    High-level function:
      - Scrape items per meal (Breakfast/Lunch/Dinner)
      - Fetch nutrition data with caching
      - Return a flat list of rows ready for CSV

    Each row looks like:
      {
        "hall": ...,
        "date": ...,
        "meal": "Breakfast" | "Lunch" | "Dinner",
        "food_name": ...,
        "serving_size": ...,
        "calories": ...,
        "total_fat": ...,
        "saturated_fat": ...,
        "carbs": ...,
        "protein": ...,
        "label_url": ...,
      }
    """
    cache = load_cache()
    location_num = get_location_num(hall_name)

    # Step 1: scrape items by meal
    meals_dict = get_items_grouped_by_meal(location_num, date_str)

    # Step 2: collect all unique label URLs across meals
    all_urls = set()
    for items in meals_dict.values():
        for item in items:
            all_urls.add(item["label_url"])

    # Split into cached vs missing
    missing_urls = []
    for url in all_urls:
        if url not in cache:
            missing_urls.append(url)

    print(f"Total unique foods: {len(all_urls)} | In cache: {len(all_urls) - len(missing_urls)} | To fetch: {len(missing_urls)}")

    # Step 3: fetch missing label pages in parallel
    if missing_urls:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_url = {
                pool.submit(parse_label_page, url): url
                for url in missing_urls
            }
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    data = future.result()
                    cache[url] = data
                except Exception as e:
                    print(f"Error fetching {url}: {e}")

        save_cache(cache)

    # Step 4: build flat list of rows
    rows = []
    for meal_key, items in meals_dict.items():
        meal_name = meal_key.capitalize()  # 'breakfast' -> 'Breakfast'
        for item in items:
            url = item["label_url"]
            nut = cache.get(url, {})
            row = {
                "hall": hall_name,
                "date": date_str,
                "meal": meal_name,
                "food_name": item["name"],
                "serving_size": nut.get("serving_size"),
                "calories": nut.get("calories"),
                "total_fat": nut.get("total_fat"),
                "saturated_fat": nut.get("saturated_fat"),
                "carbs": nut.get("carbs"),
                "protein": nut.get("protein"),
                "label_url": url,
            }
            rows.append(row)

    return rows


def write_rows_to_csv(rows, filename: str):
    """Write the list of row dicts into a CSV file."""
    fieldnames = [
        "hall",
        "date",
        "meal",
        "food_name",
        "serving_size",
        "calories",
        "total_fat",
        "saturated_fat",
        "carbs",
        "protein",
        "label_url",
    ]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {len(rows)} rows to {filename}")


# -----------------------------
# Example usage
# -----------------------------

if __name__ == "__main__":
    import os

    # 👇 change these to whatever you want
    hall = "south"          # "south", "yahentamitsi", or "251"
    date = "12/1/2025"      # e.g. "12/1/2025"
    output_csv = f"{hall}_{date.replace('/', '-')}.csv"

    print("Current working directory:", os.getcwd())
    print("Hall:", hall, "Date:", date)
    print("CSV will be named:", output_csv)

    rows = enrich_meals_with_nutrition(hall, date, max_workers=10)
    print("Number of rows scraped:", len(rows))

    write_rows_to_csv(rows, output_csv)

    full_path = os.path.join(os.getcwd(), output_csv)
    print("Full CSV path:", full_path)

