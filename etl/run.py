"""ETL entry point.

Usage:
  python -m etl.run [--days N]

Scrapes UMD dining data for today through today + N days (default 7),
then loads it into the Postgres database configured by DATABASE_URL.

Intended to run as a scheduled job (cron, Docker CMD, or a CI step).
"""

import argparse
import logging
import sys

from app.database import SessionLocal
from etl import loader, scraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the UMD Dining ETL pipeline")
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days ahead to scrape (default: 7)",
    )
    args = parser.parse_args()

    logger.info("Starting ETL — scraping %d days of menus", args.days)
    halls, menu_items = scraper.scrape(days_ahead=args.days)
    logger.info("Scraped %d halls, %d menu items", len(halls), len(menu_items))

    db = SessionLocal()
    try:
        loader.load(db, halls, menu_items)
    finally:
        db.close()

    logger.info("ETL complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
