"""
run_ingestion.py
----------------
Master ingestion script. Run this to collect data from all sources.
Automatically falls back to synthetic data if APIs are unavailable.

Usage:
    python run_ingestion.py                   # all sources
    python run_ingestion.py --source datagov  # only data.gov.in
    python run_ingestion.py --source twitter  # only Twitter
    python run_ingestion.py --source synthetic --records 3000

    # Check what's in the DB after running:
    python run_ingestion.py --stats
"""

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from db_writer import get_stats, initialise_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_datagov(max_records: int = 5000):
    logger.info("=" * 50)
    logger.info("SOURCE: data.gov.in (CPGRAMS official dataset)")
    logger.info("=" * 50)
    try:
        from datagov_collector import run
        df = run(max_records=max_records)
        logger.info(f"data.gov.in: {len(df)} records collected.")
        return len(df)
    except ValueError as e:
        logger.warning(f"Skipping data.gov.in: {e}")
        return 0
    except Exception as e:
        logger.error(f"data.gov.in failed: {e}")
        return 0


def run_twitter():
    logger.info("=" * 50)
    logger.info("SOURCE: Twitter/X")
    logger.info("=" * 50)
    try:
        from twitter_collector import run
        df = run()
        logger.info(f"Twitter: {len(df)} tweets collected.")
        return len(df)
    except ValueError as e:
        logger.warning(f"Skipping Twitter: {e}")
        return 0
    except Exception as e:
        logger.error(f"Twitter failed: {e}")
        return 0


def run_synthetic(n_records: int = 2000):
    logger.info("=" * 50)
    logger.info("SOURCE: Synthetic generator (fallback / supplement)")
    logger.info("=" * 50)
    from synthetic_generator import run
    df = run(n_records=n_records)
    logger.info(f"Synthetic: {len(df)} records generated.")
    return len(df)


def print_stats():
    stats = get_stats()
    print("\n" + "=" * 50)
    print("DATABASE SUMMARY")
    print("=" * 50)
    print(f"Total records:    {stats['total_records']}")
    print(f"\nBy source:")
    for s in stats["by_source"]:
        print(f"  {s['source']:<25} {s['n']}")
    print(f"\nTop 5 departments:")
    for d in stats["top_departments"]:
        print(f"  {d['department']:<40} {d['n']}")
    print(f"\nTop 5 states:")
    for s in stats["top_states"]:
        print(f"  {s['state']:<25} {s['n']}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run grievance data ingestion")
    parser.add_argument("--source",  choices=["all", "datagov", "twitter", "synthetic"],
                        default="all", help="Which source to collect from")
    parser.add_argument("--records", type=int, default=2000,
                        help="Records for synthetic generator")
    parser.add_argument("--stats",   action="store_true",
                        help="Print DB stats and exit")
    args = parser.parse_args()

    if args.stats:
        print_stats()
        sys.exit(0)

    initialise_db()
    total = 0

    if args.source in ("all", "datagov"):
        total += run_datagov()

    if args.source in ("all", "twitter"):
        total += run_twitter()

    # Always run synthetic if real sources got < 1000 records combined
    if args.source == "synthetic" or (args.source == "all" and total < 1000):
        logger.info("Supplementing with synthetic data...")
        total += run_synthetic(n_records=args.records)

    print_stats()
    logger.info(f"\nIngestion complete. Total new records this run: {total}")