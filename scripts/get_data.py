"""Download the Telco customer-churn dataset into data/raw/.

The dataset is also committed to the repo, so this is only needed for a fresh fetch.

Usage:
    python scripts/get_data.py
"""
from __future__ import annotations

import urllib.request

from churn import config


def main() -> None:
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {config.DATA_URL}")
    urllib.request.urlretrieve(config.DATA_URL, config.RAW_CSV)
    print(f"Saved -> {config.RAW_CSV}")


if __name__ == "__main__":
    main()
