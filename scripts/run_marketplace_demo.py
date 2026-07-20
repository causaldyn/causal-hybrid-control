"""Kantorovich OT marketplace demo: driver->rider dispatch + dual surge prices, vs naive baselines.

Run: uv run python scripts/run_marketplace_demo.py
"""

from __future__ import annotations

from chc.matching import MarketplaceMatching, marketplace_report


def main() -> None:
    print("== marketplace dispatch: Kantorovich optimal transport vs naive (synthetic city) ==")
    print(marketplace_report(MarketplaceMatching.synthetic_city(n_zones=12, seed=1)))


if __name__ == "__main__":
    main()
