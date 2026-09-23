"""Report date ranges that lack proven broker symbol/token mappings."""

from __future__ import annotations

import argparse
from datetime import date

from .provider import DEFAULT_PROVIDER, UniverseCoverageError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, type=date.fromisoformat, help="Inclusive ISO date")
    parser.add_argument("--end", required=True, type=date.fromisoformat, help="Inclusive ISO date")
    parser.add_argument("--universe", default="NIFTY50_PIT_TRADABLE")
    parser.add_argument("--broker", default="kite")
    arguments = parser.parse_args()
    try:
        missing = DEFAULT_PROVIDER.unresolved_broker_mappings(
            arguments.start, arguments.end, broker=arguments.broker, universe_id=arguments.universe
        )
    except (ValueError, UniverseCoverageError) as error:
        parser.error(str(error))
    if not missing:
        print("All requested universe membership dates have broker mappings.")
        return
    print(f"Unresolved {arguments.broker} mappings for {arguments.universe}:")
    for security_id, symbol, start, end in missing:
        print(f"{security_id} {symbol}: {start.isoformat()} through {end.isoformat()}")
    print(f"{len(missing)} unresolved security/date ranges.")


if __name__ == "__main__":
    main()
