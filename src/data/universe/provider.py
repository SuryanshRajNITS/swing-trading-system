"""Read-only, date-aware access to version-controlled universe reference data."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.configuration import PROJECT_ROOT


DEFAULT_REFERENCE_DIRECTORY = PROJECT_ROOT / "data" / "reference"
PIT_TRADABLE = "NIFTY50_PIT_TRADABLE"
PIT_INDEX = "NIFTY50_PIT_INDEX"
CURRENT = "NIFTY50_CURRENT"


class UniverseDataError(ValueError):
    """Reference data is malformed or internally inconsistent."""


class UniverseCoverageError(LookupError):
    """No explicit reference-data coverage exists for a requested date."""


def _parse_date(value: str, *, field: str, allow_empty: bool = False) -> date | None:
    if not value and allow_empty:
        return None
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        raise UniverseDataError(f"Invalid {field} date in universe reference data.") from None


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise UniverseDataError(f"Reference file {path.name} has no header.")
            return list(reader)
    except OSError:
        raise UniverseDataError(f"Could not read universe reference file {path.name}.") from None


@dataclass(frozen=True)
class Security:
    security_id: str
    symbol: str
    company_name: str | None
    isin: str | None
    universe_id: str
    membership_type: str
    tradable: bool


@dataclass(frozen=True)
class MembershipInterval:
    security_id: str
    universe_id: str
    effective_from: date
    effective_to: date | None
    announcement_date: date | None
    membership_type: str
    tradable: bool
    source_id: str


@dataclass(frozen=True)
class BrokerInstrument:
    security_id: str
    broker: str
    exchange: str
    tradingsymbol: str
    instrument_token: int
    effective_from: date
    effective_to: date | None
    source_id: str


@dataclass(frozen=True)
class SymbolInterval:
    security_id: str
    symbol: str
    company_name: str | None
    effective_from: date
    effective_to: date | None
    source_id: str


@dataclass(frozen=True)
class UniverseEvent:
    event_type: str
    universe_id: str
    security_id: str | None
    announcement_date: date | None
    effective_date: date
    from_symbol: str | None
    to_symbol: str | None
    source_id: str
    notes: str


class UniverseProvider:
    """Query explicit universe coverage; never substitutes one universe for another."""

    def __init__(self, reference_directory: str | Path = DEFAULT_REFERENCE_DIRECTORY) -> None:
        self.reference_directory = Path(reference_directory)
        self._securities = {row["security_id"]: row for row in _read_csv(self.reference_directory / "securities.csv")}
        self._symbols = _read_csv(self.reference_directory / "symbol_history.csv")
        self._memberships = _read_csv(self.reference_directory / "universe_membership.csv")
        self._broker = _read_csv(self.reference_directory / "broker_instrument_history.csv")
        self._coverage = _read_csv(self.reference_directory / "universe_coverage.csv")
        self._sources = _read_csv(self.reference_directory / "sources.csv")
        self._events = _read_csv(self.reference_directory / "universe_events.csv")
        self._validate()

    def _validate(self) -> None:
        source_ids = {row["source_id"] for row in self._sources}
        if len(source_ids) != len(self._sources):
            raise UniverseDataError("Source metadata contains duplicate source_id values.")
        for row in self._securities.values():
            identity_source = row.get("identity_source_id", "")
            if identity_source and identity_source not in source_ids:
                raise UniverseDataError("Security identity references an unknown source_id.")
        for row in self._memberships:
            if row["security_id"] not in self._securities:
                raise UniverseDataError("Membership row references an unknown security_id.")
            start = _parse_date(row["effective_from"], field="effective_from")
            end = _parse_date(row["effective_to"], field="effective_to", allow_empty=True)
            if end is not None and start is not None and end <= start:
                raise UniverseDataError("Membership effective_to must be later than effective_from.")
            if row["tradable"] not in {"true", "false"}:
                raise UniverseDataError("Membership tradable values must be true or false.")
            if row["tradable"] == "true" and row["membership_type"] == "DUMMY":
                raise UniverseDataError("Dummy corporate-action rows cannot be tradable.")
            if row["source_id"] not in source_ids:
                raise UniverseDataError("Membership row references an unknown source_id.")
        for row in self._coverage:
            if row["source_id"] not in source_ids:
                raise UniverseDataError("Coverage row references an unknown source_id.")
        for row in self._symbols:
            if row["source_id"] not in source_ids:
                raise UniverseDataError("Symbol history references an unknown source_id.")
        for row in self._broker:
            if row["source_id"] not in source_ids:
                raise UniverseDataError("Broker mapping references an unknown source_id.")
            try:
                token = int(row["instrument_token"])
            except (TypeError, ValueError):
                token = 0
            if token <= 0:
                raise UniverseDataError("Broker mapping has an invalid instrument_token.")
        for row in self._events:
            if row["security_id"] and row["security_id"] not in self._securities:
                raise UniverseDataError("Universe event references an unknown security_id.")
            _parse_date(row["effective_date"], field="effective_date")
            _parse_date(row["announcement_date"], field="announcement_date", allow_empty=True)
            if row["source_id"] not in source_ids:
                raise UniverseDataError("Universe event references an unknown source_id.")
        self._validate_no_overlaps(self._memberships, ("universe_id", "security_id"), "membership")
        self._validate_no_overlaps(self._symbols, ("security_id",), "symbol history")
        self._validate_no_overlaps(self._broker, ("broker", "exchange", "tradingsymbol"), "broker mapping")
        self._validate_no_overlaps(self._broker, ("security_id", "broker", "exchange"), "security broker mapping")
        self._validate_no_overlaps(self._broker, ("broker", "exchange", "instrument_token"), "broker token")
        self._validate_no_overlaps(self._coverage, ("universe_id",), "universe coverage")
        for row in self._symbols:
            if row["security_id"] not in self._securities:
                raise UniverseDataError("Symbol history references an unknown security_id.")
        for row in self._broker:
            if row["security_id"] not in self._securities:
                raise UniverseDataError("Broker mapping references an unknown security_id.")

    @staticmethod
    def _validate_no_overlaps(
        rows: list[dict[str, str]], keys: tuple[str, ...], label: str
    ) -> None:
        grouped: dict[tuple[str, ...], list[tuple[date, date | None]]] = {}
        for row in rows:
            key = tuple(row[field] for field in keys)
            start = _parse_date(row["effective_from"], field="effective_from")
            end = _parse_date(row["effective_to"], field="effective_to", allow_empty=True)
            assert start is not None
            grouped.setdefault(key, []).append((start, end))
        for intervals in grouped.values():
            intervals.sort(key=lambda interval: interval[0])
            for (_, previous_end), (next_start, _) in zip(intervals, intervals[1:]):
                if previous_end is None or next_start < previous_end:
                    raise UniverseDataError(f"Overlapping {label} date ranges.")

    @staticmethod
    def _active(row: dict[str, str], trading_date: date) -> bool:
        start = _parse_date(row["effective_from"], field="effective_from")
        end = _parse_date(row["effective_to"], field="effective_to", allow_empty=True)
        return bool(start and start <= trading_date and (end is None or trading_date < end))

    def _assert_coverage(self, trading_date: date, universe_id: str) -> None:
        for row in self._coverage:
            if row["universe_id"] == universe_id and self._active(row, trading_date):
                return
        raise UniverseCoverageError(
            f"No coverage for {universe_id} on {trading_date.isoformat()}; "
            "no fallback to another universe is performed."
        )

    def symbol_for(self, security_id: str, trading_date: date) -> str:
        matches = [
            row for row in self._symbols
            if row["security_id"] == security_id and self._active(row, trading_date)
        ]
        if len(matches) != 1:
            raise UniverseCoverageError(
                f"No unique symbol mapping for {security_id} on {trading_date.isoformat()}."
            )
        return matches[0]["symbol"]

    def latest_symbol_for(self, security_id: str) -> SymbolInterval:
        """Return the latest evidenced symbol for an identity, independent of index membership."""
        rows = self.symbol_history_over(security_id, date.min, date.max)
        if not rows:
            raise UniverseCoverageError(f"No symbol history exists for {security_id}.")
        latest_start = max(item.effective_from for item in rows)
        latest = [item for item in rows if item.effective_from == latest_start]
        if len(latest) != 1:
            raise UniverseCoverageError(f"No unique latest symbol exists for {security_id}.")
        return latest[0]

    def security_identity_for(self, security_id: str) -> tuple[str | None, str | None]:
        """Return (ISIN, identity source ID) for continuity checks."""
        row = self._securities.get(security_id)
        if row is None:
            raise UniverseCoverageError(f"No stable security identity exists for {security_id}.")
        return row.get("isin") or None, row.get("identity_source_id") or None

    def symbol_history_over(self, security_id: str, start_date: date, end_date: date) -> list[SymbolInterval]:
        """Return source-backed symbol intervals intersecting a date range."""
        if start_date > end_date:
            raise ValueError("start_date must not be later than end_date")
        result = []
        for row in self._symbols:
            if row["security_id"] != security_id:
                continue
            start = _parse_date(row["effective_from"], field="effective_from")
            end = _parse_date(row["effective_to"], field="effective_to", allow_empty=True)
            assert start is not None
            if start > end_date or (end is not None and end <= start_date):
                continue
            result.append(SymbolInterval(
                security_id=security_id, symbol=row["symbol"],
                company_name=row["company_name"] or None, effective_from=start,
                effective_to=end, source_id=row["source_id"],
            ))
        return sorted(result, key=lambda item: item.effective_from)

    def identity_events_over(
        self, security_id: str, start_date: date, end_date: date,
    ) -> list[UniverseEvent]:
        """Return explicitly recorded security identity events in a date interval."""
        if start_date > end_date:
            raise ValueError("start_date must not be later than end_date")
        events = []
        for row in self._events:
            effective = _parse_date(row["effective_date"], field="effective_date")
            assert effective is not None
            if (row["security_id"] != security_id or row["universe_id"] != "SECURITY_IDENTITY"
                    or not start_date <= effective <= end_date):
                continue
            events.append(UniverseEvent(
                event_type=row["event_type"], universe_id=row["universe_id"],
                security_id=security_id,
                announcement_date=_parse_date(row["announcement_date"], field="announcement_date", allow_empty=True),
                effective_date=effective, from_symbol=row["from_symbol"] or None,
                to_symbol=row["to_symbol"] or None, source_id=row["source_id"], notes=row["notes"],
            ))
        return sorted(events, key=lambda item: item.effective_date)

    def members_on(self, trading_date: date, universe_id: str = PIT_TRADABLE) -> list[Security]:
        self._assert_coverage(trading_date, universe_id)
        result: list[Security] = []
        for row in self._memberships:
            if row["universe_id"] != universe_id or not self._active(row, trading_date):
                continue
            if universe_id == PIT_TRADABLE and row["tradable"] != "true":
                continue
            security = self._securities[row["security_id"]]
            result.append(
                Security(
                    security_id=row["security_id"],
                    symbol=self.symbol_for(row["security_id"], trading_date),
                    company_name=self._company_name(row["security_id"], trading_date),
                    isin=security["isin"] or None,
                    universe_id=universe_id,
                    membership_type=row["membership_type"],
                    tradable=row["tradable"] == "true",
                )
            )
        return sorted(result, key=lambda security: security.symbol)

    def current_members(self, trading_date: date | None = None) -> list[Security]:
        """Query the separately curated current universe (not a PIT fallback)."""
        if trading_date is None:
            trading_date = datetime.now(ZoneInfo("Asia/Kolkata")).date()
        return self.members_on(trading_date, CURRENT)

    def members_over(self, start_date: date, end_date: date, universe_id: str = PIT_TRADABLE) -> list[MembershipInterval]:
        if start_date > end_date:
            raise ValueError("start_date must not be later than end_date")
        self._assert_coverage(start_date, universe_id)
        self._assert_coverage(end_date, universe_id)
        records = []
        for row in self._memberships:
            if row["universe_id"] != universe_id:
                continue
            start = _parse_date(row["effective_from"], field="effective_from")
            end = _parse_date(row["effective_to"], field="effective_to", allow_empty=True)
            assert start is not None
            if (end is None or end > start_date) and start <= end_date:
                records.append(
                    MembershipInterval(
                        security_id=row["security_id"],
                        universe_id=universe_id,
                        effective_from=start,
                        effective_to=end,
                        announcement_date=_parse_date(row["announcement_date"], field="announcement_date", allow_empty=True),
                        membership_type=row["membership_type"],
                        tradable=row["tradable"] == "true",
                        source_id=row["source_id"],
                    )
                )
        return sorted(records, key=lambda item: (item.effective_from, item.security_id))

    def changes_on(self, effective_date: date, universe_id: str | None = None) -> list[UniverseEvent]:
        """Return source-backed additions/removals and identity events for an effective date."""
        return [
            UniverseEvent(
                event_type=row["event_type"],
                universe_id=row["universe_id"],
                security_id=row["security_id"] or None,
                announcement_date=_parse_date(row["announcement_date"], field="announcement_date", allow_empty=True),
                effective_date=_parse_date(row["effective_date"], field="effective_date"),  # type: ignore[arg-type]
                from_symbol=row["from_symbol"] or None,
                to_symbol=row["to_symbol"] or None,
                source_id=row["source_id"],
                notes=row["notes"],
            )
            for row in self._events
            if row["effective_date"] == effective_date.isoformat()
            and (universe_id is None or row["universe_id"] in {universe_id, "NIFTY50_PIT"})
        ]

    def broker_instrument_for(
        self, security_id: str, trading_date: date, *, broker: str = "kite", exchange: str = "NSE"
    ) -> BrokerInstrument:
        matches = [
            row for row in self._broker
            if row["security_id"] == security_id and row["broker"].lower() == broker.lower()
            and row["exchange"].upper() == exchange.upper() and self._active(row, trading_date)
        ]
        if len(matches) != 1:
            raise UniverseCoverageError(
                f"No unique {broker}/{exchange} instrument mapping for {security_id} on {trading_date.isoformat()}."
            )
        row = matches[0]
        expected_symbol = self.symbol_for(security_id, trading_date)
        if row["tradingsymbol"] != expected_symbol:
            raise UniverseDataError("Broker mapping symbol does not match security symbol history.")
        return BrokerInstrument(
            security_id=security_id,
            broker=row["broker"],
            exchange=row["exchange"],
            tradingsymbol=row["tradingsymbol"],
            instrument_token=int(row["instrument_token"]),
            effective_from=_parse_date(row["effective_from"], field="effective_from"),  # type: ignore[arg-type]
            effective_to=_parse_date(row["effective_to"], field="effective_to", allow_empty=True),
            source_id=row["source_id"],
        )

    def broker_instruments_over(
        self, security_id: str, start_date: date, end_date: date,
        *, broker: str = "kite", exchange: str = "NSE",
    ) -> list[BrokerInstrument]:
        """Return explicitly evidenced broker mappings intersecting a date range.

        The returned intervals retain their reference-data effective bounds. No
        current-token backdating or symbol fallback is performed.
        """
        if start_date > end_date:
            raise ValueError("start_date must not be later than end_date")
        matches = []
        for row in self._broker:
            if (row["security_id"] != security_id or row["broker"].lower() != broker.lower()
                    or row["exchange"].upper() != exchange.upper()):
                continue
            effective_from = _parse_date(row["effective_from"], field="effective_from")
            effective_to = _parse_date(row["effective_to"], field="effective_to", allow_empty=True)
            assert effective_from is not None
            if effective_from > end_date or (effective_to is not None and effective_to <= start_date):
                continue
            # Ensure the identity record agrees at an actually covered date.
            covered_date = max(start_date, effective_from)
            if row["tradingsymbol"] != self.symbol_for(security_id, covered_date):
                raise UniverseDataError("Broker mapping symbol does not match security symbol history.")
            matches.append(BrokerInstrument(
                security_id=security_id, broker=row["broker"], exchange=row["exchange"],
                tradingsymbol=row["tradingsymbol"], instrument_token=int(row["instrument_token"]),
                effective_from=effective_from, effective_to=effective_to, source_id=row["source_id"],
            ))
        return sorted(matches, key=lambda item: item.effective_from)

    def unresolved_broker_mappings(
        self, start_date: date, end_date: date, *, broker: str = "kite", exchange: str = "NSE",
        universe_id: str = PIT_TRADABLE,
    ) -> list[tuple[str, str, date, date]]:
        """Return membership ranges not covered by a broker token mapping."""
        unresolved: list[tuple[str, str, date, date]] = []
        for interval in self.members_over(start_date, end_date, universe_id):
            if not interval.tradable:
                continue
            clipped_start = max(start_date, interval.effective_from)
            clipped_end = min(end_date, (interval.effective_to - date.resolution) if interval.effective_to else end_date)
            if clipped_start > clipped_end:
                continue
            symbol_rows = [
                row for row in self._symbols
                if row["security_id"] == interval.security_id
            ]
            for symbol_row in symbol_rows:
                sym_start = _parse_date(symbol_row["effective_from"], field="effective_from")
                sym_end = _parse_date(symbol_row["effective_to"], field="effective_to", allow_empty=True)
                assert sym_start is not None
                part_start = max(clipped_start, sym_start)
                part_end = min(clipped_end, (sym_end - date.resolution) if sym_end else clipped_end)
                if part_start > part_end:
                    continue
                mappings = [
                    row for row in self._broker
                    if row["security_id"] == interval.security_id and row["broker"].lower() == broker.lower()
                    and row["exchange"].upper() == exchange.upper()
                    and row["tradingsymbol"] == symbol_row["symbol"]
                ]
                covered: list[tuple[date, date]] = []
                for mapping in mappings:
                    map_start = _parse_date(mapping["effective_from"], field="effective_from")
                    map_end = _parse_date(mapping["effective_to"], field="effective_to", allow_empty=True)
                    assert map_start is not None
                    left = max(part_start, map_start)
                    right = min(part_end, (map_end - date.resolution) if map_end else part_end)
                    if left <= right:
                        covered.append((left, right))
                cursor = part_start
                for left, right in sorted(covered):
                    if left > cursor:
                        unresolved.append((interval.security_id, symbol_row["symbol"], cursor, left - date.resolution))
                    cursor = max(cursor, right + date.resolution)
                if cursor <= part_end:
                    unresolved.append((interval.security_id, symbol_row["symbol"], cursor, part_end))
        return _merge_unresolved(unresolved)

    def _company_name(self, security_id: str, trading_date: date) -> str | None:
        for row in self._symbols:
            if row["security_id"] == security_id and self._active(row, trading_date):
                return row["company_name"] or None
        return None


def _merge_unresolved(
    rows: list[tuple[str, str, date, date]]
) -> list[tuple[str, str, date, date]]:
    merged: list[tuple[str, str, date, date]] = []
    for security_id, symbol, start, end in sorted(rows, key=lambda row: (row[0], row[2], row[1])):
        if merged and merged[-1][0] == security_id and merged[-1][1] == symbol and merged[-1][3] + date.resolution >= start:
            previous = merged[-1]
            merged[-1] = (security_id, symbol, previous[2], max(previous[3], end))
        else:
            merged.append((security_id, symbol, start, end))
    return merged


DEFAULT_PROVIDER = UniverseProvider()
