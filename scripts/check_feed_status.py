#!/usr/bin/env python3
"""Quickly inspect ESPN/NFL feeds to see if post-game data is ready."""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests

from raw_data_manifest import ManifestEntry, RawDataManifest

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
DEFAULT_DATASETS: Tuple[str, ...] = (
    "espn_summary",
    "espn_leaders",
    "espn_game_injuries",
    "espn_game_info",
    "espn_pickcenter",
    "espn_predictor",
    "espn_winprobability",
)
POST_REQUIRED_DEFAULT = {
    "espn_summary",
    "espn_leaders",
    "espn_game_injuries",
    "espn_game_info",
}
PRE_REQUIRED_DEFAULT = {
    "espn_summary",
    "espn_pickcenter",
    "espn_predictor",
}


@dataclass
class DatasetStatus:
    dataset: str
    present: bool
    timestamp: Optional[dt.datetime]
    required: bool

    def render(self) -> str:
        if not self.present:
            return f"{self.dataset}: missing{' *' if self.required else ''}"
        if not self.timestamp:
            return f"{self.dataset}: ready"
        return f"{self.dataset}: {self.timestamp.strftime('%H:%M:%S')}"


@dataclass
class GameSnapshot:
    game_id: str
    description: str
    state: str
    detail: str
    clock: Optional[str]
    quarter: Optional[int]
    datasets: List[DatasetStatus]
    sort_key: Optional[str]

    def ready(self) -> bool:
        if self.state != "post":
            return True
        return all(ds.present for ds in self.datasets if ds.required)

    def render(self) -> str:
        clock_info = ""
        if self.state == "in" and self.clock is not None and self.quarter is not None:
            clock_info = f" (Q{self.quarter} {self.clock})"
        header = f"{self.description} — {self.detail}{clock_info}"
        lines = [header]
        for ds in self.datasets:
            lines.append(f"  - {ds.render()}")
        return "\n".join(lines)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    default_manifest_obj = RawDataManifest.from_latest()
    parser = argparse.ArgumentParser(
        description="Check which ESPN raw snapshots are ready for the current week."
    )
    parser.add_argument("--season", type=int, help="Season year (defaults to current from ESPN)")
    parser.add_argument("--week", type=int, help="Week number (defaults to current from ESPN)")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Datasets to inspect (defaults to common ESPN summaries)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=default_manifest_obj.path if default_manifest_obj else None,
        help="Path to raw manifest JSON (defaults to data/raw/manifest/latest.json)",
    )
    parser.add_argument(
        "--require-post",
        nargs="+",
        default=list(POST_REQUIRED_DEFAULT),
        help="Datasets that must exist once a game is final (default: basic ESPN summary feeds).",
    )
    parser.add_argument(
        "--require-pre",
        nargs="+",
        default=list(PRE_REQUIRED_DEFAULT),
        help="Datasets expected for upcoming/in-progress games (default: summary and predictor feeds).",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Poll every interval seconds until post-game datasets are present.",
    )
    parser.add_argument(
        "--interval", type=int, default=60, help="Polling interval in seconds for --watch",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Optional timeout in minutes for --watch (0 = wait indefinitely)",
    )
    parser.add_argument(
        "--states",
        nargs="+",
        choices=["pre", "in", "post"],
        help="Filter games by ESPN status state (e.g., --states in post). Defaults to all.",
    )
    return parser.parse_args(argv)


def fetch_scoreboard(season: Optional[int], week: Optional[int]) -> Tuple[int, int, Dict]:
    params: Dict[str, int] = {"seasontype": 2}
    if season is not None:
        params["year"] = season
    if week is not None:
        params["week"] = week

    with requests.Session() as session:
        response = session.get(SCOREBOARD_URL, params=params, timeout=15)
        response.raise_for_status()
        payload = response.json()

    resolved_season = season or payload.get("season", {}).get("year")
    resolved_week = week or payload.get("week", {}).get("number")
    if resolved_season is None or resolved_week is None:
        raise ValueError("Unable to resolve season/week from scoreboard response")
    return int(resolved_season), int(resolved_week), payload


def find_manifest_entry(
    manifest: Optional[RawDataManifest], dataset: str, identifier: str
) -> Optional[ManifestEntry]:
    if manifest is None:
        return None
    for entry in manifest.entries(dataset):
        meta = entry.metadata or {}
        if str(meta.get("event_id")) == identifier:
            return entry
        if entry.path.stem == identifier:
            return entry
        parent = entry.path.parent.name if entry.path.parent else None
        if parent == identifier:
            return entry
    return None


def build_dataset_statuses(
    manifest: Optional[RawDataManifest],
    datasets: Iterable[str],
    game_id: str,
    state: str,
    post_required: Iterable[str],
    pre_required: Iterable[str],
) -> List[DatasetStatus]:
    required_pool = set(post_required) if state == "post" else set(pre_required)
    statuses: List[DatasetStatus] = []
    for dataset in datasets:
        entry = find_manifest_entry(manifest, dataset, game_id)
        required = dataset in required_pool
        if not entry:
            statuses.append(DatasetStatus(dataset, False, None, required))
            continue
        timestamp = None
        try:
            timestamp = dt.datetime.fromtimestamp(entry.path.stat().st_mtime)
        except OSError:
            timestamp = None
        statuses.append(DatasetStatus(dataset, True, timestamp, required))
    return statuses


def describe_competition(event: Dict) -> Tuple[str, str, Dict]:
    competition = (event.get("competitions") or [None])[0] or {}
    competitors = competition.get("competitors", []) or []
    home = next((c for c in competitors if c.get("homeAway") == "home"), None)
    away = next((c for c in competitors if c.get("homeAway") == "away"), None)
    if not home or not away:
        raise ValueError(f"Malformed competition payload for event {event.get('id')}")

    def describe(side: Dict) -> str:
        team = side.get("team", {}) or {}
        abbr = team.get("abbreviation") or team.get("shortDisplayName") or team.get("name")
        score = side.get("score")
        try:
            score_display = int(score)
        except (TypeError, ValueError):
            score_display = score
        return f"{abbr} {score_display}" if score is not None else str(abbr)

    descriptor = f"{describe(away)} @ {describe(home)}"
    status = (competition.get("status") or {}).get("type", {}) or {}
    status_wrapper = competition.get("status") or {}
    detail = status.get("shortDetail") or status.get("detail") or status.get("description") or "Unknown"
    state = status.get("state") or "unknown"
    clock = status_wrapper.get("displayClock")
    period = status_wrapper.get("period")
    return descriptor, state, {"detail": detail, "clock": clock, "period": period}


def snapshot_games(
    scoreboard: Dict,
    manifest: Optional[RawDataManifest],
    datasets: Iterable[str],
    post_required: Iterable[str],
    pre_required: Iterable[str],
    allowed_states: Optional[Iterable[str]] = None,
) -> List[GameSnapshot]:
    snapshots: List[GameSnapshot] = []
    allowed = set(allowed_states or [])
    for event in scoreboard.get("events", []) or []:
        game_id = str(event.get("id"))
        try:
            description, state, status_meta = describe_competition(event)
        except ValueError as exc:
            print(f"Skipping event {event.get('id')}: {exc}", file=sys.stderr)
            continue

        if allowed and state not in allowed:
            continue

        datasets_status = build_dataset_statuses(
            manifest,
            datasets,
            game_id,
            state,
            post_required,
            pre_required,
        )
        snapshots.append(
            GameSnapshot(
                game_id=game_id,
                description=description,
                state=state,
                detail=status_meta["detail"],
                clock=status_meta.get("clock"),
                quarter=status_meta.get("period"),
                datasets=datasets_status,
                sort_key=event.get("date"),
            )
        )
    snapshots.sort(key=lambda snap: (snap.sort_key or "", snap.game_id))
    return snapshots


def render_snapshot(
    season: int,
    week: int,
    snapshots: List[GameSnapshot],
    post_required: Iterable[str],
    pre_required: Iterable[str],
) -> str:
    post_hint = ", ".join(sorted(post_required)) or "(none)"
    pre_hint = ", ".join(sorted(pre_required)) or "(none)"
    lines = [
        f"Season {season}, Week {week}",
        f"Snapshot: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Finals require: {post_hint}",
        f"Active/upcoming expect: {pre_hint}",
        "",
        "* missing = required feed not yet captured",
        "",
    ]
    for snap in snapshots:
        lines.append(snap.render())
        lines.append("")
    return "\n".join(lines).rstrip()


def all_post_games_ready(snapshots: List[GameSnapshot]) -> bool:
    return all(snap.ready() for snap in snapshots)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        season, week, scoreboard = fetch_scoreboard(args.season, args.week)
    except requests.HTTPError as exc:
        print(f"Failed to fetch scoreboard: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Error resolving season/week: {exc}", file=sys.stderr)
        return 2

    manifest: Optional[RawDataManifest] = None
    manifest_path: Optional[Path] = args.manifest
    if manifest_path:
        resolved_path = manifest_path
    else:
        resolved_path = Path("data/raw/manifest/latest.json")
        manifest_path = resolved_path
    if resolved_path.exists():
        manifest = RawDataManifest(resolved_path)
    else:
        print(
            f"Warning: manifest {resolved_path} not found. Run collect_raw_data.py to generate raw snapshots.",
            file=sys.stderr,
        )

    datasets = args.datasets
    post_required = set(args.require_post)
    pre_required = set(args.require_pre)
    allowed_states = args.states
    end_time: Optional[dt.datetime] = None
    if args.watch and args.timeout:
        end_time = dt.datetime.now() + dt.timedelta(minutes=args.timeout)

    while True:
        snapshots = snapshot_games(
            scoreboard,
            manifest,
            datasets,
            post_required,
            pre_required,
            allowed_states,
        )
        print(render_snapshot(season, week, snapshots, post_required, pre_required))
        print("")

        if not args.watch:
            break

        if all_post_games_ready(snapshots):
            print("All post-game datasets accounted for. Exiting.")
            break

        if end_time and dt.datetime.now() >= end_time:
            print("Timeout reached before all post-game datasets were ready.")
            return 1

        time.sleep(max(args.interval, 5))
        try:
            season, week, scoreboard = fetch_scoreboard(args.season, args.week)
            if manifest_path and manifest_path.exists():
                manifest = RawDataManifest(manifest_path)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Warning: failed to refresh data: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
