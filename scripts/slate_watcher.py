#!/usr/bin/env python3
"""Monitor the ESPN scoreboard and trigger a preset when a slate finishes.

Usage examples
--------------

Watch the current slate, polling every 60 seconds, and run the weekly refresh
once every game that is currently in progress wraps up::

    python scripts/slate_watcher.py --preset weekly-refresh

Watch the slate but include games that start within the next 10 minutes::

    python scripts/slate_watcher.py --preset weekly-refresh --lead-minutes 10

Integrate with cron/Task Scheduler by invoking collect_raw_data first and then
running the watcher.  When the watcher exits it will (by default) run
``collect_raw_data.py`` once more to capture the final scores before calling the
preset via ``generate_cache_cli.py``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import requests


SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
DEFAULT_SLEEP = 60  # seconds
TRACK_STATES = {"in", "inprogress"}
FINAL_STATES = {"post"}
GAME_ANALYSES_PATH = Path("data/game_analyses.json")
ANALYSIS_CACHE_PATH = Path("data/analysis_cache.json")


def _log(msg: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")


def fetch_scoreboard(season: Optional[int], week: Optional[int]) -> Dict:
    params = {}
    if season:
        params["year"] = season
    if week:
        params["week"] = week

    try:
        resp = requests.get(SCOREBOARD_URL, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to fetch ESPN scoreboard: {exc}") from exc


def _event_start(event: Dict) -> Optional[datetime]:
    date_str = event.get("date")
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except ValueError:
        return None


def _event_state(event: Dict) -> str:
    competitions = event.get("competitions") or []
    if competitions:
        status = competitions[0].get("status", {})
        state = (status.get("type") or {}).get("state")
        if state:
            return state.lower()
    return (event.get("status", {}).get("type", {}) or {}).get("state", "").lower()


def select_tracked_events(
    events: Iterable[Dict],
    lead: timedelta,
) -> Dict[str, Dict]:
    now = datetime.now(timezone.utc)
    tracked: Dict[str, Dict] = {}
    for event in events:
        event_id = str(event.get("id"))
        if not event_id:
            continue

        state = _event_state(event)
        if state in TRACK_STATES:
            tracked[event_id] = event
            continue

        if lead > timedelta(0) and state == "pre":
            start = _event_start(event)
            if start and 0 <= (start - now) <= lead:
                tracked[event_id] = event

    return tracked


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        try:
            return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None


def load_post_game_timestamps() -> Dict[str, datetime]:
    if not GAME_ANALYSES_PATH.exists():
        return {}
    try:
        data = json.loads(GAME_ANALYSES_PATH.read_text())
    except Exception:
        return {}

    result: Dict[str, datetime] = {}
    for gid, entry in data.items():
        if not isinstance(entry, dict):
            continue
        if entry.get('analysis_type') != 'post_game':
            continue
        ts = _parse_timestamp(entry.get('timestamp'))
        if ts:
            result[str(gid)] = ts
    return result


def load_analysis_cache_timestamp() -> Optional[datetime]:
    if not ANALYSIS_CACHE_PATH.exists():
        return None
    try:
        cache = json.loads(ANALYSIS_CACHE_PATH.read_text())
        return _parse_timestamp(cache.get('timestamp'))
    except Exception:
        return None


def finals_requiring_refresh(
    events: Iterable[Dict],
    post_timestamps: Dict[str, datetime],
    cache_timestamp: Optional[datetime],
) -> Set[str]:
    requiring_refresh: Set[str] = set()
    for event in events:
        state = _event_state(event)
        if state not in FINAL_STATES:
            continue
        event_id = str(event.get('id'))
        if not event_id:
            continue
        post_ts = post_timestamps.get(event_id)
        if post_ts is None:
            requiring_refresh.add(event_id)
            continue
        if cache_timestamp is None or cache_timestamp < post_ts:
            requiring_refresh.add(event_id)
    return requiring_refresh


def slate_watch(
    season: Optional[int],
    week: Optional[int],
    sleep_seconds: int,
    lead: timedelta,
    max_wait: Optional[timedelta],
) -> Set[str]:
    start_time = datetime.now(timezone.utc)
    scoreboard = fetch_scoreboard(season, week)
    events = scoreboard.get("events", [])
    tracked = select_tracked_events(events, lead)

    if not tracked:
        _log("No games currently in progress; running immediately.")
        return set()

    def _short_name(event: Dict) -> str:
        try:
            comps = event.get("competitions") or []
            if comps:
                teams = comps[0].get("competitors", [])
                if len(teams) == 2:
                    away = teams[1]['team']['abbreviation']
                    home = teams[0]['team']['abbreviation']
                    return f"{away}@{home}"
        except Exception:
            pass
        return event.get("shortName") or str(event.get("id"))

def slate_watch(
    season: Optional[int],
    week: Optional[int],
    sleep_seconds: int,
    lead: timedelta,
    max_wait: Optional[timedelta],
) -> Set[str]:
    start_time = datetime.now(timezone.utc)
    scoreboard = fetch_scoreboard(season, week)
    events = scoreboard.get("events", [])

    post_timestamps = load_post_game_timestamps()
    cache_timestamp = load_analysis_cache_timestamp()
    finals_to_refresh = finals_requiring_refresh(events, post_timestamps, cache_timestamp)
    tracked = select_tracked_events(events, lead)

    def _short_name(event: Dict) -> str:
        try:
            comps = event.get("competitions") or []
            if comps:
                teams = comps[0].get("competitors", [])
                if len(teams) == 2:
                    away = teams[1]['team']['abbreviation']
                    home = teams[0]['team']['abbreviation']
                    return f"{away}@{home}"
        except Exception:
            pass
        return event.get("shortName") or str(event.get("id"))

    if tracked:
        _log(
            "Tracking %d game(s): %s"
            % (len(tracked), ", ".join(_short_name(e) for e in tracked.values()))
        )
    elif finals_to_refresh:
        _log(
            "No games in progress. Will process %d existing final(s)."
            % len(finals_to_refresh)
        )
        return finals_to_refresh
    else:
        _log("No games in progress and no unprocessed finals detected.")
        return set()

    while tracked:
        if max_wait and (datetime.now(timezone.utc) - start_time) > max_wait:
            _log("Reached max wait; exiting watch loop.")
            break

        time.sleep(max(1, sleep_seconds))
        scoreboard = fetch_scoreboard(season, week)
        events_map = {str(e.get("id")): e for e in scoreboard.get("events", [])}
        post_timestamps = load_post_game_timestamps()
        cache_timestamp = load_analysis_cache_timestamp()

        for event_id in list(tracked.keys()):
            event = events_map.get(event_id)
            state = _event_state(event) if event else "post"

            if state in FINAL_STATES:
                tracked.pop(event_id, None)
                matchup = event.get("shortName") if event else event_id
                _log(f"Game final: {matchup}")
                post_ts = post_timestamps.get(event_id)
                if post_ts is None or cache_timestamp is None or cache_timestamp < post_ts:
                    finals_to_refresh.add(event_id)
            elif state in TRACK_STATES:
                continue
            else:
                tracked.pop(event_id, None)

        if tracked:
            _log(
                "Still waiting on %d game(s): %s"
                % (
                    len(tracked),
                    ", ".join(
                        events_map.get(gid, tracked[gid]).get("shortName", gid)
                        for gid in tracked.keys()
                    ),
                )
            )

    return finals_to_refresh


def run_command(cmd: List[str]) -> int:
    _log("Running: %s" % " ".join(cmd))
    start = time.time()
    proc = subprocess.run(cmd, check=False)
    elapsed = time.time() - start
    _log(f"Command exited with code {proc.returncode} (elapsed {elapsed:.1f}s)")
    return proc.returncode


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Watch ESPN scoreboard and trigger preset")
    parser.add_argument("--preset", required=True, help="Preset to run via generate_cache_cli.py")
    parser.add_argument("--season", type=int, help="Optional season override (defaults to current)")
    parser.add_argument("--week", type=int, help="Optional week override (defaults to current)")
    parser.add_argument("--sleep", type=int, default=DEFAULT_SLEEP, help="Polling interval in seconds")
    parser.add_argument(
        "--lead-minutes",
        type=int,
        default=0,
        help="Also track pregame matchups starting within this many minutes",
    )
    parser.add_argument(
        "--max-wait",
        type=int,
        help="Maximum number of minutes to wait before giving up",
    )
    parser.add_argument(
        "--no-collect",
        action="store_true",
        help="Skip running collect_raw_data.py before the preset",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Watch the slate but do not run the preset",
    )
    parser.add_argument(
        "--force-run",
        action="store_true",
        help="Skip slate detection and run the preset immediately",
    )
    args = parser.parse_args(argv)

    lead = timedelta(minutes=max(0, args.lead_minutes))
    max_wait = timedelta(minutes=args.max_wait) if args.max_wait else None

    if args.force_run:
        _log("Force run enabled; skipping slate detection.")
        finals = {"forced"}
        if args.dry_run:
            _log("Dry run complete; exiting without running preset.")
            return 0
    else:
        try:
            finals = slate_watch(
                season=args.season,
                week=args.week,
                sleep_seconds=args.sleep,
                lead=lead,
                max_wait=max_wait,
            )
        except RuntimeError as exc:
            _log(str(exc))
            return 1

        if finals:
            _log(f"Detected {len(finals)} completed game(s) in this slate.")
        else:
            return 0

        if args.dry_run:
            _log("Dry run complete; exiting without running preset.")
            return 0

    if not args.no_collect:
        rc = run_command([sys.executable, "collect_raw_data.py", "--verbose"])
        if rc != 0:
            return rc

    rc = run_command([sys.executable, "generate_cache_cli.py", "--preset", args.preset])
    return rc


if __name__ == "__main__":
    sys.exit(main())
