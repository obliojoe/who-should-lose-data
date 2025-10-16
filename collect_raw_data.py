#!/usr/bin/env python3
"""Gather external data sources and store raw snapshots with a manifest."""

import argparse
import hashlib
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import nflreadpy as nfl
except ImportError as exc:  # pragma: no cover - defensive guard for missing dep
    raise SystemExit("nflreadpy is required to run collect_raw_data") from exc


LOGGER = logging.getLogger("collect_raw_data")
DEFAULT_RAW_DIR = Path("data/raw")
SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
MAX_EVENT_WORKERS = 8
MAX_TEAM_WORKERS = 8


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def write_dataframe(path: Path, frame: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    frame.to_csv(path, index=False)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_json(session: requests.Session, url: str, params: Optional[Dict] = None) -> Dict:
    response = session.get(url, params=params, timeout=15)
    response.raise_for_status()
    return response.json()


def resolve_season_week(
    session: requests.Session,
    season: Optional[int],
    week: Optional[int],
) -> Tuple[int, int, Dict]:
    params: Dict[str, int] = {"seasontype": 2}
    if season is not None:
        params["year"] = season
    if week is not None:
        params["week"] = week

    scoreboard = fetch_json(session, SCOREBOARD_URL, params)
    resolved_season = season or scoreboard.get("season", {}).get("year")
    resolved_week = week or scoreboard.get("week", {}).get("number")

    if resolved_season is None or resolved_week is None:
        raise ValueError("Unable to resolve season/week from scoreboard response")

    return int(resolved_season), int(resolved_week), scoreboard


def collect_scoreboard(
    scoreboard: Dict,
    output_dir: Path,
    season: int,
    week: int,
) -> Path:
    path = output_dir / "espn" / "scoreboard" / f"season_{season}_week_{week}.json"
    write_json(path, scoreboard)
    return path


def collect_standings(
    session: requests.Session,
    season: int,
    output_dir: Path,
) -> List[Tuple[str, Path]]:
    group_map = {"afc": 8, "nfc": 7}
    saved: List[Tuple[str, Path]] = []

    for label, group_id in group_map.items():
        url = (
            f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/"
            f"seasons/{season}/types/2/groups/{group_id}/standings/0"
        )
        try:
            payload = fetch_json(session, url)
        except requests.HTTPError as exc:
            LOGGER.warning("Failed to fetch %s standings: %s", label, exc)
            continue

        path = (
            output_dir
            / "espn"
            / "standings"
            / f"season_{season}"
            / f"{label}.json"
        )
        write_json(path, payload)
        saved.append((label, path))

    return saved


def collect_espn_event_payloads(
    session: requests.Session,
    events: Iterable[Dict],
    output_dir: Path,
    season: int,
    week: int,
) -> Tuple[List[Dict], List[str]]:
    summary_url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"
    event_list = list(events or [])
    if not event_list:
        return [], []

    base_headers = dict(session.headers)

    def process_event(event: Dict) -> Tuple[List[Dict], List[str]]:
        event_id = event.get("id")
        if not event_id:
            return [], []

        local_entries: List[Dict] = []
        local_team_ids: List[str] = []

        with requests.Session() as worker_session:
            worker_session.headers.update(base_headers)
            try:
                payload = fetch_json(worker_session, summary_url, params={"event": event_id})
            except requests.HTTPError as exc:
                LOGGER.warning("Failed to fetch summary for %s: %s", event_id, exc)
                return [], []

        game_dir = (
            output_dir
            / "espn"
            / "games"
            / f"season_{season}"
            / f"week_{week}"
            / str(event_id)
        )
        ensure_dir(game_dir)

        summary_path = game_dir / "summary.json"
        write_json(summary_path, payload)
        local_entries.append({
            "dataset": "espn_summary",
            "path": summary_path,
            "metadata": {
                "season": season,
                "week": week,
                "event_id": event_id,
            },
        })

        sections = {
            "espn_pickcenter": payload.get("pickcenter"),
            "espn_odds": payload.get("odds"),
            "espn_predictor": payload.get("predictor"),
            "espn_winprobability": payload.get("winprobability"),
            "espn_leaders": payload.get("leaders"),
            "espn_game_info": payload.get("gameInfo"),
            "espn_against_the_spread": payload.get("againstTheSpread"),
            "espn_game_injuries": payload.get("injuries"),
            "espn_broadcasts": payload.get("broadcasts"),
            "espn_last_five_games": payload.get("lastFiveGames"),
            "espn_standings_game": payload.get("standings"),
        }

        for dataset_name, section_payload in sections.items():
            if section_payload in (None, [], {}, ""):
                continue
            section_filename = dataset_name.split("_", 1)[1] + ".json"
            section_path = game_dir / section_filename
            write_json(section_path, section_payload)
            local_entries.append({
                "dataset": dataset_name,
                "path": section_path,
                "metadata": {
                    "season": season,
                    "week": week,
                    "event_id": event_id,
                    "section": dataset_name,
                },
            })

        for competition in event.get("competitions", []) or []:
            for competitor in competition.get("competitors", []) or []:
                team = competitor.get("team", {})
                team_id = team.get("id")
                if team_id is not None:
                    local_team_ids.append(str(team_id))

        return local_entries, local_team_ids

    artifact_entries: List[Dict] = []
    team_ids: List[str] = []
    max_workers = min(MAX_EVENT_WORKERS, len(event_list)) or 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_event, event) for event in event_list]
        for future in tqdm(as_completed(futures), total=len(futures), desc="ESPN summaries", unit="game"):
            try:
                entries, ids = future.result()
            except Exception as exc:  # pragma: no cover - defensive guard
                LOGGER.warning("Unhandled error processing event: %s", exc)
                continue
            artifact_entries.extend(entries)
            team_ids.extend(ids)

    return artifact_entries, team_ids


def fetch_team_injuries(session: requests.Session, team_id: str) -> Dict:
    base_url = (
        f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/teams/{team_id}/injuries"
    )
    try:
        listing = fetch_json(session, base_url)
    except requests.HTTPError as exc:
        LOGGER.warning("Failed to fetch injuries index for team %s: %s", team_id, exc)
        return {"team_id": team_id, "items": []}

    injuries: List[Dict] = []
    athlete_cache: Dict[str, Dict] = {}
    for item in listing.get("items", []) or []:
        ref = item.get("$ref")
        if not ref:
            continue
        try:
            detail = fetch_json(session, ref)
        except requests.HTTPError as exc:
            LOGGER.debug("Failed to fetch injury detail for team %s: %s", team_id, exc)
            continue

        athlete_ref = detail.get("athlete", {}).get("$ref")
        if athlete_ref:
            athlete = athlete_cache.get(athlete_ref)
            if athlete is None:
                try:
                    athlete = fetch_json(session, athlete_ref)
                except requests.HTTPError:
                    athlete = None
                athlete_cache[athlete_ref] = athlete
            if athlete:
                detail["athlete"] = {
                    "id": athlete.get("id"),
                    "fullName": athlete.get("fullName"),
                    "displayName": athlete.get("displayName"),
                    "position": athlete.get("position", {}).get("abbreviation"),
                }

        injuries.append(detail)

    return {
        "team_id": team_id,
        "items": injuries,
        "count": len(injuries),
        "source": base_url,
    }


def fetch_team_depthchart(session: requests.Session, season: int, team_id: str) -> Dict:
    base_url = (
        f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/"
        f"seasons/{season}/teams/{team_id}/depthcharts"
    )
    try:
        listing = fetch_json(session, base_url)
    except requests.HTTPError as exc:
        LOGGER.warning("Failed to fetch depthchart index for team %s: %s", team_id, exc)
        return {"team_id": team_id, "items": [], "count": 0, "source": base_url}

    results: List[Dict] = []
    athlete_cache: Dict[str, Dict] = {}

    for item in listing.get("items", []) or []:
        depth_name = item.get("name") or "unknown"
        positions = item.get("positions", {}) or {}

        depth_entry: Dict[str, Dict] = {"name": depth_name, "positions": {}}
        for pos_key, pos_data in positions.items():
            position_info = pos_data.get("position", {})
            athletes_info = []
            for athlete_entry in pos_data.get("athletes", []) or []:
                athlete_ref = athlete_entry.get("athlete", {}).get("$ref")
                athlete_detail = None
                if athlete_ref:
                    athlete_detail = athlete_cache.get(athlete_ref)
                    if athlete_detail is None:
                        try:
                            athlete_detail = fetch_json(session, athlete_ref)
                        except requests.HTTPError:
                            athlete_detail = None
                        athlete_cache[athlete_ref] = athlete_detail

                if athlete_detail:
                    athletes_info.append(
                        {
                            "rank": athlete_entry.get("rank"),
                            "slot": athlete_entry.get("slot"),
                            "id": athlete_detail.get("id"),
                            "fullName": athlete_detail.get("fullName"),
                            "displayName": athlete_detail.get("displayName"),
                            "position": athlete_detail.get("position", {}).get("abbreviation"),
                            "jersey": athlete_detail.get("jersey"),
                            "status": athlete_detail.get("status"),
                        }
                    )
                else:
                    athletes_info.append(
                        {
                            "rank": athlete_entry.get("rank"),
                            "slot": athlete_entry.get("slot"),
                            "athlete_ref": athlete_ref,
                        }
                    )

            depth_entry["positions"][pos_key] = {
                "position": {
                    "id": position_info.get("id"),
                    "name": position_info.get("name"),
                    "displayName": position_info.get("displayName"),
                    "abbreviation": position_info.get("abbreviation"),
                },
                "athletes": athletes_info,
            }

        results.append(depth_entry)

    return {
        "team_id": team_id,
        "items": results,
        "count": len(results),
        "source": base_url,
    }


def collect_team_endpoints(
    session: requests.Session,
    team_ids: Iterable[str],
    team_abbr_map: Dict[str, str],
    output_dir: Path,
    season: int,
    week: int,
) -> List[Dict]:
    unique_ids = sorted({tid for tid in team_ids if tid})
    if not unique_ids:
        return []

    base_headers = dict(session.headers)

    def process_team(team_id: str) -> List[Dict]:
        entries: List[Dict] = []
        identifier = str(team_id)
        abbr = team_abbr_map.get(identifier)
        filename = f"{identifier}-{abbr}.json" if abbr else f"{identifier}.json"

        with requests.Session() as worker_session:
            worker_session.headers.update(base_headers)
            for endpoint_name in ("injuries", "depthchart", "news"):
                try:
                    if endpoint_name == "injuries":
                        payload = fetch_team_injuries(worker_session, team_id)
                    elif endpoint_name == "depthchart":
                        payload = fetch_team_depthchart(worker_session, season, team_id)
                    else:
                        url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/news"
                        payload = fetch_json(worker_session, url, params={"team": team_id, "limit": 20})
                except requests.HTTPError as exc:
                    LOGGER.warning("Failed to fetch %s for team %s: %s", endpoint_name, team_id, exc)
                    continue

                path = (
                    output_dir
                    / "espn"
                    / endpoint_name
                    / f"season_{season}_week_{week}"
                    / filename
                )
                write_json(path, payload)
                entries.append({
                    "dataset": f"espn_{endpoint_name}",
                    "path": path,
                    "metadata": {
                        "team_id": team_id,
                        "team_abbr": abbr,
                        "season": season,
                        "week": week,
                    },
                })

        return entries

    results: List[Dict] = []
    max_workers = min(MAX_TEAM_WORKERS, len(unique_ids)) or 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_team, team_id) for team_id in unique_ids]
        for future in tqdm(as_completed(futures), total=len(futures), desc="ESPN team data", unit="team"):
            try:
                entries = future.result()
            except Exception as exc:  # pragma: no cover - defensive guard
                LOGGER.warning("Unhandled error processing team endpoint: %s", exc)
                continue
            results.extend(entries)

    return results


def collect_sagarin(output_dir: Path, timestamp: str) -> Optional[Path]:
    url = "http://sagarin.com/sports/nflsend.htm"
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
    except requests.RequestException as exc:
        LOGGER.warning("Unable to fetch Sagarin ratings page: %s", exc)
        return None

    path = output_dir / "sagarin" / "html" / f"raw_{timestamp}.html"
    ensure_dir(path.parent)
    path.write_text(response.text, encoding="utf-8")
    return path


def collect_nflreadpy_datasets(
    season: int,
    week: int,
    output_dir: Path,
) -> List[Tuple[str, Path, int]]:
    dataset_plan: List[Dict] = [
        {"name": "schedules", "loader": nfl.load_schedules, "filter_week": False},
        {"name": "team_stats", "loader": nfl.load_team_stats, "filter_week": True},
        {"name": "pbp", "loader": nfl.load_pbp, "filter_week": True},
        {"name": "player_stats", "loader": nfl.load_player_stats, "filter_week": True},
        {"name": "snap_counts", "loader": nfl.load_snap_counts, "filter_week": True},
        {"name": "nextgen_stats", "loader": nfl.load_nextgen_stats, "filter_week": True},
        {"name": "ff_opportunity", "loader": nfl.load_ff_opportunity, "filter_week": True},
        {"name": "depth_charts", "loader": nfl.load_depth_charts, "filter_week": True},
        {"name": "rosters_weekly", "loader": nfl.load_rosters_weekly, "filter_week": True},
    ]

    results: List[Tuple[str, Path, int]] = []

    for cfg in tqdm(dataset_plan, desc="nflreadpy datasets", unit="dataset"):
        dataset_name = cfg["name"]
        loader = cfg["loader"]
        filter_by_week = cfg.get("filter_week", False)
        min_season = cfg.get("min_season")
        max_season = cfg.get("max_season")

        if min_season is not None and season < min_season:
            LOGGER.info("Skipping %s: season %s < %s", dataset_name, season, min_season)
            continue
        if max_season is not None and season > max_season:
            LOGGER.info("Skipping %s: season %s > %s", dataset_name, season, max_season)
            continue

        try:
            frame = loader([season]).to_pandas()
        except Exception as exc:  # pragma: no cover - upstream data variability
            LOGGER.warning("Failed to load %s: %s", dataset_name, exc)
            continue

        if filter_by_week and "week" in frame.columns:
            frame = frame[frame["week"] == week] if not frame.empty else frame

        path = (
            output_dir
            / "nflreadpy"
            / dataset_name
            / f"season_{season}_week_{week}.csv"
        )
        write_dataframe(path, frame)
        results.append((dataset_name, path, len(frame)))

    return results


def build_manifest(
    output_dir: Path,
    season: int,
    week: int,
    timestamp: str,
    artifacts: List[Dict],
) -> Path:
    manifest = {
        "season": season,
        "week": week,
        "generated_at": timestamp,
        "artifacts": artifacts,
    }

    manifest_dir = output_dir / "manifest"
    ensure_dir(manifest_dir)
    manifest_path = manifest_dir / f"season_{season}_week_{week}_{timestamp}.json"
    write_json(manifest_path, manifest)

    latest_path = manifest_dir / "latest.json"
    write_json(latest_path, manifest)

    return manifest_path


def load_all_team_ids() -> Tuple[List[str], Dict[str, str]]:
    teams_path = Path("data/teams.csv")
    if not teams_path.exists():
        LOGGER.warning("teams.csv not found; team-level ESPN endpoints will be limited to active games")
        return [], {}

    teams_df = pd.read_csv(teams_path)
    if "espn_api_id" not in teams_df.columns:
        LOGGER.warning("espn_api_id missing from teams.csv; cannot fetch full team endpoints")
        return [], {}

    ids: List[str] = []
    mapping: Dict[str, str] = {}
    for _, row in teams_df.dropna(subset=["espn_api_id"]).iterrows():
        identifier = str(int(row["espn_api_id"]))
        ids.append(identifier)
        mapping[identifier] = row.get("team_abbr", "")

    return ids, mapping


def collect_single_week(
    session: requests.Session,
    output_dir: Path,
    season: int,
    week: int,
    timestamp: Optional[str] = None,
) -> Path:
    start_time = datetime.utcnow()
    timestamp = timestamp or datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    params = {"seasontype": 2, "year": season, "week": week}
    scoreboard = fetch_json(session, SCOREBOARD_URL, params)

    artifacts: List[Dict] = []

    scoreboard_path = collect_scoreboard(scoreboard, output_dir, season, week)
    artifacts.append(
        {
            "dataset": "espn_scoreboard",
            "source": "espn",
            "path": str(scoreboard_path),
            "sha256": hash_file(scoreboard_path),
            "metadata": {
                "season": season,
                "week": week,
                "events": len(scoreboard.get("events", [])),
            },
        }
    )

    event_artifacts, competitor_team_ids = collect_espn_event_payloads(
        session,
        scoreboard.get("events", []),
        output_dir,
        season,
        week,
    )
    for entry in event_artifacts:
        path = entry["path"]
        metadata = entry.get("metadata", {}).copy()
        metadata.setdefault("season", season)
        metadata.setdefault("week", week)
        artifacts.append(
            {
                "dataset": entry["dataset"],
                "source": "espn",
                "path": str(path),
                "sha256": hash_file(path),
                "metadata": metadata,
            }
        )

    combined_team_ids = set(competitor_team_ids)
    all_team_ids, team_abbr_map = load_all_team_ids()
    combined_team_ids.update(all_team_ids)
    team_entries = collect_team_endpoints(session, combined_team_ids, team_abbr_map, output_dir, season, week)
    for entry in team_entries:
        path = entry["path"]
        metadata = entry.get("metadata", {}).copy()
        metadata.setdefault("season", season)
        metadata.setdefault("week", week)
        artifacts.append(
            {
                "dataset": entry["dataset"],
                "source": "espn",
                "path": str(path),
                "sha256": hash_file(path),
                "metadata": metadata,
            }
        )

    standings_paths = collect_standings(session, season, output_dir)
    for label, path in standings_paths:
        artifacts.append(
            {
                "dataset": f"espn_standings_{label}",
                "source": "espn",
                "path": str(path),
                "sha256": hash_file(path),
                "metadata": {},
            }
        )

    nflreadpy_items = collect_nflreadpy_datasets(season, week, output_dir)
    for dataset_name, path, record_count in nflreadpy_items:
        artifacts.append(
            {
                "dataset": f"nflreadpy_{dataset_name}",
                "source": "nflreadpy",
                "path": str(path),
                "sha256": hash_file(path),
                "metadata": {"records": record_count},
            }
        )

    sagarin_path = collect_sagarin(output_dir, timestamp)
    if sagarin_path:
        artifacts.append(
            {
                "dataset": "sagarin_html",
                "source": "sagarin.com",
                "path": str(sagarin_path),
                "sha256": hash_file(sagarin_path),
                "metadata": {},
            }
        )

    manifest_path = build_manifest(output_dir, season, week, timestamp, artifacts)
    elapsed = datetime.utcnow() - start_time
    LOGGER.info(
        "Week %s collected in %.1fs. Manifest written to %s",
        week,
        elapsed.total_seconds(),
        manifest_path,
    )
    return manifest_path


def collect_raw_data(args: argparse.Namespace) -> Path:
    configure_logging(args.verbose)

    output_dir = Path(args.output_dir or DEFAULT_RAW_DIR)
    ensure_dir(output_dir)

    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (compatible; collect_raw_data/1.0; "
            "+https://github.com/who-should-lose/who-should-lose-data)"
        )
    })

    if args.start_week or args.end_week:
        if args.week:
            raise ValueError("Use either --week or --start-week/--end-week, not both")

        base_season = args.season
        if base_season is None:
            base_season, _, _ = resolve_season_week(session, None, None)

        start_week = args.start_week or 1
        end_week = args.end_week or start_week
        if start_week > end_week:
            raise ValueError("--start-week must be <= --end-week")

        total_start = datetime.utcnow()
        latest_manifest_path = None
        for week in range(start_week, end_week + 1):
            LOGGER.info("Collecting week %s", week)
            latest_manifest_path = collect_single_week(
                session=session,
                output_dir=output_dir,
                season=base_season,
                week=week,
            )

        if latest_manifest_path is None:
            raise RuntimeError("No manifest generated during backfill")

        elapsed = datetime.utcnow() - total_start
        LOGGER.info(
            "Finished backfill (%s-%s) in %.1fs",
            start_week,
            end_week,
            elapsed.total_seconds(),
        )
        return latest_manifest_path

    season, week, _ = resolve_season_week(session, args.season, args.week)
    return collect_single_week(
        session=session,
        output_dir=output_dir,
        season=season,
        week=week,
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect raw data snapshots")
    parser.add_argument("--season", type=int, help="NFL season year")
    parser.add_argument("--week", type=int, help="NFL week number")
    parser.add_argument("--start-week", type=int, help="First week to collect (inclusive)")
    parser.add_argument("--end-week", type=int, help="Last week to collect (inclusive)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_RAW_DIR),
        help="Directory to store raw artifacts",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        collect_raw_data(args)
    except Exception as exc:  # pragma: no cover - top-level guard
        LOGGER.error("Raw data collection failed: %s", exc, exc_info=args.verbose)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
