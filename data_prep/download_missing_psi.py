"""Complete missing and 0-byte NASA PSI files in a local data directory.

Uses NASA PSI's public GEODE endpoints (`repo/investigations/{acc}`, `files/v2`,
and `studies/{acc}/download?redirect=false`) to:
  1. Download missing Combustion Science folders (`PSI-60`, `PSI-62`, `PSI-117`).
  2. Download missing experimental summary tables for `FLEX-2_PSI-68` and `FLEX_PSI-69`.
  3. Replace any local `0`-byte files that have non-zero size on NASA PSI S3.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import threading
import time
import urllib.parse

import httpx

BASE = "https://psi.nasa.gov"
HEADERS = {"User-Agent": "Mozilla/5.0 (PSI-Lens Data Completion)"}
STRIP_PREFIX_RE = re.compile(r"^PSI-\d+_[^_]+_")

FOLDER_TO_ACC = {
    "ACME-BRE_PSI-20": "PSI-20",
    "ACME-CFI-G_PSI-159": "PSI-159",
    "ACME-CLD-Flame_PSI-21": "PSI-21",
    "ACME-E-FIELD-Flames_PSI-22": "PSI-22",
    "ACME-Flame-Design_PSI-10": "PSI-10",
    "ACME-s-Flame_PSI-23": "PSI-23",
    "BASS-II_PSI-25": "PSI-25",
    "BASS_PSI-26": "PSI-26",
    "CFI_PSI-39": "PSI-39",
    "DAFT-DAFT-2_PSI-47": "PSI-47",
    "FLEX-2_PSI-68": "PSI-68",
    "FLEX_PSI-69": "PSI-69",
    "PSI-115": "PSI-115",
    "PSI-142": "PSI-142",
    "SAFFIRE-III_PSI-100": "PSI-100",
    "SAFFIRE-II_PSI-99": "PSI-99",
    "SAFFIRE-I_PSI-98": "PSI-98",
    "SAME-R_PSI-101": "PSI-101",
    "SAME_PSI-102": "PSI-102",
    "SLICE_PSI-106": "PSI-106",
    "SPICE_PSI-107": "PSI-107",
}

MISSING_STUDIES = ("PSI-60", "PSI-62", "PSI-117")
CROSS_FLEX_FALLBACK = ("PSI-68", "PSI-69", "PSI-39")


def walk_tree(nodes: list[dict]) -> list[dict]:
    """Flatten nested GEODE /files/v2 nodes into file descriptors."""
    out: list[dict] = []
    for node in nodes:
        if node.get("files"):
            out.extend(walk_tree(node["files"]))
        elif not node.get("directory", False) and node.get("key") and not node["key"].endswith("/"):
            out.append(node)
    return out


def fetch_study_files(client: httpx.Client, acc: str) -> tuple[int, list[dict]]:
    """Return (latest_version, flattened_file_list) for a NASA PSI accession."""
    meta = client.get(f"{BASE}/geode-py/ws/repo/investigations/{acc}", timeout=30.0)
    meta.raise_for_status()
    version = int(meta.json().get("version") or 1)
    tree = client.post(
        f"{BASE}/geode-py/ws/files/v2",
        json={
            "folder": f"{acc}/",
            "fileType": "study",
            "studyId": acc,
            "version": version,
            "obfuscationcode": None,
        },
        timeout=60.0,
    )
    tree.raise_for_status()
    return version, walk_tree(tree.json().get("files") or [])


def build_name_index(files: list[dict]) -> dict[str, list[dict]]:
    """Index remote files by exact name and prefix-stripped name."""
    index: dict[str, list[dict]] = {}
    for entry in files:
        if int(float(entry.get("bytes_size") or 0)) <= 0:
            continue
        raw_name = Path(entry["key"]).name
        index.setdefault(raw_name, []).append(entry)
        stripped = STRIP_PREFIX_RE.sub("", raw_name)
        if stripped != raw_name:
            index.setdefault(stripped, []).append(entry)
    return index


def pick_remote_match(candidates: list[dict], local_path: Path) -> dict:
    """Prefer exact filename matches, then category/subfolder consistency."""
    local_name = local_path.name
    for cand in candidates:
        if Path(cand["key"]).name == local_name:
            return cand
    parent = local_path.parent.name.lower()
    if parent == "pdf":
        for cand in candidates:
            key = cand["key"].lower()
            if "calibration" in key or "science" in key or "engineering" in key or "raw data" in key:
                return cand
    return candidates[0]


def fetch_presigned_url(client: httpx.Client, acc: str, entry: dict, study_version: int) -> str:
    """Resolve an S3 presigned URL from NASA PSI's study download endpoint."""
    file_param = entry.get("id") or Path(entry["key"]).name
    ver = entry.get("version") or study_version
    qs = urllib.parse.urlencode({"file": str(file_param), "version": str(ver), "redirect": "false"})
    resp = client.get(f"{BASE}/geode-py/ws/studies/{acc}/download?{qs}", timeout=30.0)
    resp.raise_for_status()
    url = resp.text.strip().strip('"')
    if not url.startswith("http"):
        raise RuntimeError(f"Unexpected download URL response for {acc}/{file_param}: {url[:120]}")
    return url


def download_to_path(
    client: httpx.Client, acc: str, entry: dict, study_version: int, dest: Path
) -> int:
    """Download one remote file atomically to `dest`, returning byte size."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.part")
    last_err: Exception | None = None
    for attempt in range(4):
        try:
            s3_url = fetch_presigned_url(client, acc, entry, study_version)
            with client.stream("GET", s3_url, timeout=120.0) as stream:
                stream.raise_for_status()
                written = 0
                with tmp.open("wb") as handle:
                    for chunk in stream.iter_bytes(chunk_size=65536):
                        if chunk:
                            handle.write(chunk)
                            written += len(chunk)
            if written <= 0:
                raise RuntimeError(f"Downloaded 0 bytes for {dest}")
            tmp.replace(dest)
            return written
        except Exception as exc:
            last_err = exc
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            time.sleep(0.6 * (attempt + 1))
    raise RuntimeError(f"Failed downloading {acc} -> {dest}: {last_err}")


def subdir_for_remote(acc: str, key: str) -> str:
    """Map a remote PSI object key into clean extension/category subdirs for new folders."""
    ext = Path(key).suffix.lower().lstrip(".")
    if ext in {"pdf", "csv", "txt", "xlsx", "xls", "zip"}:
        return ext
    rel = key.removeprefix(f"psi-data/{acc}/")
    return rel.split("/", 1)[0].lower().replace(" ", "_")


def plan_downloads(
    client: httpx.Client, data_dir: Path, include_raw_studies: bool = True
) -> tuple[list[tuple[str, int, dict, Path]], list[str]]:
    """Build concrete download jobs across missing folders, tables, and 0-byte placeholders."""
    jobs: list[tuple[str, int, dict, Path]] = []
    notes: list[str] = []
    seen_dests: set[Path] = set()
    trees: dict[str, tuple[int, list[dict], dict[str, list[dict]]]] = {}

    def ensure_tree(acc: str) -> tuple[int, list[dict], dict[str, list[dict]]]:
        if acc not in trees:
            ver, files = fetch_study_files(client, acc)
            trees[acc] = (ver, files, build_name_index(files))
        return trees[acc]

    # 1. Missing Combustion Science studies (PSI-60, PSI-62, PSI-117)
    if include_raw_studies:
        for acc in MISSING_STUDIES:
            ver, files, _ = ensure_tree(acc)
            study_dir = data_dir / acc
            for entry in files:
                if int(float(entry.get("bytes_size") or 0)) <= 0:
                    continue
                name = Path(entry["key"]).name
                dest = study_dir / subdir_for_remote(acc, entry["key"]) / name
                if dest.exists() and dest.stat().st_size > 0:
                    continue
                if dest not in seen_dests:
                    seen_dests.add(dest)
                    jobs.append((acc, ver, entry, dest))

    # 2. Missing experimental tables in FLEX-2_PSI-68 and FLEX_PSI-69
    for folder, acc in (("FLEX-2_PSI-68", "PSI-68"), ("FLEX_PSI-69", "PSI-69")):
        ver, files, _ = ensure_tree(acc)
        for entry in files:
            key_lower = entry["key"].lower()
            if "/experimental table/" in key_lower and key_lower.endswith(".csv"):
                name = Path(entry["key"]).name
                dest = data_dir / folder / "csv" / name
                if (not dest.exists() or dest.stat().st_size == 0) and dest not in seen_dests:
                    seen_dests.add(dest)
                    jobs.append((acc, ver, entry, dest))

    # 3. Replace all local 0-byte placeholder files where remote size > 0
    zero_paths = sorted(
        p for p in data_dir.rglob("*") if p.is_file() and p.stat().st_size == 0
    )
    by_folder: dict[str, list[Path]] = {}
    for p in zero_paths:
        top = p.relative_to(data_dir).parts[0]
        by_folder.setdefault(top, []).append(p)

    for top, paths in by_folder.items():
        acc = FOLDER_TO_ACC.get(top)
        if not acc:
            continue
        ver, _, index = ensure_tree(acc)
        fallback_accs = CROSS_FLEX_FALLBACK if acc in ("PSI-68", "PSI-69") else ()
        unmatched_count = 0
        for local_path in paths:
            cands = index.get(local_path.name)
            use_acc, use_ver = acc, ver
            if not cands:
                for fb_acc in fallback_accs:
                    if fb_acc == acc:
                        continue
                    fb_ver, _, fb_index = ensure_tree(fb_acc)
                    if local_path.name in fb_index:
                        cands = fb_index[local_path.name]
                        use_acc, use_ver = fb_acc, fb_ver
                        break
            if not cands:
                unmatched_count += 1
                continue
            entry = pick_remote_match(cands, local_path)
            if local_path not in seen_dests:
                seen_dests.add(local_path)
                jobs.append((use_acc, use_ver, entry, local_path))
        if unmatched_count:
            notes.append(
                f"{top}: {unmatched_count} 0-byte files had no non-empty match in public PSI tree"
            )

    return jobs, notes


def run(data_dir: Path, workers: int = 12, dry_run: bool = False) -> dict:
    """Plan and execute downloads for `data_dir`, returning completion counts."""
    with httpx.Client(headers=HEADERS, follow_redirects=True) as client:
        jobs, notes = plan_downloads(client, data_dir)

    total_bytes = sum(int(float(entry.get("bytes_size") or 0)) for _, _, entry, _ in jobs)
    print(f"Planned {len(jobs)} downloads (~{total_bytes / (1024 * 1024):.2f} MB)")
    for note in notes:
        print("Note:", note)
    if dry_run or not jobs:
        return {"planned": len(jobs), "downloaded": 0, "failed": 0, "bytes": 0, "notes": notes}

    lock = threading.Lock()
    done = 0
    downloaded_bytes = 0
    failures: list[str] = []

    thread_local = threading.local()

    def get_client() -> httpx.Client:
        cli = getattr(thread_local, "client", None)
        if cli is None:
            cli = httpx.Client(headers=HEADERS, follow_redirects=True)
            thread_local.client = cli
        return cli

    def worker(job: tuple[str, int, dict, Path]) -> tuple[Path, int]:
        acc, ver, entry, dest = job
        written = download_to_path(get_client(), acc, entry, ver, dest)
        return dest, written

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(worker, job): job for job in jobs}
        for future in as_completed(future_map):
            _, _, _, dest = future_map[future]
            try:
                _, written = future.result()
                with lock:
                    done += 1
                    downloaded_bytes += written
                    if done % 100 == 0 or done == len(jobs):
                        print(
                            f"Progress: {done}/{len(jobs)} files "
                            f"({downloaded_bytes / (1024 * 1024):.2f} MB)"
                        )
            except Exception as exc:
                with lock:
                    done += 1
                    failures.append(f"{dest}: {exc}")

    print(
        f"Finished: {len(jobs) - len(failures)} downloaded, "
        f"{len(failures)} failed, {downloaded_bytes / (1024 * 1024):.2f} MB"
    )
    for err in failures[:10]:
        print("Failure:", err)
    return {
        "planned": len(jobs),
        "downloaded": len(jobs) - len(failures),
        "failed": len(failures),
        "bytes": downloaded_bytes,
        "notes": notes,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("/home/bartek/data"))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    summary = run(args.data_dir.resolve(), workers=args.workers, dry_run=args.dry_run)
    if summary.get("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
