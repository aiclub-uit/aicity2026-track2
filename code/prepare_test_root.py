"""Build a unified public-test data root that mounts the AICC26 Track-2 test
set under ``test_root/data/`` in a layout run4.py understands.

Sources combined:
  videos + per-scenario caption metadata   (inference/WTS_DATASET_PUBLIC_TEST/)
  bbox + gathered VQA JSON                 (external_data/.../EXTERNAL_WTS_DATASET_TEST/)

Three video sub-types coexist in the test set:
  - WTS standard scenarios   (multi-camera, e.g. *_Camera1_0.mp4 or *_192.168.0.X_X.mp4)
  - WTS normal_trimmed       (single overhead camera; under public/normal_trimmed/<scen>/...)
  - BDD_PC_5K external       (single videoNNNN.mp4 under external/BDD_PC_5K/videos/)

Outputs at DEST_DATA = /workspace/AICC/test_root/data, mounted AS the
``val`` split so run4's hard-coded split paths Just Work:

  videos/val/<scen>/<view>/<vid>.mp4                    (symlinks)
  annotations/caption/val/<scen>/<view>/<scen>_caption.json   (DICT with placeholder
                                                                caption_pedestrian/_vehicle
                                                                so run4 builds samples
                                                                instead of dropping them)
  annotations/vqa/val/<scen>/<view>/<scen>.json         (LIST[1] with merged event_phase
                                                          + placeholder correct='a')
  annotations/bbox_annotated/{ped,veh}/val/<scen>/<view>/<base>_bbox.json
                                                         (real or synth-empty)

Also writes ``test_root/vqa_uuid_map.json`` mapping
``(scenario, phase_label, q_idx)`` → official UUID — used post-eval to swap
run4's internal IDs back to the AICC challenge UUIDs in subtask2_vqa.json.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG: source locations
# ---------------------------------------------------------------------------
EXT_ROOT = Path(
    "/workspace/AICC/external_data/EXTERNAL_WTS_DATASET_TEST-20260605T103846Z-3-002/"
    "EXTERNAL_WTS_DATASET_TEST"
)
INF_ROOT = Path("/workspace/AICC/inference/WTS_DATASET_PUBLIC_TEST")

WTS_BBOX_ROOT = EXT_ROOT / "SubTask1-Caption" / "WTS_DATASET_PUBLIC_TEST_BBOX"
BDD_BBOX_ROOT = WTS_BBOX_ROOT / "external" / "BDD_PC_5K"
VQA_GATHERED = EXT_ROOT / "SubTask2-VQA" / "WTS_VQA_PUBLIC_TEST.json"

WTS_VIDEOS_ROOT = INF_ROOT / "videos" / "test" / "public"          # incl. normal_trimmed/
BDD_VIDEOS_ROOT = INF_ROOT / "external" / "BDD_PC_5K" / "videos" / "test" / "public"

WTS_CAPTION_ROOT = INF_ROOT / "annotations" / "caption" / "test" / "public_challenge"
BDD_CAPTION_ROOT = INF_ROOT / "external" / "BDD_PC_5K" / "annotations" / "caption" / "test" / "public_challenge"

# ---------------------------------------------------------------------------
# CONFIG: destination
# ---------------------------------------------------------------------------
DEST_ROOT = Path("/workspace/AICC/test_root")
DEST_DATA = DEST_ROOT / "data"
SPLIT = "val"  # mount as val so run4 picks up without code changes

# String → numeric phase label mapping (matches run4.PHASE_STR_TO_NUM).
PHASE_STR_TO_NUM = {
    "prerecognition": "0",
    "recognition": "1",
    "judgement": "2",
    "action": "3",
    "avoidance": "4",
}


def normalize_phase_label(label) -> str:
    s = str(label).strip().lower()
    return PHASE_STR_TO_NUM.get(s, s)


def vlog(msg):
    print(f"[prepare] {msg}", flush=True)


def relink(src: Path, dst: Path):
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)
    return True


# ---------------------------------------------------------------------------
# Video classification
# ---------------------------------------------------------------------------
WTS_STD_RE = re.compile(
    r"^(?P<scen>.+?)(?:_Camera\d+_\d+|_\d+\.\d+\.\d+\.\d+_\d+(?:_\d+)?)\.mp4$"
)
WTS_VEH_RE = re.compile(r"^(?P<scen>.+?)_vehicle_view\.mp4$")
WTS_NORM_RE = re.compile(r"^(?P<scen>\d{8}_\d{6}_normal_.+_event_\d+)\.mp4$")
BDD_RE = re.compile(r"^(?P<scen>video\d+)\.mp4$")


def classify_video(name: str):
    if WTS_VEH_RE.match(name):
        return ("wts", WTS_VEH_RE.match(name).group("scen"), "vehicle_view")
    if WTS_NORM_RE.match(name):
        return ("wts_normal", WTS_NORM_RE.match(name).group("scen"), "overhead_view")
    if WTS_STD_RE.match(name):
        return ("wts", WTS_STD_RE.match(name).group("scen"), "overhead_view")
    if BDD_RE.match(name):
        return ("bdd", BDD_RE.match(name).group("scen"), "overhead_view")
    return ("unknown", name[:-4] if name.endswith(".mp4") else name, "overhead_view")


def resolve_video_src(kind: str, scen: str, view: str, name: str) -> Path | None:
    if kind == "wts":
        p = WTS_VIDEOS_ROOT / scen / view / name
        return p if p.exists() else None
    if kind == "wts_normal":
        p = WTS_VIDEOS_ROOT / "normal_trimmed" / scen / view / name
        return p if p.exists() else None
    if kind == "bdd":
        p = BDD_VIDEOS_ROOT / name
        return p if p.exists() else None
    return None


def resolve_bbox_src(kind: str, scen: str, view: str, name: str, *, vehicle: bool) -> Path | None:
    base = name[:-4]
    target = "vehicle" if vehicle else "pedestrian"
    if kind == "wts":
        for kind_dir in ("bbox_annotated", "bbox_generated"):
            p = (
                WTS_BBOX_ROOT / "annotations" / kind_dir / target /
                "test" / "public" / scen / view / f"{base}_bbox.json"
            )
            if p.exists():
                return p
        return None
    if kind == "wts_normal":
        for kind_dir in ("bbox_annotated", "bbox_generated"):
            p = (
                WTS_BBOX_ROOT / "annotations" / kind_dir / target /
                "test" / "public" / "normal_trimmed" / scen / view / f"{base}_bbox.json"
            )
            if p.exists():
                return p
        return None
    if kind == "bdd":
        for kind_dir in ("bbox_annotated", "bbox_generated"):
            p = (
                BDD_BBOX_ROOT / "annotations" / kind_dir /
                "test" / "public" / f"{base}_bbox.json"
            )
            if p.exists():
                return p
        return None
    return None


def resolve_real_caption_src(kind: str, scen: str, view: str) -> Path | None:
    """Return the path of the official caption metadata for this (scen, view), if any."""
    if kind == "wts":
        p = WTS_CAPTION_ROOT / scen / view / f"{scen}_caption.json"
        return p if p.exists() else None
    if kind == "wts_normal":
        p = WTS_CAPTION_ROOT / "normal_trimmed" / scen / view / f"{scen}_caption.json"
        return p if p.exists() else None
    if kind == "bdd":
        # BDD caption files live FLAT under public_challenge/ (no scen subdir).
        p = BDD_CAPTION_ROOT / f"{scen}_caption.json"
        return p if p.exists() else None
    return None


# ---------------------------------------------------------------------------
# Caption GT placeholder — run4 drops phases whose caption_pedestrian /
# caption_vehicle are empty. For the test set there are no ground-truth
# captions, so we inject a non-empty placeholder. Run4 reads the prompt's
# image paths regardless of GT text; the eval pipeline overwrites the
# placeholder with the model's actual prediction in subtask1_captioning.json.
# ---------------------------------------------------------------------------
PLACEHOLDER_CAPTION = "_"


def synthesize_caption_dict(*, kind: str, scen: str, view: str,
                            videos: list[str], event_phase_source: list[dict]) -> dict:
    """Return a SynWTS-shaped caption dict with run4-acceptable fields."""
    phases = []
    seen_labels: set[str] = set()
    for ph in event_phase_source:
        label_raw = (ph.get("labels") or [""])[0]
        label = normalize_phase_label(label_raw)
        if label in seen_labels:
            continue
        seen_labels.add(label)
        phases.append({
            "labels": [label],
            "start_time": ph.get("start_time", "0"),
            "end_time": ph.get("end_time", "0"),
            "caption_pedestrian": PLACEHOLDER_CAPTION,
            "caption_vehicle": PLACEHOLDER_CAPTION,
        })
    phases.sort(key=lambda p: p["labels"][0])
    out: dict = {"id": 0, "event_phase": phases}
    if view == "vehicle_view":
        out["vehicle_view"] = videos[0] if videos else ""
    else:
        out["overhead_videos"] = sorted(videos)
    return out


def build_caption_dict_from_real(real_cap: dict, kind: str, scen: str, view: str,
                                 videos: list[str]) -> dict:
    """Take official caption metadata and:
      - normalise phase labels to numeric strings
      - inject placeholder caption_pedestrian/_vehicle so run4 keeps the phase
      - add overhead_videos / vehicle_view if missing (BDD files don't carry it)
    """
    phases = []
    seen_labels: set[str] = set()
    for ph in real_cap.get("event_phase", []):
        label_raw = (ph.get("labels") or [""])[0]
        label = normalize_phase_label(label_raw)
        if label in seen_labels:
            continue
        seen_labels.add(label)
        phases.append({
            "labels": [label],
            "start_time": ph.get("start_time", "0"),
            "end_time": ph.get("end_time", "0"),
            "caption_pedestrian": (ph.get("caption_pedestrian") or PLACEHOLDER_CAPTION),
            "caption_vehicle":    (ph.get("caption_vehicle")    or PLACEHOLDER_CAPTION),
        })
    phases.sort(key=lambda p: p["labels"][0])
    out: dict = {
        "id": real_cap.get("id", 0),
        "event_phase": phases,
    }
    # carry over overhead_videos when present, else derive from videos arg
    listed = real_cap.get("overhead_videos") or real_cap.get("overhead_video")
    if view == "vehicle_view":
        # vehicle_view real files sometimes omit the key — fall back to videos arg
        out["vehicle_view"] = real_cap.get("vehicle_view") or (videos[0] if videos else "")
    else:
        if not listed:
            listed = videos or [f"{scen}.mp4"]
        out["overhead_videos"] = sorted(set(listed))
    return out


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------
def main():
    vlog(f"source EXT     : {EXT_ROOT}")
    vlog(f"source INF     : {INF_ROOT}")
    vlog(f"dest data root : {DEST_DATA}")

    if not VQA_GATHERED.is_file():
        sys.exit(f"[error] missing VQA gathered JSON: {VQA_GATHERED}")

    # ============================================================
    # 1) Discover scenario coverage from EVERY source
    # ============================================================
    # groups: key = (kind, scen, view) → dict with videos[], vqa_phases[],
    # has_caption_meta_source flag.
    groups: dict[tuple[str, str, str], dict] = {}

    def ensure_group(kind, scen, view):
        key = (kind, scen, view)
        if key not in groups:
            groups[key] = {
                "kind": kind, "scen": scen, "view": view,
                "videos": set(),
                # VQA event_phase blocks merged across all source entries
                # (one block per unique phase label, conversations appended)
                "vqa_phases_by_label": {},
                "vqa_timing_by_label": {},  # first-seen start/end_time
            }
        return groups[key]

    # 1a) Walk WTS standard caption test scenarios
    n_wts_caption = 0
    if WTS_CAPTION_ROOT.is_dir():
        for scen_dir in WTS_CAPTION_ROOT.iterdir():
            if not scen_dir.is_dir() or scen_dir.name == "normal_trimmed":
                continue
            scen = scen_dir.name
            for view_dir in scen_dir.iterdir():
                if not view_dir.is_dir():
                    continue
                view = view_dir.name
                cap_file = view_dir / f"{scen}_caption.json"
                if not cap_file.exists():
                    continue
                g = ensure_group("wts", scen, view)
                try:
                    cap = json.load(open(cap_file))
                except Exception:
                    cap = {}
                # Add videos found in caption JSON (overhead_videos or vehicle_view)
                if view == "vehicle_view":
                    vv = cap.get("vehicle_view")
                    if vv:
                        g["videos"].add(vv)
                else:
                    for v in (cap.get("overhead_videos") or []):
                        g["videos"].add(v)
                n_wts_caption += 1

    # 1b) Walk WTS normal_trimmed caption test scenarios
    n_wts_norm_caption = 0
    norm_root = WTS_CAPTION_ROOT / "normal_trimmed"
    if norm_root.is_dir():
        for scen_dir in norm_root.iterdir():
            if not scen_dir.is_dir():
                continue
            scen = scen_dir.name
            for view_dir in scen_dir.iterdir():
                if not view_dir.is_dir():
                    continue
                view = view_dir.name
                cap_file = view_dir / f"{scen}_caption.json"
                if not cap_file.exists():
                    continue
                g = ensure_group("wts_normal", scen, view)
                try:
                    cap = json.load(open(cap_file))
                except Exception:
                    cap = {}
                if view == "vehicle_view":
                    vv = cap.get("vehicle_view")
                    if vv:
                        g["videos"].add(vv)
                else:
                    listed = cap.get("overhead_videos") or []
                    if not listed:
                        listed = [f"{scen}.mp4"]
                    g["videos"].update(listed)
                n_wts_norm_caption += 1

    # 1c) Walk BDD caption test (flat files)
    n_bdd_caption = 0
    if BDD_CAPTION_ROOT.is_dir():
        for cap_file in BDD_CAPTION_ROOT.glob("video*_caption.json"):
            scen = cap_file.name.replace("_caption.json", "")
            g = ensure_group("bdd", scen, "overhead_view")
            g["videos"].add(f"{scen}.mp4")
            n_bdd_caption += 1

    vlog(f"caption sources scanned:"
         f"  WTS={n_wts_caption}  WTS_norm={n_wts_norm_caption}  BDD={n_bdd_caption}")

    # 1d) Walk VQA gathered JSON — add scenarios + merge phases per (kind, scen, view)
    gathered = json.load(open(VQA_GATHERED))
    vlog(f"loaded {len(gathered)} entries from WTS_VQA_PUBLIC_TEST.json")
    # Process event_phase (Type-A) entries FIRST so their real per-phase timing is
    # registered before Type-B (no-phase) entries possibly create the same label.
    gathered.sort(key=lambda e: 0 if (e.get("event_phase")) else 1)
    vqa_skipped = 0
    n_typeb = 0
    for entry in gathered:
        videos = entry.get("videos") or []
        phases = entry.get("event_phase") or []
        # Type-B entries: conversations live at the ITEM level with NO event_phase
        # (scene-level / environment questions, e.g. weather/clothing/road). These were
        # previously DROPPED by the `not phases` guard below -> 7,308 missing UUIDs
        # (37% of the 19,624 official test set!). Wrap them in ONE synthetic phase
        # ("2"=judgement, valid for to_phase_num) so they flow through the normal VQA
        # machinery and get answered. start/end span the whole clip -> middle frame,
        # which is fine for scene-level questions.
        if not phases and entry.get("conversations"):
            phases = [{
                "labels": ["2"],
                "start_time": "0", "end_time": "9999",
                "conversations": entry["conversations"],
            }]
            n_typeb += 1
        if not videos or not phases:
            continue
        kind, scen, view = classify_video(videos[0])
        if kind == "unknown":
            vqa_skipped += 1
            continue
        g = ensure_group(kind, scen, view)
        for v in videos:
            g["videos"].add(v)
        # Merge phases by label
        for ph in phases:
            label = normalize_phase_label((ph.get("labels") or [""])[0])
            if label not in g["vqa_phases_by_label"]:
                g["vqa_phases_by_label"][label] = []
                g["vqa_timing_by_label"][label] = (
                    ph.get("start_time", "0"), ph.get("end_time", "0"),
                )
            # Append conversations to the merged phase block
            for conv in ph.get("conversations", []):
                conv = dict(conv)
                # Placeholder so run4._vqa_sample doesn't drop it.
                if not str(conv.get("correct", "")).strip():
                    conv["correct"] = "a"
                g["vqa_phases_by_label"][label].append(conv)
    vlog(f"VQA entries skipped (unknown video pattern): {vqa_skipped}")
    vlog(f"Type-B entries (item-level conversations, no event_phase): {n_typeb}")

    by_kind = defaultdict(int)
    for (k, _, _) in groups:
        by_kind[k] += 1
    for k in sorted(by_kind):
        vlog(f"  groups[{k:10s}] = {by_kind[k]}")
    vlog(f"  TOTAL distinct (kind, scen, view) groups = {len(groups)}")

    # ============================================================
    # 2) For each group, materialise data + collect uuid_map entries
    # ============================================================
    # uuid_map[(scenario, phase_label, q_idx)] = official_uuid
    # (q_idx is the index within the merged phase block in our test root)
    uuid_map: dict[str, str] = {}

    n_vid_ok = n_vid_miss = 0
    n_bbox_real_ped = n_bbox_real_veh = 0
    n_bbox_synth = 0
    n_cap_written = n_vqa_written = 0
    n_groups_with_vqa = n_groups_caption_only = 0

    for key, g in groups.items():
        kind, scen, view = key
        videos_in_group = sorted(g["videos"])
        if not videos_in_group:
            # Some groups may have NO video resolved — skip
            continue

        # 2a) Symlink videos
        for v in videos_in_group:
            src = resolve_video_src(kind, scen, view, v)
            dst = DEST_DATA / "videos" / SPLIT / scen / view / v
            if src is None:
                n_vid_miss += 1
                continue
            relink(src, dst)
            n_vid_ok += 1

            # 2b) Symlink bbox (or synth empty so run4 keeps the candidate)
            base = v[:-4]
            for vehicle_flag in (False, True):
                target_root = "vehicle" if vehicle_flag else "pedestrian"
                bbox_dst = (
                    DEST_DATA / "annotations" / "bbox_annotated" / target_root /
                    SPLIT / scen / view / f"{base}_bbox.json"
                )
                bbox_src = resolve_bbox_src(kind, scen, view, v, vehicle=vehicle_flag)
                if bbox_src is not None:
                    relink(bbox_src, bbox_dst)
                    if vehicle_flag:
                        n_bbox_real_veh += 1
                    else:
                        n_bbox_real_ped += 1
                else:
                    bbox_dst.parent.mkdir(parents=True, exist_ok=True)
                    if not (bbox_dst.exists() or bbox_dst.is_symlink()):
                        json.dump({"annotations": []}, open(bbox_dst, "w"))
                        n_bbox_synth += 1

        # 2c) Caption JSON — load real if available, else synthesize from VQA phases.
        real_cap_path = resolve_real_caption_src(kind, scen, view)
        cap_dst = (
            DEST_DATA / "annotations" / "caption" / SPLIT / scen / view /
            f"{scen}_caption.json"
        )
        cap_dst.parent.mkdir(parents=True, exist_ok=True)
        if real_cap_path is not None:
            real_cap = json.load(open(real_cap_path))
            cap_dict = build_caption_dict_from_real(
                real_cap, kind, scen, view, videos_in_group,
            )
        else:
            # Synthesize from VQA timing (best we can do for VQA-only scenarios)
            synth_phases = []
            for label in sorted(g["vqa_timing_by_label"]):
                t_start, t_end = g["vqa_timing_by_label"][label]
                synth_phases.append({
                    "labels": [label],
                    "start_time": t_start, "end_time": t_end,
                })
            cap_dict = synthesize_caption_dict(
                kind=kind, scen=scen, view=view,
                videos=videos_in_group,
                event_phase_source=synth_phases,
            )
        json.dump(cap_dict, open(cap_dst, "w"), ensure_ascii=False, indent=2)
        n_cap_written += 1

        # 2d) VQA JSON (only when we have phases from gathered JSON for this group)
        if g["vqa_phases_by_label"]:
            n_groups_with_vqa += 1
            phases_out = []
            for label in sorted(g["vqa_phases_by_label"]):
                convs = g["vqa_phases_by_label"][label]
                t_start, t_end = g["vqa_timing_by_label"][label]
                phases_out.append({
                    "labels": [label],
                    "start_time": t_start, "end_time": t_end,
                    "conversations": convs,
                })
                # Build uuid_map entries
                for idx, conv in enumerate(convs):
                    uuid_official = conv.get("id", "")
                    if uuid_official:
                        # Key MUST include view: a scenario can have BOTH
                        # overhead_view and vehicle_view VQA blocks, with
                        # phase/idx starting at 0 in each. Without view in
                        # the key, the vehicle block silently overwrites
                        # the overhead block's UUIDs.
                        uuid_map[f"{scen}|{view}|{label}|{idx}"] = uuid_official
            vqa_entry = {
                "id": 0,
                "event_phase": phases_out,
            }
            if view == "vehicle_view":
                vqa_entry["vehicle_view"] = videos_in_group[0]
            else:
                vqa_entry["overhead_videos"] = videos_in_group
            vqa_dst = (
                DEST_DATA / "annotations" / "vqa" / SPLIT / scen / view /
                f"{scen}.json"
            )
            vqa_dst.parent.mkdir(parents=True, exist_ok=True)
            json.dump([vqa_entry], open(vqa_dst, "w"), ensure_ascii=False, indent=2)
            n_vqa_written += 1
        else:
            n_groups_caption_only += 1

    # 3) Save uuid_map
    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    map_path = DEST_ROOT / "vqa_uuid_map.json"
    json.dump(uuid_map, open(map_path, "w"))

    vlog("")
    vlog("=" * 50)
    vlog(f"videos symlinked       : {n_vid_ok}   (missing src: {n_vid_miss})")
    vlog(f"bbox real (ped)        : {n_bbox_real_ped}")
    vlog(f"bbox real (veh)        : {n_bbox_real_veh}")
    vlog(f"bbox synth empty       : {n_bbox_synth}")
    vlog(f"caption JSONs written  : {n_cap_written}")
    vlog(f"VQA JSONs written      : {n_vqa_written}")
    vlog(f"groups with VQA        : {n_groups_with_vqa}")
    vlog(f"groups caption-only    : {n_groups_caption_only}")
    vlog(f"uuid_map entries       : {len(uuid_map)}")
    vlog(f"uuid_map written       : {map_path}")
    vlog("=" * 50)
    vlog(f"DEST_DATA ready at {DEST_DATA}")


if __name__ == "__main__":
    main()
