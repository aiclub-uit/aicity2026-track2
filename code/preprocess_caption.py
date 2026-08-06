#!/usr/bin/env python
import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterable
import cv2
from tqdm import tqdm

PROJECT_ROOT = Path(os.environ.get("AICC26_PROJECT_ROOT", "/workspace"))

_SCRIPT_DIR = Path(__file__).resolve().parent

def _resolve_data_root() -> Path:
    env = os.environ.get("AICC26_DATA_ROOT")
    if env:
        return Path(env)
    snapshots_dir = (
        PROJECT_ROOT / ".cache" / "huggingface" / "hub"
        / "datasets--mlcglab--synwts" / "snapshots"
    )
    if snapshots_dir.is_dir():
        snapshots = sorted(
            (d for d in snapshots_dir.iterdir() if d.is_dir()),
            key=lambda d: d.stat().st_mtime,
        )
        if snapshots:
            return snapshots[-1] / "data"
    for candidate in (
        _SCRIPT_DIR.parent / "synwts" / "data",
        PROJECT_ROOT / "AICC" / "synwts" / "data",
        PROJECT_ROOT / "synwts" / "data",
        PROJECT_ROOT / "data_synwts" / "data",
    ):
        if candidate.is_dir():
            return candidate
    return snapshots_dir / "data"

DATA_ROOT = _resolve_data_root()

VIDEOS_DIR = DATA_ROOT / "videos"

ANN_DIR = DATA_ROOT / "annotations"

CAPTION_DIR = ANN_DIR / "caption"

VQA_DIR = ANN_DIR / "vqa"

BBOX_PED_DIR = ANN_DIR / "bbox_annotated" / "pedestrian"

BBOX_VEH_DIR = ANN_DIR / "bbox_annotated" / "vehicle"

WORK_ROOT = Path(os.environ.get("AICC26_WORK_ROOT_QWEN7B_CAP", str(PROJECT_ROOT / "output_qwen7b_caption")))

PROC_ROOT = WORK_ROOT / "processed"

FRAMES_GLOBAL = PROC_ROOT / "frames_global"

FRAMES_LOCAL = PROC_ROOT / "frames_local"

FRAMES_LOCAL_PED = PROC_ROOT / "frames_local_ped"

PED_CROP = os.environ.get("AICC26_PED_CROP", "0") == "1"

APC_SPATIAL_ROOT = PROC_ROOT / "apc_spatial_vqa"

CAPTION_DATASET = PROC_ROOT / "caption_{split}.json"

VQA_DATASET = PROC_ROOT / "vqa_{split}.json"

CHECKPOINT_ROOT = WORK_ROOT / "checkpoints"

SUBMISSION_ROOT = WORK_ROOT / "submissions"

PHASE_STR_TO_NUM = {
    "prerecognition": "0",
    "recognition": "1",
    "judgement": "2",
    "action": "3",
    "avoidance": "4",
}

PHASE_NUM_TO_STR = {v: k for k, v in PHASE_STR_TO_NUM.items()}

def to_phase_num(label: Any) -> str:
    s = str(label).strip().lower()
    if s in PHASE_STR_TO_NUM:
        return PHASE_STR_TO_NUM[s]
    if s in PHASE_NUM_TO_STR:
        return s
    raise ValueError(f"Unknown phase label: {label!r}")

@dataclass(frozen=True)
class FrameConfig:
    local_pad_ratio: float = 0.25
    local_pad_ratio_ped: float = 0.6
    center_crop_ratio: float = 0.6
    line_thickness: int = 3
    jpeg_quality: int = 92
    global_max_side: int = 1280
    local_max_side: int = 768

FRAME_CFG = FrameConfig()

def ensure_dirs() -> None:
    for d in (WORK_ROOT, PROC_ROOT, FRAMES_GLOBAL, FRAMES_LOCAL, FRAMES_LOCAL_PED,
              APC_SPATIAL_ROOT, CHECKPOINT_ROOT, SUBMISSION_ROOT):
        d.mkdir(parents=True, exist_ok=True)

def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str | Path, payload: Any, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=indent)

def list_scenarios(split_dir: Path) -> list[str]:
    if not split_dir.exists():
        return []
    return sorted(
        d.name for d in split_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
        and d.name != "normal_trimmed"
    )

def camera_id_from_video(scenario: str, video_name: str) -> str:
    base = video_name.replace(".mp4", "")
    prefix = scenario + "_"
    return base[len(prefix):] if base.startswith(prefix) else base

@dataclass(frozen=True)
class BBox:
    x: float
    y: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return int(self.x), int(self.y), int(self.x + self.w), int(self.y + self.h)

def load_bbox_records(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = read_json(p)
    except Exception as exc:
        print(f"[bbox][warn] Failed to read {p}: {exc}")
        return []
    return data.get("annotations", []) if isinstance(data, dict) else []

def _bbox_by_image(records: Iterable[dict], phase_num: str) -> dict[int, BBox]:
    by_image: dict[int, BBox] = {}
    for record in records:
        if str(record.get("phase_number")) != str(phase_num):
            continue
        bbox = record.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        try:
            image_id = int(record.get("image_id", 0))
        except (TypeError, ValueError):
            continue
        candidate = BBox(*bbox)
        current = by_image.get(image_id)
        if current is None or candidate.area > current.area:
            by_image[image_id] = candidate
    return by_image

def _nearest_bbox(by_image: dict[int, BBox], target_frame_id: int) -> BBox | None:
    if not by_image:
        return None
    nearest_id = min(by_image, key=lambda k: abs(k - target_frame_id))
    return by_image[nearest_id]

def union_bbox(*boxes: BBox | None) -> BBox | None:
    valid = [b for b in boxes if b is not None]
    if not valid:
        return None
    x0 = min(b.x for b in valid)
    y0 = min(b.y for b in valid)
    x1 = max(b.x + b.w for b in valid)
    y1 = max(b.y + b.h for b in valid)
    return BBox(x0, y0, x1 - x0, y1 - y0)

def pad_bbox(box: BBox, pad_ratio: float, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    pad_w = box.w * pad_ratio
    pad_h = box.h * pad_ratio
    return (
        max(0, int(box.x - pad_w)),
        max(0, int(box.y - pad_h)),
        min(img_w, int(box.x + box.w + pad_w)),
        min(img_h, int(box.y + box.h + pad_h)),
    )

ANTI_HALLUCINATION_NOTE = (
    "Important: Only describe objects and behaviors that are clearly visible "
    "in the frames. Do not invent hand-held items (phones, bags, umbrellas) "
    "and do not speculate about specific poses such as falling or jumping "
    "unless they are unambiguously shown."
)

PEDESTRIAN_CAPTION_PROMPT = (
    "{image_context}\n"
    "Describe the pedestrian (highlighted by a green box if visible) for the {phase_name} phase. Cover four pillars: "
    "(1) Location relative to the vehicle and the road, "
    "(2) Attention / line of sight, "
    "(3) Behavior (specific actions, movement, posture), "
    "(4) Context (apparent age range, height, clothing, weather, road surface). "
    f"{ANTI_HALLUCINATION_NOTE}"
)

VEHICLE_CAPTION_PROMPT = (
    "{image_context}\n"
    "Describe the vehicle (highlighted by a red box if visible) for the {phase_name} phase. Cover four pillars: "
    "(1) Location relative to the pedestrian and the road, "
    "(2) Visibility of the pedestrian from the vehicle, "
    "(3) Behavior (action: moving / reversing / parking / stopped, speed cues), "
    "(4) Context (surrounding environment, pedestrian appearance summary, "
    "weather, road surface). "
    f"{ANTI_HALLUCINATION_NOTE}"
)

VQA_PROMPT_TEMPLATE = (
    "{image_context}\n"
    "Phase: {phase_name}.\n"
    "Question: {question}\n"
    "Options:\n{options}\n"
    "Answer with a single letter (a, b, c, or d). Choose the best option."
)

def format_options(opts: dict) -> str:
    return "\n".join(
        f"({k}) {opts[k]}"
        for k in ("a", "b", "c", "d")
        if k in opts and opts[k] is not None
    )

PED_COLOR = (0, 255, 0)

VEH_COLOR = (0, 0, 255)

FRAME_SUFFIX_ORDER = ("start", "mid", "ens", "end")

MERGE_FRAME_DISTANCE = 1

@dataclass(frozen=True)
class _SelectedFrame:
    suffix: str
    frame_id: int
    ped_bb: BBox | None
    veh_bb: BBox | None

@dataclass(frozen=True)
class _Job:
    split: str
    scenario: str
    view: str
    video_name: str

    @property
    def camera_id(self) -> str:
        return camera_id_from_video(self.scenario, self.video_name)

    @property
    def base_video(self) -> str:
        return self.video_name.replace(".mp4", "")

def _resize_keep_aspect(img, max_side: int):
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img
    s = max_side / longest
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

def _draw_bbox(img, box: BBox | None, color, label: str) -> None:
    if box is None:
        return
    x0, y0, x1, y1 = box.xyxy
    cv2.rectangle(img, (x0, y0), (x1, y1), color, FRAME_CFG.line_thickness)
    cv2.putText(
        img, label, (x0, max(15, y0 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
    )

def _center_crop(frame, ratio: float = FRAME_CFG.center_crop_ratio):
    h, w = frame.shape[:2]
    cw, ch = int(w * ratio), int(h * ratio)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return frame[y0:y0 + ch, x0:x0 + cw]

def _clip_frame_id(frame_id: int, start_frame: int, end_frame: int) -> int:
    return min(max(frame_id, start_frame), end_frame)

def _phase_frame_bounds(
    t_start: float, t_end: float, fps: float, total_frames: int,
) -> tuple[int, int, int]:
    start_frame = max(0, int(t_start * fps))
    end_frame = max(start_frame, int(t_end * fps) - 1)
    if total_frames > 0:
        last_frame = total_frames - 1
        start_frame = min(start_frame, last_frame)
        end_frame = min(end_frame, last_frame)
        if end_frame < start_frame:
            end_frame = start_frame
    t_mid = (t_start + t_end) / 2.0
    mid_frame = _clip_frame_id(max(0, int(t_mid * fps)), start_frame, end_frame)
    return start_frame, mid_frame, end_frame

def _bbox_pair_at(
    ped_by_image: dict[int, BBox],
    veh_by_image: dict[int, BBox],
    frame_id: int,
) -> tuple[BBox | None, BBox | None]:
    return ped_by_image.get(frame_id), veh_by_image.get(frame_id)

def _scan_phase_frames(anchor: int, boundary: int, step: int) -> range:
    stop = boundary + step
    return range(anchor, stop, step)

def _select_overhead_frame(
    *,
    suffix: str,
    anchor_frame: int,
    start_frame: int,
    end_frame: int,
    direction: int,
    ped_by_image: dict[int, BBox],
    veh_by_image: dict[int, BBox],
) -> _SelectedFrame:
    anchor_frame = _clip_frame_id(anchor_frame, start_frame, end_frame)
    boundary = end_frame if direction > 0 else start_frame
    scan = _scan_phase_frames(anchor_frame, boundary, 1 if direction > 0 else -1)

    for frame_id in scan:
        ped_bb, veh_bb = _bbox_pair_at(ped_by_image, veh_by_image, frame_id)
        if ped_bb is not None and veh_bb is not None:
            return _SelectedFrame(suffix, frame_id, ped_bb, veh_bb)

    for frame_id in scan:
        ped_bb, veh_bb = _bbox_pair_at(ped_by_image, veh_by_image, frame_id)
        if ped_bb is not None or veh_bb is not None:
            return _SelectedFrame(suffix, frame_id, ped_bb, veh_bb)

    ped_bb, veh_bb = _bbox_pair_at(ped_by_image, veh_by_image, start_frame)
    return _SelectedFrame("start", start_frame, ped_bb, veh_bb)

def _bbox_count(selected: _SelectedFrame) -> int:
    return int(selected.ped_bb is not None) + int(selected.veh_bb is not None)

def _with_suffix(selected: _SelectedFrame, suffix: str) -> _SelectedFrame:
    return _SelectedFrame(suffix, selected.frame_id, selected.ped_bb, selected.veh_bb)

def _merge_close_mid_end(
    mid_frame: _SelectedFrame,
    end_frame: _SelectedFrame,
) -> list[_SelectedFrame]:
    if abs(mid_frame.frame_id - end_frame.frame_id) > MERGE_FRAME_DISTANCE:
        return [mid_frame, end_frame]
    if mid_frame.suffix == "start" and end_frame.suffix == "start":
        return [_with_suffix(mid_frame, "start")]
    representative = mid_frame if _bbox_count(mid_frame) >= _bbox_count(end_frame) else end_frame
    return [_with_suffix(representative, "ens")]

def _clear_phase_outputs(out_g: Path, out_l: Path | None, base: str, label: str) -> None:
    for path in out_g.glob(f"{base}_phase{label}_*.jpg"):
        path.unlink()
    if out_l is not None:
        for path in out_l.glob(f"{base}_phase{label}_*.jpg"):
            path.unlink()

def _overhead_videos_for(scenario: str, split: str) -> list[str]:
    cap_path = CAPTION_DIR / split / scenario / "overhead_view" / f"{scenario}_caption.json"
    if not cap_path.exists():
        return []
    listed = read_json(cap_path).get("overhead_videos") or []
    keep: list[str] = []
    for v in listed:
        base = v.replace(".mp4", "")
        if (
            (VIDEOS_DIR / split / scenario / "overhead_view" / v).exists()
            and (BBOX_PED_DIR / split / scenario / "overhead_view" / f"{base}_bbox.json").exists()
            and (BBOX_VEH_DIR / split / scenario / "overhead_view" / f"{base}_bbox.json").exists()
        ):
            keep.append(v)
    return keep

def _score_overhead_video(scenario: str, split: str, video_name: str) -> float:
    base = video_name.replace(".mp4", "")
    vid_path = VIDEOS_DIR / split / scenario / "overhead_view" / video_name
    cap_path = CAPTION_DIR / split / scenario / "overhead_view" / f"{scenario}_caption.json"
    ped_path = BBOX_PED_DIR / split / scenario / "overhead_view" / f"{base}_bbox.json"
    veh_path = BBOX_VEH_DIR / split / scenario / "overhead_view" / f"{base}_bbox.json"
    if not (vid_path.exists() and cap_path.exists()):
        return 0.0

    cv_cap = cv2.VideoCapture(str(vid_path))
    if not cv_cap.isOpened():
        return 0.0
    fps = cv_cap.get(cv2.CAP_PROP_FPS) or 30.0
    cv_cap.release()

    ped_records = load_bbox_records(ped_path)
    veh_records = load_bbox_records(veh_path)
    cap_data = read_json(cap_path)

    score = 0.0
    for phase in cap_data.get("event_phase", []):
        label = str((phase.get("labels") or [""])[0])
        try:
            t_start = float(phase.get("start_time", 0.0))
            t_end = float(phase.get("end_time", t_start))
        except (TypeError, ValueError):
            continue
        t_mid = (t_start + max(t_end, t_start)) / 2.0
        frame_id = max(0, int(t_mid * fps))
        ped_by_image = _bbox_by_image(ped_records, label)
        veh_by_image = _bbox_by_image(veh_records, label)
        ped_bb = _nearest_bbox(ped_by_image, frame_id)
        veh_bb = _nearest_bbox(veh_by_image, frame_id)
        score += (ped_bb.area if ped_bb else 0.0) + (veh_bb.area if veh_bb else 0.0)
    return score

@lru_cache(maxsize=4096)
def _best_overhead_video(scenario: str, split: str) -> str | None:
    candidates = _overhead_videos_for(scenario, split)
    if not candidates:
        return None
    scored = [(v, _score_overhead_video(scenario, split, v)) for v in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0]

def _vehicle_video_for(scenario: str, split: str) -> str | None:
    v = f"{scenario}_vehicle_view.mp4"
    base = v.replace(".mp4", "")
    cap_path = CAPTION_DIR / split / scenario / "vehicle_view" / f"{scenario}_caption.json"
    if not (
        (VIDEOS_DIR / split / scenario / "vehicle_view" / v).exists()
        and cap_path.exists()
        and (BBOX_PED_DIR / split / scenario / "vehicle_view" / f"{base}_bbox.json").exists()
    ):
        return None
    return v

def _build_jobs(splits: tuple[str, ...] = ("train", "val")) -> list[_Job]:
    jobs: list[_Job] = []
    for split in splits:
        for scenario in list_scenarios(CAPTION_DIR / split):
            best_ov = _best_overhead_video(scenario, split)
            if best_ov is not None:
                jobs.append(_Job(split, scenario, "overhead_view", best_ov))
            vh = _vehicle_video_for(scenario, split)
            if vh is not None:
                jobs.append(_Job(split, scenario, "vehicle_view", vh))
    return jobs

def _caption_path_for(job: _Job) -> Path:
    return CAPTION_DIR / job.split / job.scenario / job.view / f"{job.scenario}_caption.json"

def _process_job(job: _Job) -> int:
    is_overhead = job.view == "overhead_view"
    vid_path = VIDEOS_DIR / job.split / job.scenario / job.view / job.video_name
    cap_path = _caption_path_for(job)
    if not vid_path.exists() or not cap_path.exists():
        return 0

    base = job.base_video
    ped_path = BBOX_PED_DIR / job.split / job.scenario / job.view / f"{base}_bbox.json"
    ped_records = load_bbox_records(ped_path)
    if is_overhead:
        veh_path = BBOX_VEH_DIR / job.split / job.scenario / job.view / f"{base}_bbox.json"
        veh_records = load_bbox_records(veh_path)
    else:
        veh_records = []
    cap_data = read_json(cap_path)

    cv_cap = cv2.VideoCapture(str(vid_path))
    if not cv_cap.isOpened():
        return 0
    fps = cv_cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cv_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    out_g = FRAMES_GLOBAL / job.split / job.scenario / job.view
    out_g.mkdir(parents=True, exist_ok=True)
    out_l: Path | None = None
    out_l_ped: Path | None = None
    if is_overhead:
        out_l = FRAMES_LOCAL / job.split / job.scenario / job.view
        out_l.mkdir(parents=True, exist_ok=True)
        if PED_CROP:
            out_l_ped = FRAMES_LOCAL_PED / job.split / job.scenario / job.view
            out_l_ped.mkdir(parents=True, exist_ok=True)

    written = 0
    for phase in cap_data.get("event_phase", []):
        label = str((phase.get("labels") or [""])[0])
        try:
            t_start = float(phase.get("start_time", 0.0))
            t_end = float(phase.get("end_time", t_start))
        except (TypeError, ValueError):
            continue
        if t_end < t_start:
            t_end = t_start
        start_frame, mid_anchor, end_anchor = _phase_frame_bounds(
            t_start, t_end, fps, total_frames,
        )
        _clear_phase_outputs(out_g, out_l, base, label)

        if is_overhead:
            ped_by_image = _bbox_by_image(ped_records, label)
            veh_by_image = _bbox_by_image(veh_records, label)
            selected_frames = _merge_close_mid_end(
                _select_overhead_frame(
                    suffix="mid",
                    anchor_frame=mid_anchor,
                    start_frame=start_frame,
                    end_frame=end_anchor,
                    direction=1,
                    ped_by_image=ped_by_image,
                    veh_by_image=veh_by_image,
                ),
                _select_overhead_frame(
                    suffix="end",
                    anchor_frame=end_anchor,
                    start_frame=start_frame,
                    end_frame=end_anchor,
                    direction=-1,
                    ped_by_image=ped_by_image,
                    veh_by_image=veh_by_image,
                ),
            )
        else:
            ped_by_image = _bbox_by_image(ped_records, label)
            selected_frames = [
                _SelectedFrame("mid", mid_anchor, ped_by_image.get(mid_anchor), None),
                _SelectedFrame("end", end_anchor, ped_by_image.get(end_anchor), None),
            ]

        for selected in selected_frames:
            cv_cap.set(cv2.CAP_PROP_POS_FRAMES, selected.frame_id)
            ok, frame = cv_cap.read()
            if not ok or frame is None:
                continue
            h, w = frame.shape[:2]
            drawn = frame.copy()
            _draw_bbox(drawn, selected.ped_bb, PED_COLOR, "Pedestrian")

            if is_overhead:
                _draw_bbox(drawn, selected.veh_bb, VEH_COLOR, "Vehicle")
                global_img = _resize_keep_aspect(drawn.copy(), FRAME_CFG.global_max_side)
                u = union_bbox(selected.ped_bb, selected.veh_bb)
                if u is not None:
                    x0, y0, x1, y1 = pad_bbox(u, FRAME_CFG.local_pad_ratio, w, h)
                    local_img = drawn[y0:y1, x0:x1]
                    if local_img.shape[0] == 0 or local_img.shape[1] == 0:
                        local_img = _center_crop(drawn)
                else:
                    local_img = _center_crop(drawn)
                local_img = _resize_keep_aspect(local_img, FRAME_CFG.local_max_side)
                cv2.imwrite(
                    str(out_g / f"{base}_phase{label}_{selected.suffix}.jpg"),
                    global_img, [cv2.IMWRITE_JPEG_QUALITY, FRAME_CFG.jpeg_quality],
                )
                cv2.imwrite(
                    str(out_l / f"{base}_phase{label}_{selected.suffix}.jpg"),
                    local_img, [cv2.IMWRITE_JPEG_QUALITY, FRAME_CFG.jpeg_quality],
                )
                if out_l_ped is not None:
                    if selected.ped_bb is not None:
                        px0, py0, px1, py1 = pad_bbox(selected.ped_bb, FRAME_CFG.local_pad_ratio_ped, w, h)
                        ped_img = drawn[py0:py1, px0:px1]
                        if ped_img.shape[0] == 0 or ped_img.shape[1] == 0:
                            ped_img = local_img
                    else:
                        ped_img = local_img
                    ped_img = _resize_keep_aspect(ped_img, FRAME_CFG.local_max_side)
                    cv2.imwrite(
                        str(out_l_ped / f"{base}_phase{label}_{selected.suffix}.jpg"),
                        ped_img, [cv2.IMWRITE_JPEG_QUALITY, FRAME_CFG.jpeg_quality],
                    )
            else:
                global_img = _resize_keep_aspect(drawn, FRAME_CFG.global_max_side)
                cv2.imwrite(
                    str(out_g / f"{base}_phase{label}_{selected.suffix}.jpg"),
                    global_img, [cv2.IMWRITE_JPEG_QUALITY, FRAME_CFG.jpeg_quality],
                )
            written += 1

    cv_cap.release()
    return written

def run_extract_frames(num_workers: int = 4) -> None:
    ensure_dirs()
    jobs = _build_jobs()
    print(f"[frames] {len(jobs)} (scenario, view, camera) jobs")
    if num_workers <= 1:
        total = sum(_process_job(j) for j in tqdm(jobs, desc="frames"))
    else:
        with Pool(num_workers) as pool:
            total = sum(tqdm(pool.imap_unordered(_process_job, jobs), total=len(jobs), desc="frames"))
    print(f"[frames] wrote {total} selected phase frame sets")

def _phase_image_path(
    root: Path, split: str, scenario: str, view: str, base_video: str,
    phase_label: str, suffix: str,
) -> Path:
    return root / split / scenario / view / f"{base_video}_phase{phase_label}_{suffix}.jpg"

def _frame_paths(
    split: str, scenario: str, view: str, base_video: str, phase_label: str,
    local_root: Path = FRAMES_LOCAL,
) -> tuple[Path, ...] | None:
    paths: list[Path] = []
    for suffix in FRAME_SUFFIX_ORDER:
        g = _phase_image_path(FRAMES_GLOBAL, split, scenario, view, base_video, phase_label, suffix)
        if view == "overhead_view":
            l = _phase_image_path(local_root, split, scenario, view, base_video, phase_label, suffix)
            if g.exists() and l.exists():
                paths.extend([g, l])
        elif g.exists():
            paths.append(g)
    return tuple(paths) if paths else None

def _frame_suffix_from_path(path: Path) -> str:
    m = re.search(r"_phase[^_]+_([^_]+)\.jpg$", path.name)
    return m.group(1) if m else "selected"

def _image_kind_from_path(path: Path) -> str:
    if FRAMES_LOCAL.name in path.parts or FRAMES_LOCAL_PED.name in path.parts:
        return "local crop"
    return "global view"

def _frame_suffix_description(suffix: str) -> str:
    return {
        "start": "phase start fallback",
        "mid": "middle frame",
        "ens": "merged middle/end frame",
        "end": "end frame",
    }.get(suffix, "selected frame")

def _image_context_for_paths(paths: Iterable[Path]) -> str:
    lines = ["Images are provided from the traffic scenario in this order:"]
    for idx, path in enumerate(paths, start=1):
        suffix = _frame_suffix_from_path(path)
        lines.append(
            f"Image {idx}: {_image_kind_from_path(path)} at the "
            f"{_frame_suffix_description(suffix)}."
        )
    return "\n".join(lines)

def _format_phase_prompt(template: str, *, phase_name: str, paths: Iterable[Path], **kwargs) -> str:
    return template.format(
        phase_name=phase_name,
        image_context=_image_context_for_paths(paths),
        **kwargs,
    )

def _phase_name(label: Any) -> str:
    return PHASE_NUM_TO_STR.get(str(label), str(label))

def _caption_samples_from_json(
    *, split: str, scenario: str, view: str, base_video: str, camera_id: str,
    cap_path: Path,
) -> list[dict]:
    if not cap_path.exists():
        return []
    samples: list[dict] = []
    cap_data = read_json(cap_path)
    for phase in cap_data.get("event_phase", []):
        label = str((phase.get("labels") or [""])[0])
        paths = _frame_paths(split, scenario, view, base_video, label)
        if paths is None:
            continue
        ped_paths = paths
        if PED_CROP and view == "overhead_view":
            pp = _frame_paths(split, scenario, view, base_video, label, local_root=FRAMES_LOCAL_PED)
            if pp is not None:
                ped_paths = pp
        ph_name = _phase_name(label)
        ped_text = (phase.get("caption_pedestrian") or "").strip()
        veh_text = (phase.get("caption_vehicle") or "").strip()
        image_paths = [str(p) for p in paths]
        if ped_text:
            samples.append({
                "id": f"{scenario}_{camera_id}_phase{label}_ped",
                "scenario": scenario, "view": view, "camera_id": camera_id,
                "phase_label": label, "phase_name": ph_name, "target": "pedestrian",
                "images": [str(p) for p in ped_paths],
                "conversations": [
                    {"from": "user",
                     "value": _format_phase_prompt(
                         PEDESTRIAN_CAPTION_PROMPT, phase_name=ph_name, paths=ped_paths,
                     )},
                    {"from": "assistant", "value": ped_text},
                ],
            })
        if veh_text:
            samples.append({
                "id": f"{scenario}_{camera_id}_phase{label}_veh",
                "scenario": scenario, "view": view, "camera_id": camera_id,
                "phase_label": label, "phase_name": ph_name, "target": "vehicle",
                "images": image_paths,
                "conversations": [
                    {"from": "user",
                     "value": _format_phase_prompt(
                         VEHICLE_CAPTION_PROMPT, phase_name=ph_name, paths=paths,
                     )},
                    {"from": "assistant", "value": veh_text},
                ],
            })
    return samples

def _build_caption_split(split: str) -> list[dict]:
    samples: list[dict] = []
    cap_split_dir = CAPTION_DIR / split
    for scenario in tqdm(list_scenarios(cap_split_dir), desc=f"caption[{split}]"):
        ov_cap_path = cap_split_dir / scenario / "overhead_view" / f"{scenario}_caption.json"
        best_ov = _best_overhead_video(scenario, split)
        if best_ov is not None:
            base = best_ov.replace(".mp4", "")
            cam_id = camera_id_from_video(scenario, best_ov)
            samples.extend(_caption_samples_from_json(
                split=split, scenario=scenario, view="overhead_view",
                base_video=base, camera_id=cam_id, cap_path=ov_cap_path,
            ))
        vh = _vehicle_video_for(scenario, split)
        if vh is not None:
            vh_cap_path = cap_split_dir / scenario / "vehicle_view" / f"{scenario}_caption.json"
            base = vh.replace(".mp4", "")
            cam_id = camera_id_from_video(scenario, vh)
            samples.extend(_caption_samples_from_json(
                split=split, scenario=scenario, view="vehicle_view",
                base_video=base, camera_id=cam_id, cap_path=vh_cap_path,
            ))
    return samples

def run_build_caption(splits: tuple[str, ...] = ("train", "val")) -> None:
    ensure_dirs()
    for split in splits:
        samples = _build_caption_split(split)
        name = f"caption_{split}_pedcrop.json" if PED_CROP else f"caption_{split}.json"
        out = PROC_ROOT / name
        write_json(out, samples)
        print(f"[caption] {split}: {len(samples)} samples -> {out}")

_ENV_PHASE_NUM = "2"

_ENV_PHASE_NAME = "judgement"

def _vqa_sample(
    *, scenario: str, view: str, camera_id: str, phase_num: str, phase_name: str,
    qid: str, conv: dict, images: tuple[Path, ...], split: str, base_video: str,
) -> dict | None:
    correct = str(conv.get("correct", "")).strip().lower()
    if correct not in ("a", "b", "c", "d"):
        return None
    return {
        "id": qid,
        "split": split,
        "scenario": scenario, "view": view, "camera_id": camera_id,
        "base_video": base_video,
        "phase_label": phase_num, "phase_name": phase_name,
        "question": (conv.get("question") or "").strip(),
        "options": {k: conv.get(k) for k in ("a", "b", "c", "d") if k in conv},
        "images": [str(p) for p in images],
        "conversations": [
            {"from": "user",
             "value": _format_phase_prompt(
                 VQA_PROMPT_TEMPLATE,
                 phase_name=phase_name,
                 paths=images,
                 question=(conv.get("question") or "").strip(),
                 options=format_options(conv),
             )},
            {"from": "assistant", "value": correct},
        ],
    }

def _process_vqa_file(
    *, split: str, scenario: str, view: str, base_video: str, camera_id: str,
    json_path: Path,
) -> list[dict]:
    if not json_path.exists():
        return []
    data = read_json(json_path)
    if not isinstance(data, list):
        return []
    out: list[dict] = []
    for entry in data:
        for phase in entry.get("event_phase", []):
            label = (phase.get("labels") or [""])[0]
            try:
                phase_num = to_phase_num(label)
            except ValueError:
                continue
            phase_name = label if isinstance(label, str) else str(label)
            paths = _frame_paths(split, scenario, view, base_video, phase_num)
            if paths is None:
                continue
            for idx, conv in enumerate(phase.get("conversations", [])):
                qid = f"{scenario}_{camera_id}_phase{phase_num}_q{idx}"
                s = _vqa_sample(
                    scenario=scenario, view=view, camera_id=camera_id,
                    phase_num=phase_num, phase_name=phase_name,
                    qid=qid, conv=conv, images=paths,
                    split=split, base_video=base_video,
                )
                if s is not None:
                    out.append(s)
    return out

def _process_env_file(
    *, split: str, scenario: str, view: str, base_video: str, camera_id: str,
    json_path: Path,
) -> list[dict]:
    if not json_path.exists():
        return []
    data = read_json(json_path)
    if not isinstance(data, list):
        return []
    paths = _frame_paths(split, scenario, view, base_video, _ENV_PHASE_NUM)
    if paths is None:
        return []
    out: list[dict] = []
    for entry in data:
        for idx, conv in enumerate(entry.get("environment", [])):
            qid = f"{scenario}_{camera_id}_environment_q{idx}"
            s = _vqa_sample(
                scenario=scenario, view=view, camera_id=camera_id,
                phase_num=_ENV_PHASE_NUM, phase_name=_ENV_PHASE_NAME,
                qid=qid, conv=conv, images=paths,
                split=split, base_video=base_video,
            )
            if s is not None:
                out.append(s)
    return out

def _build_vqa_split(split: str) -> list[dict]:
    samples: list[dict] = []
    vqa_split_dir = VQA_DIR / split
    for scenario in tqdm(list_scenarios(vqa_split_dir), desc=f"vqa[{split}]"):
        best_ov = _best_overhead_video(scenario, split)
        if best_ov is not None:
            base = best_ov.replace(".mp4", "")
            cam_id = camera_id_from_video(scenario, best_ov)
            samples.extend(_process_vqa_file(
                split=split, scenario=scenario, view="overhead_view",
                base_video=base, camera_id=cam_id,
                json_path=vqa_split_dir / scenario / "overhead_view" / f"{scenario}.json",
            ))
            samples.extend(_process_env_file(
                split=split, scenario=scenario, view="overhead_view",
                base_video=base, camera_id=cam_id,
                json_path=vqa_split_dir / scenario / "environment" / f"{scenario}.json",
            ))
        vh = _vehicle_video_for(scenario, split)
        if vh is not None:
            base = vh.replace(".mp4", "")
            cam_id = camera_id_from_video(scenario, vh)
            samples.extend(_process_vqa_file(
                split=split, scenario=scenario, view="vehicle_view",
                base_video=base, camera_id=cam_id,
                json_path=vqa_split_dir / scenario / "vehicle_view" / f"{scenario}.json",
            ))
    return samples

def run_build_vqa(splits: tuple[str, ...] = ("train", "val")) -> None:
    ensure_dirs()
    for split in splits:
        samples = _build_vqa_split(split)
        out = Path(str(VQA_DATASET).format(split=split))
        write_json(out, samples)
        print(f"[vqa] {split}: {len(samples)} samples -> {out}")

def _frames_exist(splits: tuple[str, ...] = ("train", "val")) -> bool:
    for split in splits:
        split_dir = FRAMES_GLOBAL / split
        if not split_dir.exists() or not any(split_dir.rglob("*.jpg")):
            return False
        has_new_suffix = any(
            any(split_dir.rglob(f"*_phase*_{suffix}.jpg"))
            for suffix in ("end", "ens", "start")
        )
        if not has_new_suffix:
            return False
    return True

def _dataset_has_selected_frame_paths(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = read_json(path)
    except Exception:
        return False
    if not isinstance(data, list):
        return False
    for sample in data:
        for image_path in sample.get("images", []):
            if re.search(r"_phase[^_]+_(end|ens|start)\.jpg$", str(image_path)):
                return True
    return False

def run_preprocess_all(num_workers: int = 4, *, force: bool = False) -> None:
    ensure_dirs()

    if force or not _frames_exist():
        run_extract_frames(num_workers=num_workers)
    else:
        print(f"[preprocess] skip extract_frames (frames found under {FRAMES_GLOBAL})")

    cap_paths = [Path(str(CAPTION_DATASET).format(split=s)) for s in ("train", "val")]
    if force or not all(_dataset_has_selected_frame_paths(p) for p in cap_paths):
        run_build_caption()
    else:
        print(f"[preprocess] skip build_caption (datasets exist)")

    vqa_paths = [Path(str(VQA_DATASET).format(split=s)) for s in ("train", "val")]
    if force or not all(_dataset_has_selected_frame_paths(p) for p in vqa_paths):
        run_build_vqa()
    else:
        print(f"[preprocess] skip build_vqa (datasets exist)")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Preprocessing caption (tách từ run_caption_qwen7b.py)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="làm lại dù output đã tồn tại")
    a = ap.parse_args()
    run_preprocess_all(num_workers=a.workers, force=a.force)


if __name__ == "__main__":
    main()
