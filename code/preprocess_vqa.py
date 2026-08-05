#!/usr/bin/env python
"""preprocess_vqa.py — CÔNG ĐOẠN PREPROCESSING VQA, tách nguyên trạng từ run7_qwen7b.py.

Mọi hàm/hằng được trích VERBATIM (byte-identical, verify bằng AST) từ closure
`run_preprocess_all` của run7_qwen7b.py — không sửa logic; chỉ bỏ phần train/eval
và các import nặng (torch/transformers/peft) mà preprocessing không dùng.
Phụ thuộc: cv2 + tqdm + stdlib.

Pipeline: stats -> extract frames (chọn keyframe theo phase, vẽ bbox, crop local)
          -> build vqa_{train,val}.json -> build caption_train.json

Env:
  AICC26_DATA_ROOT          root dataset (videos/ + annotations/), mặc định <project>/synwts_data/data
  AICC26_WORK_ROOT_QWEN7B   root output (processed/...), mặc định /workspace/AICC/code/output_qwen7b
  AICC26_PROJECT_ROOT       mặc định /workspace

TRAIN: AICC26_DATA_ROOT=<synwts>/data python preprocess_vqa.py --workers 8
TEST:  test package mount như split "val" (lịch sử: prepare_test_root.py):
       AICC26_DATA_ROOT=<test_root>/data AICC26_WORK_ROOT_QWEN7B=<work> python preprocess_vqa.py
       -> <work>/processed/vqa_val.json = metadata VQA test (= data_meta/vqa_test.json)
"""
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterable
import cv2
from tqdm import tqdm

_DEFAULT_PROJECT_ROOT = os.environ.get("AICC26_PROJECT_ROOT", "/workspace")

_DEFAULT_WORK_ROOT = os.environ.get("AICC26_WORK_ROOT_QWEN7B", os.path.join(_DEFAULT_PROJECT_ROOT, "AICC", "code", "output_qwen7b"))

PROJECT_ROOT = Path(_DEFAULT_PROJECT_ROOT)

DATA_ROOT = Path(os.environ.get("AICC26_DATA_ROOT", str(PROJECT_ROOT / "synwts_data" / "data")))

VIDEOS_DIR = DATA_ROOT / "videos"

ANN_DIR = DATA_ROOT / "annotations"

VQA_DIR = ANN_DIR / "vqa"

BBOX_PED_DIR = ANN_DIR / "bbox_annotated" / "pedestrian"

BBOX_VEH_DIR = ANN_DIR / "bbox_annotated" / "vehicle"

WORK_ROOT = Path(_DEFAULT_WORK_ROOT)

PROC_ROOT = WORK_ROOT / "processed"

FRAMES_GLOBAL = PROC_ROOT / "frames_global"

FRAMES_LOCAL = PROC_ROOT / "frames_local"

VQA_DATASET = PROC_ROOT / "vqa_{split}.json"

VQA_STATS = PROC_ROOT / "vqa_data_stats.json"

CAPTION_DIR = ANN_DIR / "caption"

CAPTION_DATASET = PROC_ROOT / "caption_train.json"

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

def _phase_name(label: Any) -> str:
    s = str(label).strip()
    if s in PHASE_NUM_TO_STR:
        return PHASE_NUM_TO_STR[s]
    return s.lower() if s else "unknown"

@dataclass(frozen=True)
class FrameConfig:
    local_pad_ratio: float = 0.25
    center_crop_ratio: float = 0.6
    line_thickness: int = 3
    jpeg_quality: int = 92
    global_max_side: int = 1280
    local_max_side: int = 768

FRAME_CFG = FrameConfig()

MERGE_FRAME_DISTANCE = 3

def ensure_dirs() -> None:
    for d in (WORK_ROOT, PROC_ROOT, FRAMES_GLOBAL, FRAMES_LOCAL,
              CHECKPOINT_ROOT, SUBMISSION_ROOT):
        d.mkdir(parents=True, exist_ok=True)

def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str | Path, payload: Any, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=indent)

def camera_id_from_video(scenario_id: str, video_name: str) -> str:
    base = video_name.replace(".mp4", "")
    if base == scenario_id:
        return "overhead_view"
    prefix = scenario_id + "_"
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
    except Exception:
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
    present = [b for b in boxes if b is not None]
    if not present:
        return None
    x0 = min(b.x for b in present)
    y0 = min(b.y for b in present)
    x1 = max(b.x + b.w for b in present)
    y1 = max(b.y + b.h for b in present)
    return BBox(x0, y0, x1 - x0, y1 - y0)

def pad_bbox(box: BBox, pad_ratio: float, width: int, height: int) -> tuple[int, int, int, int]:
    pad = pad_ratio * max(box.w, box.h)
    x0 = max(0, int(box.x - pad))
    y0 = max(0, int(box.y - pad))
    x1 = min(width, int(box.x + box.w + pad))
    y1 = min(height, int(box.y + box.h + pad))
    if x1 <= x0 or y1 <= y0:
        return 0, 0, width, height
    return x0, y0, x1, y1

VIEWS = ("overhead_view", "vehicle_view", "environment")

@dataclass(frozen=True)
class ScenarioRef:
    split: str
    rel_path: Path
    scenario_id: str

    @property
    def rel_key(self) -> str:
        return self.rel_path.as_posix()

@dataclass(frozen=True)
class _Job:
    ref: ScenarioRef
    view: str
    video_name: str

    @property
    def split(self) -> str:
        return self.ref.split

    @property
    def scenario_id(self) -> str:
        return self.ref.scenario_id

    @property
    def camera_id(self) -> str:
        return camera_id_from_video(self.scenario_id, self.video_name)

    @property
    def base_video(self) -> str:
        return self.video_name.replace(".mp4", "")

def _vqa_split_root(split: str) -> Path:
    return VQA_DIR / split

def _scenario_rel_from_json(split: str, path: Path) -> Path:
    return path.parent.parent.relative_to(_vqa_split_root(split))

def _vqa_json_path(ref: ScenarioRef, view: str) -> Path:
    return VQA_DIR / ref.split / ref.rel_path / view / f"{ref.scenario_id}.json"

def _video_path(ref: ScenarioRef, view: str, video_name: str) -> Path:
    return VIDEOS_DIR / ref.split / ref.rel_path / view / video_name

def _bbox_path(ref: ScenarioRef, view: str, video_name: str, *, vehicle: bool) -> Path:
    root = BBOX_VEH_DIR if vehicle else BBOX_PED_DIR
    base = video_name.replace(".mp4", "")
    return root / ref.split / ref.rel_path / view / f"{base}_bbox.json"

def discover_vqa_scenarios(split: str) -> list[ScenarioRef]:
    root = _vqa_split_root(split)
    refs: dict[str, ScenarioRef] = {}
    if not root.exists():
        return []
    for view in VIEWS:
        for path in root.glob(f"**/{view}/*.json"):
            rel_path = _scenario_rel_from_json(split, path)
            key = rel_path.as_posix()
            refs[key] = ScenarioRef(split=split, rel_path=rel_path, scenario_id=path.stem)
    return [refs[k] for k in sorted(refs)]

def discover_caption_scenarios(split: str) -> list[ScenarioRef]:
    """All scenarios that have caption GT — including the normal_trimmed
    scenarios that have NO VQA file (62 in train as of the 2026-05-21 update).

    Mirrors discover_vqa_scenarios but globs ``*_caption.json`` under the
    caption tree so caption-only (BDD-domain) scenarios are not dropped from
    frame extraction or the caption dataset.
    """
    root = CAPTION_DIR / split
    refs: dict[str, ScenarioRef] = {}
    if not root.exists():
        return []
    for view in ("overhead_view", "vehicle_view"):
        for path in root.glob(f"**/{view}/*_caption.json"):
            rel_path = path.parent.parent.relative_to(root)
            scenario_id = path.stem[: -len("_caption")] if path.stem.endswith("_caption") else path.stem
            key = rel_path.as_posix()
            refs[key] = ScenarioRef(split=split, rel_path=rel_path, scenario_id=scenario_id)
    return [refs[k] for k in sorted(refs)]

def discover_all_scenarios(split: str) -> list[ScenarioRef]:
    """Union of VQA + caption scenarios, deduplicated by rel_path. Used for
    frame extraction so every scenario with ANY GT (VQA or caption) gets
    frames — otherwise caption-only scenarios have no frames and their
    caption samples are silently dropped."""
    refs: dict[str, ScenarioRef] = {}
    for ref in discover_vqa_scenarios(split):
        refs[ref.rel_path.as_posix()] = ref
    for ref in discover_caption_scenarios(split):
        refs.setdefault(ref.rel_path.as_posix(), ref)
    return [refs[k] for k in sorted(refs)]

def _read_vqa_entries(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = read_json(path)
    return data if isinstance(data, list) else []

def _read_scene_entries(ref: ScenarioRef, view: str) -> list[dict]:
    """Scene metadata (video list + event_phase) for a (scenario, view),
    reading the VQA JSON when present and falling back to the CAPTION JSON.

    The SynWTS 2026-05-21 update added normal_trimmed scenarios that have
    caption GT but NO VQA file. The frame-extraction + video-resolver path
    originally read only VQA JSONs, so those 62 caption-only scenarios were
    silently dropped (no frames, no caption samples). Both JSON kinds expose
    the same ``overhead_videos`` / ``vehicle_view`` + ``event_phase`` fields;
    the only difference is the VQA file is a LIST of entries while the caption
    file is a single DICT — we normalise both to a list of entry-dicts here.
    """
    vqa = _read_vqa_entries(_vqa_json_path(ref, view))
    if vqa:
        return vqa
    cap_path = _caption_json_path(ref, view)
    if cap_path.exists():
        data = read_json(cap_path)
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
    return []

def _phase_records(ref: ScenarioRef, view: str) -> list[dict]:
    phases: list[dict] = []
    for entry in _read_scene_entries(ref, view):
        for phase in entry.get("event_phase", []):
            phases.append(phase)
    return phases

def _overhead_videos_for(ref: ScenarioRef) -> list[str]:
    keep: list[str] = []
    for entry in _read_scene_entries(ref, "overhead_view"):
        for video_name in entry.get("overhead_videos") or []:
            if (
                _video_path(ref, "overhead_view", video_name).exists()
                and _bbox_path(ref, "overhead_view", video_name, vehicle=False).exists()
                and _bbox_path(ref, "overhead_view", video_name, vehicle=True).exists()
            ):
                keep.append(video_name)
    return sorted(set(keep))

def _vehicle_video_for(ref: ScenarioRef) -> str | None:
    entries = _read_scene_entries(ref, "vehicle_view")
    if not entries:
        return None
    candidates: list[str] = []
    for entry in entries:
        named = entry.get("vehicle_view")
        candidates.append(named or f"{ref.scenario_id}_vehicle_view.mp4")
    for video_name in sorted(set(candidates)):
        if (
            _video_path(ref, "vehicle_view", video_name).exists()
            and _bbox_path(ref, "vehicle_view", video_name, vehicle=False).exists()
        ):
            return video_name
    return None

def _score_overhead_video(ref: ScenarioRef, video_name: str) -> float:
    vid_path = _video_path(ref, "overhead_view", video_name)
    if not vid_path.exists():
        return 0.0
    cv_cap = cv2.VideoCapture(str(vid_path))
    if not cv_cap.isOpened():
        return 0.0
    fps = cv_cap.get(cv2.CAP_PROP_FPS) or 30.0
    cv_cap.release()

    ped_records = load_bbox_records(_bbox_path(ref, "overhead_view", video_name, vehicle=False))
    veh_records = load_bbox_records(_bbox_path(ref, "overhead_view", video_name, vehicle=True))

    score = 0.0
    for phase in _phase_records(ref, "overhead_view"):
        label = (phase.get("labels") or [""])[0]
        try:
            phase_num = to_phase_num(label)
            t_start = float(phase.get("start_time", 0.0))
            t_end = float(phase.get("end_time", t_start))
        except (TypeError, ValueError):
            continue
        t_mid = (t_start + max(t_end, t_start)) / 2.0
        frame_id = max(0, int(t_mid * fps))
        ped_bb = _nearest_bbox(_bbox_by_image(ped_records, phase_num), frame_id)
        veh_bb = _nearest_bbox(_bbox_by_image(veh_records, phase_num), frame_id)
        score += (ped_bb.area if ped_bb else 0.0) + (veh_bb.area if veh_bb else 0.0)
    return score

@lru_cache(maxsize=4096)
def _best_overhead_video(split: str, rel_key: str, scenario_id: str) -> str | None:
    ref = ScenarioRef(split=split, rel_path=Path(rel_key), scenario_id=scenario_id)
    candidates = _overhead_videos_for(ref)
    if not candidates:
        return None
    scored = [(v, _score_overhead_video(ref, v)) for v in candidates]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[0][0]

def best_overhead_video(ref: ScenarioRef) -> str | None:
    return _best_overhead_video(ref.split, ref.rel_key, ref.scenario_id)

def _build_jobs(splits: tuple[str, ...] = ("train", "val")) -> list[_Job]:
    jobs: list[_Job] = []
    for split in splits:
        # discover_all_scenarios = VQA ∪ caption, so caption-only normal_trimmed
        # scenarios also get frames extracted (resolvers fall back to caption
        # JSON via _read_scene_entries).
        for ref in discover_all_scenarios(split):
            best_ov = best_overhead_video(ref)
            if best_ov is not None:
                jobs.append(_Job(ref, "overhead_view", best_ov))
            vehicle = _vehicle_video_for(ref)
            if vehicle is not None:
                jobs.append(_Job(ref, "vehicle_view", vehicle))
    return jobs

_GAZE_KW = frozenset({
    "aware", "awareness", "notice", "noticed", "looking", "look",
    "gaze", "attention", "attentive", "see", "visible",
    "line_of_sight", "visual_status", "perceive", "perceiving",
    "eye", "watch", "watching", "observe", "observing",
})

_APPEARANCE_KW = frozenset({
    "wearing", "clothing", "clothes", "outfit", "dressed", "dress",
    "color", "colour", "shirt", "jacket", "pants", "trousers", "shoes",
    "hat", "cap", "helmet", "bag", "backpack", "carrying", "holding",
    "age", "gender", "male", "female", "appearance", "hair", "height",
})

_POSITION_KW = frozenset({
    "position", "located", "location", "where", "distance",
    "far", "close", "near", "meter", "relative", "proximity",
    "how far", "standing", "placed",
})

_ENV_KW = frozenset({
    "weather", "lighting", "light", "road", "traffic", "signal",
    "environment", "condition", "scene", "time", "day", "night",
    "rain", "sunny", "dark", "bright", "intersection", "street",
})

def classify_question(question: str) -> str:
    """Map question text to a routing category."""
    q = question.lower()
    words = set(re.findall(r'\b\w+\b', q))

    has_vehicle_focus = bool(words & {"vehicle", "car", "bus", "truck", "driver"})
    has_ped_focus = bool(words & {"pedestrian", "person", "walker", "people"})

    # Gaze/awareness — hard questions, need cross-view
    # "visual status" and "field of view" are gaze-class but keywords have spaces/compound forms
    if (
        words & _GAZE_KW
        or "line of sight" in q
        or "visual status" in q
        or "field of view" in q
    ):
        return "gaze"

    # Appearance — need clearest ped frame
    if words & _APPEARANCE_KW:
        return "appearance"

    # Vehicle-focused questions
    if has_vehicle_focus and not has_ped_focus:
        if words & {"position", "distance", "where", "location", "far", "near", "close"}:
            return "vehicle_position"
        return "vehicle_action"

    # Position/distance — "orientation" describes spatial relationship → position frame
    if words & _POSITION_KW or "orientation" in words:
        return "position"

    # Environment
    if words & _ENV_KW:
        return "environment"

    # Default: action/trajectory
    return "action"

VQA_PROMPT_TEMPLATE = (
    "{image_context}\n"
    "Phase: {phase_name}.\n"
    "Question: {question}\n"
    "Options:\n{options}\n"
    "Choose the best option and answer with the letter followed by the full option text."
)

CAPTION_PROMPT_TEMPLATE = (
    "{image_context}\n"
    "Phase: {phase_name}.\n"
    "Describe the traffic scenario in detail."
)

def format_options(opts: dict) -> str:
    return "\n".join(
        f"({k}) {opts[k]}"
        for k in ("a", "b", "c", "d")
        if k in opts and opts[k] is not None
    )

def _frame_suffix_from_path(path: Path) -> str:
    m = re.search(r"_phase[^_]+_([^_]+)\.jpg$", path.name)
    return m.group(1) if m else "selected"

def _image_kind_from_path(path: Path) -> str:
    return "local crop" if FRAMES_LOCAL.name in path.parts else "global view"

def _frame_suffix_description(suffix: str) -> str:
    return {
        "mid": "middle of phase",
        "ens": "middle/end of phase",
        "end": "end of phase",
        "clearest": "clearest pedestrian view",
        "both": "both pedestrian and vehicle visible",
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

def _format_vqa_prompt(*, phase_name: str, paths: Iterable[Path], question: str, options: str) -> str:
    return VQA_PROMPT_TEMPLATE.format(
        phase_name=phase_name,
        image_context=_image_context_for_paths(paths),
        question=question,
        options=options,
    )

PED_COLOR = (0, 255, 0)

VEH_COLOR = (0, 0, 255)

@dataclass(frozen=True)
class _SelectedFrame:
    suffix: str
    frame_id: int
    ped_bb: BBox | None
    veh_bb: BBox | None

def _resize_keep_aspect(img, max_side: int):
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img
    scale = max_side / longest
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

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
    crop_w, crop_h = int(w * ratio), int(h * ratio)
    x0, y0 = (w - crop_w) // 2, (h - crop_h) // 2
    return frame[y0:y0 + crop_h, x0:x0 + crop_w]

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
    mid_frame = _clip_frame_id(max(0, int(((t_start + t_end) / 2.0) * fps)), start_frame, end_frame)
    return start_frame, mid_frame, end_frame

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
    scan = range(anchor_frame, boundary + (1 if direction > 0 else -1), 1 if direction > 0 else -1)

    # Pass 1: find frame with both ped + veh bbox
    for frame_id in scan:
        ped_bb = ped_by_image.get(frame_id)
        veh_bb = veh_by_image.get(frame_id)
        if ped_bb is not None and veh_bb is not None:
            return _SelectedFrame(suffix, frame_id, ped_bb, veh_bb)

    # Pass 2: find frame with at least one bbox
    for frame_id in scan:
        ped_bb = ped_by_image.get(frame_id)
        veh_bb = veh_by_image.get(frame_id)
        if ped_bb is not None or veh_bb is not None:
            return _SelectedFrame(suffix, frame_id, ped_bb, veh_bb)

    # Fallback: use anchor frame with nearest available bboxes
    ped_bb = _nearest_bbox(ped_by_image, anchor_frame)
    veh_bb = _nearest_bbox(veh_by_image, anchor_frame)
    return _SelectedFrame(suffix, anchor_frame, ped_bb, veh_bb)

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
    representative = mid_frame if _bbox_count(mid_frame) >= _bbox_count(end_frame) else end_frame
    return [_with_suffix(representative, "ens")]

def _find_clearest_ped_frame(
    ped_by_image: dict[int, BBox],
    start_frame: int,
    end_frame: int,
) -> tuple[int, BBox] | None:
    """Return (frame_id, bbox) for frame with largest ped bbox in [start, end]."""
    candidates = {
        fid: bb for fid, bb in ped_by_image.items()
        if start_frame <= fid <= end_frame
    }
    if not candidates:
        return None
    best_fid = max(candidates, key=lambda fid: candidates[fid].area)
    return best_fid, candidates[best_fid]

def _find_both_visible_frame(
    ped_by_image: dict[int, BBox],
    veh_by_image: dict[int, BBox],
    start_frame: int,
    end_frame: int,
) -> tuple[int, BBox, BBox] | None:
    """Return (frame_id, ped_bbox, veh_bbox) for frame where both are visible."""
    ped_frames = {fid for fid in ped_by_image if start_frame <= fid <= end_frame}
    veh_frames = {fid for fid in veh_by_image if start_frame <= fid <= end_frame}
    common = ped_frames & veh_frames
    if not common:
        return None
    best_fid = max(common, key=lambda fid: ped_by_image[fid].area + veh_by_image[fid].area)
    return best_fid, ped_by_image[best_fid], veh_by_image[best_fid]

def _clear_phase_outputs(out_g: Path, out_l: Path | None, base: str, phase_num: str) -> None:
    for path in out_g.glob(f"{base}_phase{phase_num}_*.jpg"):
        path.unlink()
    if out_l is not None:
        for path in out_l.glob(f"{base}_phase{phase_num}_*.jpg"):
            path.unlink()

def _write_frame_jpg(
    cv_cap,
    frame_id: int,
    out_g: Path,
    out_l: Path | None,
    base: str,
    phase_num: str,
    suffix: str,
    ped_bb: BBox | None,
    veh_bb: BBox | None,
    local_crop: str = "union",  # "union" | "ped" | "none"
) -> bool:
    cv_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cv_cap.read()
    if not ok or frame is None:
        return False

    h, w = frame.shape[:2]
    drawn = frame.copy()
    _draw_bbox(drawn, ped_bb, PED_COLOR, "Pedestrian")
    _draw_bbox(drawn, veh_bb, VEH_COLOR, "Vehicle")

    global_img = _resize_keep_aspect(drawn.copy(), FRAME_CFG.global_max_side)
    cv2.imwrite(
        str(out_g / f"{base}_phase{phase_num}_{suffix}.jpg"),
        global_img,
        [cv2.IMWRITE_JPEG_QUALITY, FRAME_CFG.jpeg_quality],
    )

    if out_l is not None and local_crop != "none":
        if local_crop == "ped" and ped_bb is not None:
            x0, y0, x1, y1 = pad_bbox(ped_bb, FRAME_CFG.local_pad_ratio, w, h)
        else:
            united = union_bbox(ped_bb, veh_bb)
            if united is not None:
                x0, y0, x1, y1 = pad_bbox(united, FRAME_CFG.local_pad_ratio, w, h)
            else:
                x0, y0, x1, y1 = 0, 0, w, h
        local_img = drawn[y0:y1, x0:x1]
        if local_img.size == 0:
            local_img = _center_crop(drawn)
        local_img = _resize_keep_aspect(local_img, FRAME_CFG.local_max_side)
        cv2.imwrite(
            str(out_l / f"{base}_phase{phase_num}_{suffix}.jpg"),
            local_img,
            [cv2.IMWRITE_JPEG_QUALITY, FRAME_CFG.jpeg_quality],
        )

    return True

def _process_job(job: _Job) -> int:
    is_overhead = job.view == "overhead_view"
    vid_path = _video_path(job.ref, job.view, job.video_name)
    if not vid_path.exists():
        return 0

    base = job.base_video
    ped_records = load_bbox_records(_bbox_path(job.ref, job.view, job.video_name, vehicle=False))
    veh_records = (
        load_bbox_records(_bbox_path(job.ref, job.view, job.video_name, vehicle=True))
        if is_overhead else []
    )

    cv_cap = cv2.VideoCapture(str(vid_path))
    if not cv_cap.isOpened():
        return 0
    fps = cv_cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cv_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    out_g = FRAMES_GLOBAL / job.split / job.ref.rel_path / job.view
    out_g.mkdir(parents=True, exist_ok=True)
    out_l: Path | None = None
    # Both overhead AND vehicle_view get local crops (vehicle for cross-view gaze)
    out_l = FRAMES_LOCAL / job.split / job.ref.rel_path / job.view
    out_l.mkdir(parents=True, exist_ok=True)

    written = 0
    for phase in _phase_records(job.ref, job.view):
        label = (phase.get("labels") or [""])[0]
        try:
            phase_num = to_phase_num(label)
            t_start = float(phase.get("start_time", 0.0))
            t_end = float(phase.get("end_time", t_start))
        except (TypeError, ValueError):
            continue
        if t_end < t_start:
            t_end = t_start

        start_frame, mid_anchor, end_anchor = _phase_frame_bounds(t_start, t_end, fps, total_frames)
        _clear_phase_outputs(out_g, out_l, base, phase_num)

        ped_by_image = _bbox_by_image(ped_records, phase_num)

        if is_overhead:
            veh_by_image = _bbox_by_image(veh_records, phase_num)

            # --- mid + end frames (trajectory) ---
            trajectory = _merge_close_mid_end(
                _select_overhead_frame(
                    suffix="mid", anchor_frame=mid_anchor,
                    start_frame=start_frame, end_frame=end_anchor,
                    direction=1,
                    ped_by_image=ped_by_image, veh_by_image=veh_by_image,
                ),
                _select_overhead_frame(
                    suffix="end", anchor_frame=end_anchor,
                    start_frame=start_frame, end_frame=end_anchor,
                    direction=-1,
                    ped_by_image=ped_by_image, veh_by_image=veh_by_image,
                ),
            )
            for sel in trajectory:
                if _write_frame_jpg(cv_cap, sel.frame_id, out_g, out_l, base, phase_num,
                                    sel.suffix, sel.ped_bb, sel.veh_bb, local_crop="union"):
                    written += 1

            # --- clearest ped frame: ped-only local crop ---
            clearest = _find_clearest_ped_frame(ped_by_image, start_frame, end_anchor)
            if clearest is not None:
                fid, ped_bb = clearest
                veh_bb = _nearest_bbox(veh_by_image, fid)
                if _write_frame_jpg(cv_cap, fid, out_g, out_l, base, phase_num,
                                    "clearest", ped_bb, veh_bb, local_crop="ped"):
                    written += 1

            # --- both_visible frame: union local crop ---
            both = _find_both_visible_frame(ped_by_image, veh_by_image, start_frame, end_anchor)
            if both is not None:
                fid, ped_bb, veh_bb = both
                if _write_frame_jpg(cv_cap, fid, out_g, out_l, base, phase_num,
                                    "both", ped_bb, veh_bb, local_crop="union"):
                    written += 1

        else:
            # Vehicle view: mid + end + clearest (for cross-view gaze questions)
            mid_ped = _nearest_bbox(ped_by_image, mid_anchor)
            end_ped = _nearest_bbox(ped_by_image, end_anchor)
            trajectory = _merge_close_mid_end(
                _SelectedFrame("mid", mid_anchor, mid_ped, None),
                _SelectedFrame("end", end_anchor, end_ped, None),
            )
            for sel in trajectory:
                if _write_frame_jpg(cv_cap, sel.frame_id, out_g, out_l, base, phase_num,
                                    sel.suffix, sel.ped_bb, None, local_crop="none"):
                    written += 1
                # Also save global to out_g (already done above via out_l=None skipped)

            # Clearest ped from dashcam — ped local crop for cross-view
            clearest = _find_clearest_ped_frame(ped_by_image, start_frame, end_anchor)
            if clearest is not None:
                fid, ped_bb = clearest
                if _write_frame_jpg(cv_cap, fid, out_g, out_l, base, phase_num,
                                    "clearest", ped_bb, None, local_crop="ped"):
                    written += 1

    cv_cap.release()
    return written

def run_extract_frames(num_workers: int = 4) -> None:
    ensure_dirs()
    jobs = _build_jobs()
    print(f"[frames] {len(jobs)} selected view jobs")
    if num_workers <= 1:
        total = sum(_process_job(job) for job in tqdm(jobs, desc="frames"))
    else:
        with Pool(num_workers) as pool:
            total = sum(tqdm(pool.imap_unordered(_process_job, jobs), total=len(jobs), desc="frames"))
    print(f"[frames] wrote {total} selected phase frame sets")

def _phase_image_path(
    root: Path,
    ref: ScenarioRef,
    view: str,
    base_video: str,
    phase_num: str,
    suffix: str,
) -> Path:
    return root / ref.split / ref.rel_path / view / f"{base_video}_phase{phase_num}_{suffix}.jpg"

def _collect_existing(
    root: Path,
    ref: ScenarioRef,
    view: str,
    base_video: str | None,
    phase_num: str,
    suffixes: list[str],
) -> list[Path]:
    """Return existing paths for the given suffixes."""
    if base_video is None:
        return []
    paths = []
    for suffix in suffixes:
        p = _phase_image_path(root, ref, view, base_video, phase_num, suffix)
        if p.exists():
            paths.append(p)
    return paths

def _trajectory_paths(
    ref: ScenarioRef,
    view: str,
    base_video: str | None,
    phase_num: str,
) -> list[Path]:
    """Return mid+end global paths, falling back to ens if merged."""
    if base_video is None:
        return []
    paths = _collect_existing(FRAMES_GLOBAL, ref, view, base_video, phase_num, ["mid", "end"])
    if not paths:
        paths = _collect_existing(FRAMES_GLOBAL, ref, view, base_video, phase_num, ["ens"])
    return paths

def _trajectory_paths_local(
    ref: ScenarioRef,
    view: str,
    base_video: str | None,
    phase_num: str,
) -> list[Path]:
    """Return mid+end LOCAL (union crop) paths, falling back to ens if merged.

    Only overhead_view writes local mid/end (local_crop="union"); vehicle_view
    uses local_crop="none" so this returns [] for vehicle and is a no-op there.
    """
    if base_video is None:
        return []
    paths = _collect_existing(FRAMES_LOCAL, ref, view, base_video, phase_num, ["mid", "end"])
    if not paths:
        paths = _collect_existing(FRAMES_LOCAL, ref, view, base_video, phase_num, ["ens"])
    return paths

def _frame_paths_for_question(
    *,
    ref: ScenarioRef,
    view: str,
    overhead_base: str | None,
    vehicle_base: str | None,
    phase_num: str,
    q_type: str,
) -> tuple[Path, ...] | None:
    """Select image paths based on question type and view."""
    paths: list[Path] = []

    if view == "overhead_view" and overhead_base:
        if q_type == "gaze":
            # Ped-only local crop (upscale) + trajectory + dashcam cross-view
            paths += _collect_existing(FRAMES_LOCAL, ref, "overhead_view", overhead_base, phase_num, ["clearest"])
            paths += _trajectory_paths(ref, "overhead_view", overhead_base, phase_num)
            if vehicle_base:
                paths += _collect_existing(FRAMES_GLOBAL, ref, "vehicle_view", vehicle_base, phase_num, ["clearest"])
                paths += _collect_existing(FRAMES_LOCAL, ref, "vehicle_view", vehicle_base, phase_num, ["clearest"])

        elif q_type == "appearance":
            paths += _collect_existing(FRAMES_GLOBAL, ref, "overhead_view", overhead_base, phase_num, ["clearest"])
            paths += _collect_existing(FRAMES_LOCAL, ref, "overhead_view", overhead_base, phase_num, ["clearest"])

        elif q_type == "position":
            # Both visible (spatial context) + clearest ped for detail
            paths += _collect_existing(FRAMES_GLOBAL, ref, "overhead_view", overhead_base, phase_num, ["both"])
            paths += _collect_existing(FRAMES_LOCAL, ref, "overhead_view", overhead_base, phase_num, ["both"])
            paths += _collect_existing(FRAMES_LOCAL, ref, "overhead_view", overhead_base, phase_num, ["clearest"])

        elif q_type == "environment":
            paths += _collect_existing(FRAMES_GLOBAL, ref, "overhead_view", overhead_base, phase_num, ["mid", "ens"])

        elif q_type == "vehicle_position":
            paths += _collect_existing(FRAMES_GLOBAL, ref, "overhead_view", overhead_base, phase_num, ["both"])
            paths += _collect_existing(FRAMES_LOCAL, ref, "overhead_view", overhead_base, phase_num, ["both"])
            paths += _collect_existing(FRAMES_LOCAL, ref, "overhead_view", overhead_base, phase_num, ["clearest"])

        elif q_type == "vehicle_action":
            paths += _trajectory_paths(ref, "overhead_view", overhead_base, phase_num)

        else:  # action / default — ped trajectory: global full frame + zoomed union crop
            paths += _trajectory_paths(ref, "overhead_view", overhead_base, phase_num)
            paths += _trajectory_paths_local(ref, "overhead_view", overhead_base, phase_num)

    elif view == "vehicle_view" and vehicle_base:
        if q_type == "gaze":
            # Dashcam is great for gaze — clearest ped crop first
            paths += _collect_existing(FRAMES_GLOBAL, ref, "vehicle_view", vehicle_base, phase_num, ["clearest"])
            paths += _collect_existing(FRAMES_LOCAL, ref, "vehicle_view", vehicle_base, phase_num, ["clearest"])
            paths += _trajectory_paths(ref, "vehicle_view", vehicle_base, phase_num)

        elif q_type == "appearance":
            paths += _collect_existing(FRAMES_GLOBAL, ref, "vehicle_view", vehicle_base, phase_num, ["clearest"])
            paths += _collect_existing(FRAMES_LOCAL, ref, "vehicle_view", vehicle_base, phase_num, ["clearest"])

        else:  # action, position, vehicle_*, environment, default
            paths += _trajectory_paths(ref, "vehicle_view", vehicle_base, phase_num)

    # Fallback: any available frame for this view.
    # Try suffixes in priority order matched to q_type context.
    if not paths:
        base = overhead_base if view == "overhead_view" else vehicle_base
        if q_type == "environment":
            fallback_order = ("mid", "ens", "end", "both", "clearest")
        elif q_type in ("position", "vehicle_position"):
            fallback_order = ("both", "mid", "ens", "clearest", "end")
        else:
            fallback_order = ("mid", "ens", "clearest", "both", "end")
        for suffix in fallback_order:
            p = _collect_existing(FRAMES_GLOBAL, ref, view, base, phase_num, [suffix])
            if not p:
                p = _collect_existing(FRAMES_LOCAL, ref, view, base, phase_num, [suffix])
            if p:
                paths += p
                break

    return tuple(paths) if paths else None

def _vqa_sample(
    *,
    ref: ScenarioRef,
    view: str,
    camera_id: str,
    phase_num: str,
    phase_name: str,
    qid: str,
    conv: dict,
    images: tuple[Path, ...],
    base_video: str,
) -> dict | None:
    correct = str(conv.get("correct", "")).strip().lower()
    if correct not in ("a", "b", "c", "d"):
        return None
    question = (conv.get("question") or "").strip()
    # Answer = "c. full option text" for richer training supervision
    option_text = (conv.get(correct) or "").strip()
    answer = f"{correct}. {option_text}" if option_text else correct

    return {
        "id": qid,
        "split": ref.split,
        "scenario": ref.scenario_id,
        "scenario_path": ref.rel_key,
        "view": view,
        "camera_id": camera_id,
        "base_video": base_video,
        "phase_label": phase_num,
        "phase_name": phase_name,
        "question": question,
        "options": {k: conv.get(k) for k in ("a", "b", "c", "d") if k in conv},
        "images": [str(p) for p in images],
        "conversations": [
            {
                "from": "user",
                "value": _format_vqa_prompt(
                    phase_name=phase_name,
                    paths=images,
                    question=question,
                    options=format_options(conv),
                ),
            },
            {"from": "assistant", "value": answer},
        ],
    }

def _question_id(conv: dict, fallback: str) -> str:
    for key in ("id", "question_id", "qid", "uuid"):
        value = conv.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return fallback

def _process_vqa_file(
    *,
    ref: ScenarioRef,
    view: str,
    base_video: str,
    camera_id: str,
    json_path: Path,
    vehicle_base: str | None = None,
) -> list[dict]:
    overhead_base = base_video if view == "overhead_view" else None
    veh_base = base_video if view == "vehicle_view" else vehicle_base

    out: list[dict] = []
    for entry in _read_vqa_entries(json_path):
        for phase in entry.get("event_phase", []):
            label = (phase.get("labels") or [""])[0]
            try:
                phase_num = to_phase_num(label)
            except ValueError:
                continue
            phase_name = _phase_name(label)
            for idx, conv in enumerate(phase.get("conversations", [])):
                fallback_qid = f"{ref.scenario_id}_{camera_id}_phase{phase_num}_q{idx}"
                qid = _question_id(conv, fallback_qid)
                question = (conv.get("question") or "").strip()
                q_type = classify_question(question)

                paths = _frame_paths_for_question(
                    ref=ref,
                    view=view,
                    overhead_base=overhead_base,
                    vehicle_base=veh_base,
                    phase_num=phase_num,
                    q_type=q_type,
                )
                if paths is None:
                    continue

                sample = _vqa_sample(
                    ref=ref,
                    view=view,
                    camera_id=camera_id,
                    phase_num=phase_num,
                    phase_name=phase_name,
                    qid=qid,
                    conv=conv,
                    images=paths,
                    base_video=base_video,
                )
                if sample is not None:
                    out.append(sample)
    return out

_ENV_PHASE_NUM = "2"

_ENV_PHASE_NAME = "judgement"

def _env_extra_paths(
    ref: ScenarioRef,
    base_video: str,
    q_type: str,
) -> list[Path]:
    """Extra frames added on top of phase-2 mid/ens for environment questions."""
    v = "overhead_view"
    p = _ENV_PHASE_NUM
    if q_type == "appearance":
        return (
            _collect_existing(FRAMES_GLOBAL, ref, v, base_video, p, ["clearest"])
            + _collect_existing(FRAMES_LOCAL, ref, v, base_video, p, ["clearest"])
        )
    if q_type == "gaze":
        return _collect_existing(FRAMES_LOCAL, ref, v, base_video, p, ["clearest"])
    if q_type in ("position", "vehicle_position"):
        return (
            _collect_existing(FRAMES_GLOBAL, ref, v, base_video, p, ["both"])
            + _collect_existing(FRAMES_LOCAL, ref, v, base_video, p, ["both"])
            + _collect_existing(FRAMES_LOCAL, ref, v, base_video, p, ["clearest"])
        )
    return []

def _process_env_file(
    *,
    ref: ScenarioRef,
    base_video: str,
    camera_id: str,
    json_path: Path,
) -> list[dict]:
    # Base: global mid/ens of phase 2 — scene-level context for all env questions
    base_paths = _collect_existing(
        FRAMES_GLOBAL, ref, "overhead_view", base_video, _ENV_PHASE_NUM, ["mid", "ens"]
    )
    if not base_paths:
        return []
    out: list[dict] = []
    for entry in _read_vqa_entries(json_path):
        for idx, conv in enumerate(entry.get("environment", [])):
            fallback_qid = f"{ref.scenario_id}_{camera_id}_environment_q{idx}"
            qid = _question_id(conv, fallback_qid)
            question = (conv.get("question") or "").strip()
            q_type = classify_question(question)
            # base scene frame + question-specific extra frames
            paths = tuple(base_paths + _env_extra_paths(ref, base_video, q_type))
            sample = _vqa_sample(
                ref=ref,
                view="overhead_view",
                camera_id=camera_id,
                phase_num=_ENV_PHASE_NUM,
                phase_name=_ENV_PHASE_NAME,
                qid=qid,
                conv=conv,
                images=paths,
                base_video=base_video,
            )
            if sample is not None:
                out.append(sample)
    return out

def _build_vqa_split(split: str) -> list[dict]:
    samples: list[dict] = []
    for ref in tqdm(discover_vqa_scenarios(split), desc=f"vqa[{split}]"):
        best_ov = best_overhead_video(ref)
        vehicle = _vehicle_video_for(ref)
        vehicle_base = vehicle.replace(".mp4", "") if vehicle else None

        if best_ov is not None:
            base = best_ov.replace(".mp4", "")
            camera_id = camera_id_from_video(ref.scenario_id, best_ov)
            samples.extend(_process_vqa_file(
                ref=ref,
                view="overhead_view",
                base_video=base,
                camera_id=camera_id,
                json_path=_vqa_json_path(ref, "overhead_view"),
                vehicle_base=vehicle_base,
            ))
            samples.extend(_process_env_file(
                ref=ref,
                base_video=base,
                camera_id=camera_id,
                json_path=_vqa_json_path(ref, "environment"),
            ))

        if vehicle is not None and vehicle_base is not None:
            camera_id = camera_id_from_video(ref.scenario_id, vehicle)
            samples.extend(_process_vqa_file(
                ref=ref,
                view="vehicle_view",
                base_video=vehicle_base,
                camera_id=camera_id,
                json_path=_vqa_json_path(ref, "vehicle_view"),
                vehicle_base=vehicle_base,
            ))
    return samples

def run_build_vqa(splits: tuple[str, ...] = ("train", "val")) -> None:
    ensure_dirs()
    for split in splits:
        samples = _build_vqa_split(split)
        out = Path(str(VQA_DATASET).format(split=split))
        write_json(out, samples)
        counts = Counter(sample["question"] for sample in samples)
        print(f"[vqa] {split}: {len(samples)} samples -> {out}")
        for question, count in counts.most_common(10):
            print(f"[vqa]   {count:5d} | {question}")

def _caption_json_path(ref: ScenarioRef, view: str) -> Path:
    return CAPTION_DIR / ref.split / ref.rel_path / view / f"{ref.scenario_id}_caption.json"

def _caption_sample(
    *,
    ref: ScenarioRef,
    view: str,
    camera_id: str,
    phase_num: str,
    phase_name: str,
    caption_id: str,
    caption_text: str,
    images: tuple[Path, ...],
    base_video: str,
) -> dict:
    prompt = CAPTION_PROMPT_TEMPLATE.format(
        image_context=_image_context_for_paths(images),
        phase_name=phase_name,
    )
    return {
        "id": caption_id,
        "split": ref.split,
        "scenario": ref.scenario_id,
        "scenario_path": ref.rel_key,
        "view": view,
        "camera_id": camera_id,
        "base_video": base_video,
        "phase_label": phase_num,
        "phase_name": phase_name,
        "question": "Describe the traffic scenario in detail.",
        "options": {},
        "images": [str(p) for p in images],
        "conversations": [
            {"from": "user", "value": prompt},
            {"from": "assistant", "value": caption_text},
        ],
    }

def _build_caption_split(split: str) -> list[dict]:
    samples: list[dict] = []
    for ref in tqdm(discover_caption_scenarios(split), desc=f"caption[{split}]"):
        best_ov = best_overhead_video(ref)
        vehicle = _vehicle_video_for(ref)

        if best_ov is not None:
            base = best_ov.replace(".mp4", "")
            cam_id = camera_id_from_video(ref.scenario_id, best_ov)
            cap_path = _caption_json_path(ref, "overhead_view")
            if cap_path.exists():
                try:
                    data = read_json(cap_path)
                except Exception:
                    data = {}
                for phase in data.get("event_phase", []):
                    text = (phase.get("caption_pedestrian") or "").strip()
                    if not text:
                        continue
                    label = (phase.get("labels") or [""])[0]
                    try:
                        phase_num = to_phase_num(label)
                    except ValueError:
                        continue
                    paths = tuple(_trajectory_paths(ref, "overhead_view", base, phase_num))
                    if not paths:
                        continue
                    samples.append(_caption_sample(
                        ref=ref, view="overhead_view", camera_id=cam_id,
                        phase_num=phase_num, phase_name=_phase_name(label),
                        caption_id=f"{ref.scenario_id}_{cam_id}_caption_ped_phase{phase_num}",
                        caption_text=text, images=paths, base_video=base,
                    ))

        if vehicle is not None:
            veh_base = vehicle.replace(".mp4", "")
            cam_id = camera_id_from_video(ref.scenario_id, vehicle)
            cap_path = _caption_json_path(ref, "vehicle_view")
            if cap_path.exists():
                try:
                    data = read_json(cap_path)
                except Exception:
                    data = {}
                for phase in data.get("event_phase", []):
                    text = (phase.get("caption_vehicle") or "").strip()
                    if not text:
                        continue
                    label = (phase.get("labels") or [""])[0]
                    try:
                        phase_num = to_phase_num(label)
                    except ValueError:
                        continue
                    paths = tuple(_trajectory_paths(ref, "vehicle_view", veh_base, phase_num))
                    if not paths:
                        continue
                    samples.append(_caption_sample(
                        ref=ref, view="vehicle_view", camera_id=cam_id,
                        phase_num=phase_num, phase_name=_phase_name(label),
                        caption_id=f"{ref.scenario_id}_{cam_id}_caption_veh_phase{phase_num}",
                        caption_text=text, images=paths, base_video=veh_base,
                    ))

    return samples

def run_build_caption() -> None:
    ensure_dirs()
    samples = _build_caption_split("train")
    write_json(CAPTION_DATASET, samples)
    print(f"[caption] train: {len(samples)} samples -> {CAPTION_DATASET}")

def _iter_raw_vqa_files(split: str, view: str) -> list[Path]:
    root = _vqa_split_root(split)
    if not root.exists():
        return []
    return sorted(root.glob(f"**/{view}/*.json"))

def _iter_conversations(path: Path, view: str) -> Iterable[tuple[str, str, dict]]:
    for entry in _read_vqa_entries(path):
        if view == "environment":
            for conv in entry.get("environment", []):
                yield _ENV_PHASE_NAME, _ENV_PHASE_NUM, conv
        else:
            for phase in entry.get("event_phase", []):
                label = (phase.get("labels") or [""])[0]
                try:
                    phase_num = to_phase_num(label)
                except ValueError:
                    continue
                for conv in phase.get("conversations", []):
                    yield _phase_name(label), phase_num, conv

def collect_vqa_data_stats(splits: tuple[str, ...] = ("train", "val")) -> dict:
    stats: dict[str, Any] = {"data_root": str(DATA_ROOT), "splits": {}}
    for split in splits:
        refs = discover_vqa_scenarios(split)
        split_stats: dict[str, Any] = {
            "scenario_paths": len(refs),
            "json_files": {},
            "selected_jobs": {},
            "questions_by_source": {},
            "questions_by_phase": {},
            "question_types": {},
            "question_categories": {},
            "invalid_correct": 0,
        }
        selected_jobs: Counter[str] = Counter()
        for ref in refs:
            if _overhead_videos_for(ref):
                selected_jobs["overhead_view"] += 1
            if _vehicle_video_for(ref) is not None:
                selected_jobs["vehicle_view"] += 1
        split_stats["selected_jobs"] = dict(selected_jobs)

        by_source: Counter[str] = Counter()
        by_phase: Counter[str] = Counter()
        by_question: Counter[str] = Counter()
        by_category: Counter[str] = Counter()
        invalid = 0
        for view in VIEWS:
            files = _iter_raw_vqa_files(split, view)
            split_stats["json_files"][view] = len(files)
            for path in files:
                for phase_name, phase_num, conv in _iter_conversations(path, view):
                    correct = str(conv.get("correct", "")).strip().lower()
                    if correct not in ("a", "b", "c", "d"):
                        invalid += 1
                        continue
                    q = (conv.get("question") or "").strip()
                    by_source[view] += 1
                    by_phase[f"{view}:{phase_num}:{phase_name}"] += 1
                    by_question[q] += 1
                    by_category[classify_question(q)] += 1

        split_stats["questions_by_source"] = dict(by_source)
        split_stats["questions_by_phase"] = dict(by_phase)
        split_stats["question_types"] = dict(by_question.most_common())
        split_stats["question_categories"] = dict(by_category)
        split_stats["invalid_correct"] = invalid
        split_stats["total_questions"] = sum(by_source.values())
        stats["splits"][split] = split_stats
    return stats

def run_data_stats() -> None:
    ensure_dirs()
    stats = collect_vqa_data_stats()
    write_json(VQA_STATS, stats)
    print(f"[stats] wrote {VQA_STATS}")
    for split, split_stats in stats["splits"].items():
        print(f"\n[stats] {split}")
        print(f"  scenario_paths: {split_stats['scenario_paths']}")
        print(f"  selected_jobs: {split_stats['selected_jobs']}")
        print(f"  questions_by_source: {split_stats['questions_by_source']}")
        print(f"  question_categories: {split_stats['question_categories']}")
        print(f"  total_questions: {split_stats['total_questions']}")

def _frames_exist(splits: tuple[str, ...] = ("train", "val")) -> bool:
    for split in splits:
        split_dir = FRAMES_GLOBAL / split
        if not split_dir.exists() or not any(split_dir.rglob("*.jpg")):
            return False
        # Check for new frame types from the improved pipeline
        has_new = any(
            any(split_dir.rglob(f"*_phase*_{suffix}.jpg"))
            for suffix in ("clearest", "both")
        )
        if not has_new:
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
            if re.search(r"_phase[^_]+_(clearest|both|end|ens)\.jpg$", str(image_path)):
                return True
    return False

def run_preprocess_all(num_workers: int = 4, *, force: bool = False) -> None:
    ensure_dirs()
    run_data_stats()

    if force or not _frames_exist():
        run_extract_frames(num_workers=num_workers)
    else:
        print(f"[preprocess] skip extract_frames (frames found under {FRAMES_GLOBAL})")

    vqa_paths = [Path(str(VQA_DATASET).format(split=split)) for split in ("train", "val")]
    if force or not all(_dataset_has_selected_frame_paths(path) for path in vqa_paths):
        run_build_vqa()
    else:
        print("[preprocess] skip build_vqa (datasets exist)")

    if force or not CAPTION_DATASET.exists():
        run_build_caption()
    else:
        print(f"[preprocess] skip build_caption (exists: {CAPTION_DATASET})")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Preprocessing VQA (tách từ run7_qwen7b.py)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="làm lại dù output đã tồn tại")
    a = ap.parse_args()
    run_preprocess_all(num_workers=a.workers, force=a.force)


if __name__ == "__main__":
    main()
