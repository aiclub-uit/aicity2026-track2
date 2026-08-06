#!/usr/bin/env python
import json, os, sys, argparse, subprocess
sys.path.insert(0, '/workspace/AICC/code')
import cv2
try:
    import preprocess_vqa as R7
except ImportError:
    import run7_qwen7b as R7

OUT = '/workspace/AICC/code/cosmos_work'
RAW = f'{OUT}/raw'
META = f'{OUT}/meta.jsonl'
SPECS = f'{OUT}/specs'
RESTYLED = f'{OUT}/restyled'
PROC = '/workspace/AICC/code/output_qwen7b/processed'
DEST = '/workspace/AICC/code/output_qwen7b/processed_cosmos'
PROMPT = ("Real-world CCTV surveillance footage from a fixed overhead traffic camera at a road "
          "intersection. Photorealistic scene: worn asphalt with tire marks and faded lane markings, "
          "realistic pedestrians and vehicles with natural proportions, natural daylight with soft "
          "shadows, slight lens distortion, mild sensor noise and video compression artifacts typical "
          "of security cameras. Muted realistic colors.")


def _bb(b):
    return [b.x, b.y, b.w, b.h] if b is not None else None


def cmd_dump():
    jobs = [j for j in R7._build_jobs(("train",)) if j.view == "overhead_view"]
    print(f'{len(jobs)} overhead train jobs')
    os.makedirs(RAW, exist_ok=True)
    rows = []
    n_frames = 0
    for job in jobs:
        vid_path = R7._video_path(job.ref, job.view, job.video_name)
        if not vid_path.exists():
            continue
        base = job.base_video
        ped_records = R7.load_bbox_records(R7._bbox_path(job.ref, job.view, job.video_name, vehicle=False))
        veh_records = R7.load_bbox_records(R7._bbox_path(job.ref, job.view, job.video_name, vehicle=True))
        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        rel = f'{job.split}/{job.ref.rel_path}/{job.view}'
        dumped = {}
        def dump_frame(fid):
            if fid in dumped:
                return dumped[fid]
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, frame = cap.read()
            if not ok or frame is None:
                dumped[fid] = None; return None
            p = f'{RAW}/{rel}/{base}_f{fid}.jpg'
            os.makedirs(os.path.dirname(p), exist_ok=True)
            cv2.imwrite(p, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            dumped[fid] = (p, frame.shape[1], frame.shape[0])
            return dumped[fid]
        for phase in R7._phase_records(job.ref, job.view):
            label = (phase.get("labels") or [""])[0]
            try:
                phase_num = R7.to_phase_num(label)
                t0 = float(phase.get("start_time", 0.0)); t1 = float(phase.get("end_time", t0))
            except (TypeError, ValueError):
                continue
            if t1 < t0: t1 = t0
            sf, mid_a, end_a = R7._phase_frame_bounds(t0, t1, fps, total)
            ped_by = R7._bbox_by_image(ped_records, phase_num)
            veh_by = R7._bbox_by_image(veh_records, phase_num)
            sels = list(R7._merge_close_mid_end(
                R7._select_overhead_frame(suffix="mid", anchor_frame=mid_a, start_frame=sf,
                                          end_frame=end_a, direction=1, ped_by_image=ped_by, veh_by_image=veh_by),
                R7._select_overhead_frame(suffix="end", anchor_frame=end_a, start_frame=sf,
                                          end_frame=end_a, direction=-1, ped_by_image=ped_by, veh_by_image=veh_by)))
            outs = [(s.suffix, s.frame_id, s.ped_bb, s.veh_bb, "union") for s in sels]
            cl = R7._find_clearest_ped_frame(ped_by, sf, end_a)
            if cl is not None:
                fid, pbb = cl
                outs.append(("clearest", fid, pbb, R7._nearest_bbox(veh_by, fid), "ped"))
            bo = R7._find_both_visible_frame(ped_by, veh_by, sf, end_a)
            if bo is not None:
                fid, pbb, vbb = bo
                outs.append(("both", fid, pbb, vbb, "union"))
            for suffix, fid, pbb, vbb, mode in outs:
                d = dump_frame(fid)
                if d is None:
                    continue
                p, w, h = d
                rows.append(dict(raw=p, rel=rel, base=base, phase=phase_num, suffix=suffix,
                                 w=w, h=h, ped_bb=_bb(pbb), veh_bb=_bb(vbb), mode=mode,
                                 gname=f'{base}_phase{phase_num}_{suffix}.jpg'))
                n_frames += 1
        cap.release()
    with open(META, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    uniq = len({r['raw'] for r in rows})
    print(f'meta rows: {len(rows)} | unique raw frames: {uniq}')
    import glob
    exist = {os.path.basename(p) for p in glob.glob(f'{PROC}/frames_global/train/*/overhead_view/*.jpg')}
    got = {r['gname'] for r in rows}
    print(f'coverage vs processed: {len(got & exist)}/{len(exist)} (thiếu {len(exist-got)}, thừa {len(got-exist)})')


def cmd_spec():
    rows = [json.loads(l) for l in open(META)]
    used = {os.path.basename(p) for s in json.load(open(f'{PROC}/vqa_train.json'))
            for p in s['images'] if 'overhead_view' in p}
    rows = [r for r in rows if r['gname'] in used]
    print(f'meta sau lọc vqa_train: {len(rows)} rows')
    uniq = sorted({r['raw'] for r in rows})
    os.makedirs(SPECS, exist_ok=True)
    os.makedirs(f'{OUT}/mp4', exist_ok=True)
    specs = []
    for i, raw in enumerate(uniq):
        key = os.path.relpath(raw, RAW).replace('/', '__')[:-4]
        mp4 = f'{OUT}/mp4/{key}.mp4'
        if not os.path.exists(mp4):
            subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-loop', '1', '-i', raw,
                            '-frames:v', '1', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                            '-pix_fmt', 'yuv420p', mp4], check=True)
        spec = dict(name=key, prompt=PROMPT, video_path=mp4, num_video_frames_per_chunk=1,
                    max_frames=1, seed=1, edge={})
        sp = f'{SPECS}/{key}.json'
        json.dump(spec, open(sp, 'w'))
        specs.append(sp)
    print(f'{len(specs)} specs -> {SPECS}')


def cmd_rebuild():
    rows = [json.loads(l) for l in open(META)]
    n_ok = n_fb = 0
    for r in rows:
        key = os.path.relpath(r['raw'], RAW).replace('/', '__')[:-4]
        sty = f'{RESTYLED}/{key}.jpg'
        gdst = f'{DEST}/frames_global/{r["rel"]}/{r["gname"]}'
        ldst = f'{DEST}/frames_local/{r["rel"]}/{r["gname"]}'
        os.makedirs(os.path.dirname(gdst), exist_ok=True)
        os.makedirs(os.path.dirname(ldst), exist_ok=True)
        if os.path.exists(sty):
            img = cv2.imread(sty)
            sx, sy = img.shape[1] / r['w'], img.shape[0] / r['h']
            def scale_bb(b):
                if b is None: return None
                return R7.BBox(x=b[0]*sx, y=b[1]*sy, w=b[2]*sx, h=b[3]*sy)
            pbb, vbb = scale_bb(r['ped_bb']), scale_bb(r['veh_bb'])
            drawn = img.copy()
            R7._draw_bbox(drawn, pbb, R7.PED_COLOR, "Pedestrian")
            R7._draw_bbox(drawn, vbb, R7.VEH_COLOR, "Vehicle")
            g = R7._resize_keep_aspect(drawn.copy(), R7.FRAME_CFG.global_max_side)
            cv2.imwrite(gdst, g, [cv2.IMWRITE_JPEG_QUALITY, R7.FRAME_CFG.jpeg_quality])
            W, H = img.shape[1], img.shape[0]
            if r['mode'] == 'ped' and pbb is not None:
                x0, y0, x1, y1 = R7.pad_bbox(pbb, R7.FRAME_CFG.local_pad_ratio, W, H)
            else:
                uni = R7.union_bbox(pbb, vbb)
                if uni is not None:
                    x0, y0, x1, y1 = R7.pad_bbox(uni, R7.FRAME_CFG.local_pad_ratio, W, H)
                else:
                    x0, y0, x1, y1 = 0, 0, W, H
            loc = drawn[y0:y1, x0:x1]
            if loc.size == 0:
                loc = R7._center_crop(drawn)
            loc = R7._resize_keep_aspect(loc, R7.FRAME_CFG.local_max_side)
            cv2.imwrite(ldst, loc, [cv2.IMWRITE_JPEG_QUALITY, R7.FRAME_CFG.jpeg_quality])
            n_ok += 1
        else:
            import shutil
            src_g = f'{PROC}/frames_global/{r["rel"]}/{r["gname"]}'
            src_l = f'{PROC}/frames_local/{r["rel"]}/{r["gname"]}'
            if os.path.exists(src_g): shutil.copy2(src_g, gdst)
            if os.path.exists(src_l): shutil.copy2(src_l, ldst)
            n_fb += 1
    print(f'rebuild: {n_ok} restyled, {n_fb} fallback')
    d = json.load(open(f'{PROC}/vqa_train.json'))
    os.makedirs(f'{DEST}', exist_ok=True)
    n_map = n_tot = 0
    for s in d:
        imgs = []
        for p in s['images']:
            n_tot += 1
            q = p.replace('/processed/', '/processed_cosmos/')
            if 'overhead_view' in p and os.path.exists(q):
                imgs.append(q); n_map += 1
            else:
                imgs.append(p)
        s['images'] = imgs
    json.dump(d, open(f'{DEST}/vqa_train.json', 'w'))
    print(f'vqa_train: {n_map}/{n_tot} ảnh remap sang cosmos ({100*n_map/n_tot:.0f}%)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['dump', 'spec', 'rebuild'])
    a = ap.parse_args()
    {'dump': cmd_dump, 'spec': cmd_spec, 'rebuild': cmd_rebuild}[a.stage]()
