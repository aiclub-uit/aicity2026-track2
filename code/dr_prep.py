#!/usr/bin/env python
"""dr_prep.py — Photometric domain randomization cho frame train (Sim2Real).
Mô phỏng degradation camera thật (CCTV/dashcam): blur, nén JPEG, phơi sáng lệch,
noise, mất nét do downscale. KHÔNG hue (giữ GT màu), KHÔNG geometric (giữ GT vị trí).
Pre-gen 1 biến thể/frame vào frames_dr/ + build processed_dr/vqa_train.json
(mỗi ảnh: p=0.6 dùng biến thể DR, 0.4 giữ sạch — seed cố định theo path).
"""
import json, os, io, random, hashlib, sys
from multiprocessing import Pool
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np

SRC = '/workspace/AICC/code/output_qwen7b/processed'
DR_ROOT = '/workspace/AICC/code/output_qwen7b/frames_dr'
OUT_DIR = '/workspace/AICC/code/output_qwen35/processed_dr'
P_AUG = 0.6

def _rng(path):
    return random.Random(int(hashlib.md5(path.encode()).hexdigest()[:8], 16))

def augment(im, rng):
    ops = rng.sample(['blur', 'downup', 'jpeg', 'bright', 'contrast', 'noise'],
                     k=rng.randint(1, 3))
    for op in ops:
        if op == 'blur':
            im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.6, 2.0)))
        elif op == 'downup':
            w, h = im.size; s = rng.uniform(0.35, 0.7)
            im = im.resize((max(8,int(w*s)), max(8,int(h*s))), Image.BICUBIC).resize((w, h), Image.BICUBIC)
        elif op == 'jpeg':
            buf = io.BytesIO(); im.save(buf, 'JPEG', quality=rng.randint(25, 60))
            buf.seek(0); im = Image.open(buf).convert('RGB')
        elif op == 'bright':
            im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.6, 1.4))
        elif op == 'contrast':
            im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.65, 1.35))
        elif op == 'noise':
            a = np.asarray(im).astype(np.float32)
            a += np.random.RandomState(rng.randint(0, 2**31)).normal(0, rng.uniform(3, 10), a.shape)
            im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return im

def dr_path(p):
    rel = os.path.relpath(p, SRC)
    return os.path.join(DR_ROOT, rel)

def work(p):
    out = dr_path(p)
    if os.path.exists(out):
        return 0
    os.makedirs(os.path.dirname(out), exist_ok=True)
    im = Image.open(p).convert('RGB')
    augment(im, _rng(p)).save(out, 'JPEG', quality=92)
    return 1

def main():
    d = json.load(open(f'{SRC}/vqa_train.json'))
    paths = sorted({p for s in d for p in s['images']})
    print(f'{len(d)} samples, {len(paths)} unique frames')
    with Pool(16) as pool:
        made = sum(pool.map(work, paths, chunksize=32))
    print(f'augmented: {made} frames -> {DR_ROOT}')
    os.makedirs(OUT_DIR, exist_ok=True)
    n_aug = n_tot = 0
    for s in d:
        imgs = []
        for p in s['images']:
            n_tot += 1
            if _rng('choice:' + s['id'] + p).random() < P_AUG:
                imgs.append(dr_path(p)); n_aug += 1
            else:
                imgs.append(p)
        s['images'] = imgs
    json.dump(d, open(f'{OUT_DIR}/vqa_train.json', 'w'))
    print(f'train json: {n_aug}/{n_tot} ảnh dùng DR ({100*n_aug/n_tot:.0f}%) -> {OUT_DIR}/vqa_train.json')

if __name__ == '__main__':
    main()
