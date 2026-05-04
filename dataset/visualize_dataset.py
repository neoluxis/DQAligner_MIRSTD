#!/usr/bin/env python3
"""
Visualize IRDST-style dataset sequences: render image sequence to video with boxes.

Usage:
    python visualize_dataset.py --root dataset/IRDST --id 27 --out out.mp4 --fps 10

The script looks for images in `root/images/<id>/` and optional `gt.txt` or `IR_label.json`.
"""
import argparse
import json
import os
from pathlib import Path
import cv2
from PIL import Image
from tqdm import tqdm


def parse_gt(gt_path: Path):
    if not gt_path.exists():
        return []
    out = []
    with gt_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 4:
                out.append(None)
                continue
            try:
                nums = [float(p) for p in parts[:4]]
            except Exception:
                out.append(None)
                continue
            # CST-AntiUAV: x,y are top-left coordinates (x,y,w,h)
            if nums[0] == 0 and nums[1] == 0 and nums[2] == 0 and nums[3] == 0:
                out.append(None)
            else:
                x, y, w, h = nums
                x1 = int(x)
                y1 = int(y)
                x2 = int(x + w)
                y2 = int(y + h)
                out.append((x1, y1, x2, y2))
    return out


def parse_exist(json_path: Path):
    if not json_path.exists():
        return []
    try:
        j = json.loads(json_path.read_text())
        return j.get('exist', [])
    except Exception:
        return []


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, help='IRDST dataset root')
    p.add_argument('--id', required=True, help='Sequence id (folder name)')
    p.add_argument('--out', required=True, help='Output video path')
    p.add_argument('--fps', type=int, default=10)
    args = p.parse_args()

    root = Path(args.root)
    seq = root / 'images' / str(args.id)
    if not seq.exists():
        print('Sequence not found:', seq)
        return

    imgs = sorted([p for p in seq.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    if not imgs:
        print('No images in', seq)
        return

    gt = parse_gt(seq / 'gt.txt')
    exist = parse_exist(seq / 'IR_label.json')

    # prepare video writer
    first = Image.open(imgs[0])
    w, h = first.size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(args.out, fourcc, args.fps, (w, h))

    for i, p in enumerate(tqdm(imgs, desc=f'seq {args.id}')):
        im = cv2.imread(str(p))
        if im is None:
            continue
        # draw bbox if available
        bbox = None
        if i < len(gt) and gt[i] is not None:
            bbox = gt[i]
        elif i < len(exist) and exist[i]:
            bbox = None
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'id={args.id} file={p.name}'
        cv2.putText(im, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        vw.write(im)

    vw.release()
    print('Saved video to', args.out)


if __name__ == '__main__':
    main()
