#!/usr/bin/env python3
"""
Convert CST-AntiUAV dataset to IRDST-style layout.

Creates sequence folders under `dst/images/<id>/` and optional masks under `dst/masks/<id>/`.
Also appends the new sequence ids to `dst/ImageSets/train_new.txt` / `val_new.txt` based on split.

Usage:
    python convert_cst_to_irdst.py --src dataset/CST_AntiUAV/CST-AntiUAV --dst dataset/IRDST --split train

"""
import argparse
import json
import os
import shutil
from pathlib import Path
from PIL import Image, ImageDraw
from tqdm import tqdm

def parse_gt(gt_path):
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
            # CST-AntiUAV stores x,y as top-left corner (not center): x,y,w,h
            if nums[0] == 0 and nums[1] == 0 and nums[2] == 0 and nums[3] == 0:
                out.append(None)
            else:
                x, y, w, h = nums
                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h
                out.append((x1, y1, x2, y2))
    return out


def parse_exist(json_path):
    if not json_path.exists():
        return None
    try:
        j = json.loads(json_path.read_text())
        if 'exist' in j:
            return j['exist']
    except Exception:
        return None
    return None


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def next_ids(dst_images: Path, count: int):
    existing = []
    if dst_images.exists():
        for d in dst_images.iterdir():
            if d.is_dir() and d.name.isdigit():
                existing.append(int(d.name))
    start = 1
    if existing:
        start = max(existing) + 1
    return list(range(start, start + count))


def copy_sequence(src_seq: Path, dst_images: Path, dst_masks: Path = None, make_masks: bool = True, seq_id: str = ''):
    # src_seq contains image files and gt.txt, IR_label.json
    frames = sorted([p for p in src_seq.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    if not frames:
        return False

    gt = parse_gt(src_seq / 'gt.txt')
    exist = parse_exist(src_seq / 'IR_label.json')

    # create destination images/masks folder
    ensure_dir(dst_images)
    if dst_masks and make_masks:
        ensure_dir(dst_masks)

    for i, src_img in enumerate(tqdm(frames, desc=f'{src_seq.name} -> {seq_id}', leave=False)):
        dst_img = dst_images / src_img.name
        try:
            # use copy to keep files portable
            shutil.copy2(src_img, dst_img)
        except Exception:
            # fallback to simple open/save
            im = Image.open(src_img)
            im.save(dst_img)

        if dst_masks and make_masks:
            mask = Image.new('L', Image.open(src_img).size, 0)
            draw = ImageDraw.Draw(mask)
            bbox = None
            if i < len(gt) and gt[i] is not None:
                bbox = gt[i]
            elif exist and i < len(exist) and exist[i] and (i < len(gt) and gt[i] is None):
                # exist but no gt bbox: skip
                bbox = None
            if bbox:
                x1, y1, x2, y2 = bbox
                draw.rectangle([x1, y1, x2, y2], fill=255)
            mask_path = dst_masks / (src_img.stem + '.png')
            mask.save(mask_path)

    # copy auxiliary files too
    for fname in ('IR_label.json', 'gt.txt', 'exist.txt'):
        fsrc = src_seq / fname
        if fsrc.exists():
            try:
                shutil.copy2(fsrc, dst_images / fname)
            except Exception:
                pass

    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True, help='CST-AntiUAV root (contains train/val/test)')
    p.add_argument('--dst', required=True, help='IRDST dataset root')
    p.add_argument('--split', choices=('train', 'val', 'test'), default='train')
    p.add_argument('--make-masks', action='store_true', help='Generate simple rectangle masks from gt.txt')
    p.add_argument('--copy', action='store_true', help='Copy images (default). Use --symlink to create symlinks.')
    p.add_argument('--symlink', action='store_true', help='Create symlinks instead of copying')
    args = p.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    src_split = src_root / args.split
    if not src_split.exists():
        print('Source split not found:', src_split)
        return

    dst_images_root = dst_root / 'images'
    dst_masks_root = dst_root / 'masks'
    ensure_dir(dst_images_root)
    ensure_dir(dst_masks_root)

    seq_dirs = sorted([d for d in src_split.iterdir() if d.is_dir()])
    if not seq_dirs:
        print('No sequences found in', src_split)
        return

    ids = next_ids(dst_images_root, len(seq_dirs))

    # create or update ImageSets file
    imageset_file = dst_root / 'ImageSets' / (args.split + '_new.txt')
    ensure_dir(imageset_file.parent)

    with imageset_file.open('a') as isf:
        for seq_dir, seq_id in tqdm(zip(seq_dirs, ids), total=len(seq_dirs), desc=f'{args.split} sequences'):
            dst_images = dst_images_root / str(seq_id)
            dst_masks = dst_masks_root / str(seq_id)
            ensure_dir(dst_images)
            if args.symlink:
                # create symlinks named as original filenames
                for f in sorted(seq_dir.iterdir()):
                    if f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                        target = dst_images / f.name
                        if not target.exists():
                            os.symlink(os.path.abspath(f), str(target))
            else:
                # do copy by reusing copy_sequence logic
                copy_sequence(
                    seq_dir,
                    dst_images,
                    dst_masks if args.make_masks else None,
                    make_masks=args.make_masks,
                    seq_id=str(seq_id),
                )

            isf.write(str(seq_id) + '\n')

    print('Converted', len(seq_dirs), 'sequences to', dst_root)


if __name__ == '__main__':
    main()
