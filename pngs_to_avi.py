import argparse
import os
import re

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Convert an ordered PNG sequence into an AVI video.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing PNG frames."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output AVI file path."
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Output video FPS [default: 30]."
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.png",
        help="Filename pattern hint, currently used for help text only [default: *.png]."
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Force output video to grayscale."
    )
    parser.add_argument(
        "--resize_width",
        type=int,
        default=0,
        help="Optional resize width. 0 keeps original width."
    )
    parser.add_argument(
        "--resize_height",
        type=int,
        default=0,
        help="Optional resize height. 0 keeps original height."
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="MJPG",
        help="FourCC codec for AVI [default: MJPG]."
    )
    return parser.parse_args()


def natural_key(text):
    parts = re.split(r"(\d+)", text)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def list_png_files(input_dir):
    files = [
        name for name in os.listdir(input_dir)
        if name.lower().endswith(".png") and os.path.isfile(os.path.join(input_dir, name))
    ]
    files.sort(key=natural_key)
    return files


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_frame(frame_path, grayscale):
    if grayscale:
        frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    else:
        frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Failed to read frame: {frame_path}")
    return frame


def maybe_resize(frame, resize_width, resize_height):
    if resize_width > 0 and resize_height > 0:
        return cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
    return frame


def main():
    args = parse_args()

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    png_files = list_png_files(args.input_dir)
    if not png_files:
        raise FileNotFoundError(f"No PNG files found in: {args.input_dir}")

    first_frame_path = os.path.join(args.input_dir, png_files[0])
    first_frame = load_frame(first_frame_path, args.grayscale)
    first_frame = maybe_resize(first_frame, args.resize_width, args.resize_height)

    if args.grayscale:
        height, width = first_frame.shape
        is_color = False
    else:
        height, width = first_frame.shape[:2]
        is_color = True

    ensure_parent_dir(args.output_path)
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = cv2.VideoWriter(args.output_path, fourcc, args.fps, (width, height), isColor=is_color)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for: {args.output_path}")

    try:
        for idx, file_name in enumerate(png_files, start=1):
            frame_path = os.path.join(args.input_dir, file_name)
            frame = load_frame(frame_path, args.grayscale)
            frame = maybe_resize(frame, args.resize_width, args.resize_height)

            if args.grayscale:
                if frame.shape != (height, width):
                    raise ValueError(f"Frame size mismatch: {frame_path} -> {frame.shape}, expected {(height, width)}")
            else:
                if frame.shape[:2] != (height, width):
                    raise ValueError(
                        f"Frame size mismatch: {frame_path} -> {frame.shape[:2]}, expected {(height, width)}"
                    )

            writer.write(frame)

            if idx == 1 or idx == len(png_files) or idx % 50 == 0:
                print(f"Written {idx}/{len(png_files)} frames")
    finally:
        writer.release()

    print(f"Input directory: {args.input_dir}")
    print(f"Output video: {args.output_path}")
    print(f"Frames: {len(png_files)}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {args.fps}")
    print(f"Mode: {'grayscale' if args.grayscale else 'color'}")


if __name__ == "__main__":
    main()
