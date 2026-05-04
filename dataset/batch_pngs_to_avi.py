import argparse
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Batch convert IRDST PNG sequences into AVI files.")
    parser.add_argument(
        "--input_root",
        type=str,
        default="dataset/IRDST/images",
        help="Root directory containing sequence subfolders [default: dataset/IRDST/images]."
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="dataset/avi/IRDST",
        help="Output directory for AVI files [default: dataset/avi/IRDST]."
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Output video FPS [default: 30]."
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Write AVI files in grayscale mode."
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="MJPG",
        help="AVI codec [default: MJPG]."
    )
    return parser.parse_args()


def natural_key(text):
    return [int(part) if part.isdigit() else part.lower() for part in __import__("re").split(r"(\d+)", text)]


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    converter = os.path.join(script_dir, "pngs_to_avi.py")

    input_root = os.path.join(script_dir, args.input_root) if not os.path.isabs(args.input_root) else args.input_root
    output_root = os.path.join(script_dir, args.output_root) if not os.path.isabs(args.output_root) else args.output_root

    if not os.path.isdir(input_root):
        raise FileNotFoundError(f"Input root not found: {input_root}")

    os.makedirs(output_root, exist_ok=True)

    sequences = [
        name for name in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, name))
    ]
    sequences.sort(key=natural_key)

    if not sequences:
        raise FileNotFoundError(f"No sequence folders found in: {input_root}")

    for idx, seq_name in enumerate(sequences, start=1):
        input_dir = os.path.join(input_root, seq_name)
        output_path = os.path.join(output_root, f"{seq_name}.avi")

        cmd = [
            sys.executable,
            converter,
            "--input_dir", input_dir,
            "--output_path", output_path,
            "--fps", str(args.fps),
            "--codec", args.codec,
        ]
        if args.grayscale:
            cmd.append("--grayscale")

        print(f"[{idx}/{len(sequences)}] Converting {seq_name} -> {output_path}")
        subprocess.run(cmd, check=True)

    print(f"Finished converting {len(sequences)} sequences to: {output_root}")


if __name__ == "__main__":
    main()
