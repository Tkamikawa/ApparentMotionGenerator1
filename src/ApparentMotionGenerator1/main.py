import cv2
import numpy as np
import argparse

def emboss(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[-2, -1, 0],
                       [-1,  1, 1],
                       [-1,  1, 2]])
    embossed = cv2.filter2D(gray, -1, kernel, delta=128)
    return cv2.cvtColor(embossed, cv2.COLOR_GRAY2BGR)

def invert(img: np.ndarray) -> np.ndarray:
    return cv2.bitwise_not(img)

def write_mp4(frames, output_path="illusion.mp4", fps=12):
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()
    print(f"Saved MP4 to {output_path}")

def main(img_path: str, output_path: str, nframes: int, fps: int):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    embossed = emboss(img)
    invimg = invert(img)

    seq = [embossed, invimg, img]
    frames = [seq[i % 3] for i in range(nframes)]

    write_mp4(frames, output_path, fps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Illusion video generator")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("-o", "--output", default="illusion.mp4", help="Output MP4 path (default: illusion.mp4)")
    parser.add_argument("-n", "--nframes", type=int, default=80, help="Number of frames (default: 80)")
    parser.add_argument("--fps", type=int, default=12, help="Frames per second (default: 12)")
    args = parser.parse_args()

    main(args.input, args.output, args.nframes, args.fps)
    