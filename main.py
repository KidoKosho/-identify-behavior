#!/usr/bin/env python3
import os
import sys
import argparse
from urllib.parse import urlparse
import cv2
import time
from models import ModelLoader
from pipeline import Pipeline


def build_parser():
    parser = argparse.ArgumentParser(description="Run YOLO pipeline on video")
    parser.add_argument("--invideo", "--video", dest="video", type=str, required=True,
                        help="Path to input video or HLS stream URL")
    parser.add_argument("--outimg", "--output", dest="output", type=str, default="./imgout",
                        help="Output directory")
    parser.add_argument("--display", action="store_true", default=True,
                        help="Display the processed video in an OpenCV window")
    parser.add_argument("--window-scale", dest="window_scale", type=float, default=1.0,
                        help="Scale factor for display window (0.5 = 50% smaller, 0.25 = 75% smaller)")
    return parser


def resolve_video_name(video_path):
    parsed = urlparse(video_path)
    source = parsed.path if parsed.scheme else video_path
    base_name = os.path.basename(source)
    stem = os.path.splitext(base_name)[0]
    return stem or "video"


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    video_path = args.video
    if not video_path:
        print("❌ No input video provided")
        sys.exit(1)

    if not video_path.startswith("http") and not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        sys.exit(1)

    print(f"📹 Video: {video_path}")
    print(f"📁 Output dir: {args.output}")

    os.makedirs(args.output, exist_ok=True)
    models = ModelLoader()
    pipeline = Pipeline(models, video_name=resolve_video_name(video_path), output_dir=args.output)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        sys.exit(1)

    frame_count = 0
    start_time = time.time()
    print("🚀 Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        processed = pipeline.process_frame(frame)

        if args.display:
            if args.window_scale < 1.0:
                h, w = processed.shape[:2]
                new_w = int(w * args.window_scale)
                new_h = int(h * args.window_scale)
                display_frame = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                display_frame = processed
            
            cv2.imshow("Security Detection", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"🔄 Frame {frame_count}, FPS: {fps:.2f}")

    cap.release()
    cv2.destroyAllWindows()
    export_path = pipeline.export_tracking_data()
    if export_path:
        print(f"📊 Tracking export saved to: {export_path}")
    print(f"✅ Done! Processed {frame_count} frames.")
    print(f"📁 Event images saved to: {args.output}")


if __name__ == "__main__":
    main()