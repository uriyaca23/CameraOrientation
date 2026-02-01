import cv2
import os

input_path = r'c:/Users/uriya/PycharmProjects/CameraOrientation/assets/video.mp4'
output_path = r'c:/Users/uriya/PycharmProjects/CameraOrientation/assets/video_fixed.mp4'

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("Error opening video file")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Converting HEVC to H.264...")
print(f"Res: {w}x{h}, FPS: {fps}, Frames: {total_frames}")

# Try H.264 (avc1) first
fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

if not out.isOpened():
    print("avc1 codec not found, trying mp4v...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

if not out.isOpened():
    print("Failed to open video writer.")
    exit(1)

count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
    count += 1
    if count % 100 == 0:
        print(f"Processed {count}/{total_frames} frames ({count/total_frames*100:.1f}%)")

cap.release()
out.release()
print("Conversion Complete.")
