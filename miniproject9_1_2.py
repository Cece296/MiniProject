import cv2
import os

# -------- CONFIGURE THESE --------
video_path = "DJI_0199.MOV"
output_dir = "extractedframes"
start_frame = 1200  # number of frames to skip
every_nth = 25      # save every Nth frame
# ---------------------------------

# Make sure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Open video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error opening video file.")
    exit()

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
saved_count = 0
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Only consider frames after the initial skip
    if frame_idx >= start_frame and (frame_idx - start_frame) % every_nth == 0:
        saved_count += 1
        filename = f"frame_{saved_count:04d}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame)
        print(f"Saved {filename}")

    frame_idx += 1

cap.release()
print(f"Done! {saved_count} frames saved in '{output_dir}'.")
