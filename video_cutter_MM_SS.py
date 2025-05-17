import cv2
import os
import sys
import tkinter as tk
from tkinter import filedialog

def ask_for_time_range(duration_sec):
    """
    Prompt the user to enter start/end times in MM:SS format.
    Returns (start_sec, end_sec, start_label, end_label).
    """
    max_min = int(duration_sec // 60)
    max_sec = int(duration_sec % 60)
    prompt = f"(0 to {max_min}:{max_sec:02d})"
    while True:
        start_str = input(f"Enter start time MM:SS {prompt}: ")
        end_str   = input(f"Enter   end time MM:SS (must be > start) {prompt}: ")
        try:
            sm, ss = map(int, start_str.split(":"))
            em, es = map(int, end_str.split(":"))
        except ValueError:
            print("  ✗ Invalid format. Use MM:SS, e.g. 02:30")
            continue
        start_sec = sm*60 + ss
        end_sec   = em*60 + es
        if 0 <= start_sec < end_sec <= duration_sec:
            # we'll use these labels in the filename
            return start_sec, end_sec, start_str.replace(":", "_"), end_str.replace(":", "_")
        print("  ✗ Times out of range or start ≥ end. Try again.")

# 1) Select the video:
root = tk.Tk()
root.withdraw()
video_path = filedialog.askopenfilename(
    title="Select video file",
    filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
)
if not video_path:
    print("No file selected. Exiting.")
    sys.exit()

# 2) Open & inspect:
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: could not open video.")
    sys.exit()

fps          = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration_sec = total_frames / fps

# 3) Ask for time range (MM:SS):
start_sec, end_sec, start_lbl, end_lbl = ask_for_time_range(duration_sec)

# 4) Compute corresponding frames:
start_frame = int(start_sec * fps)
end_frame   = int(end_sec   * fps)

# 5) Prepare writer (same size & fps):
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

base, ext    = os.path.splitext(os.path.basename(video_path))
out_fname    = f"{base}_{start_lbl}s_to_{end_lbl}s{ext}"
out_path     = os.path.join(os.path.dirname(video_path), out_fname)
out_writer   = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

# 6) Cut and save:
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
for _ in range(start_frame, end_frame):
    ret, frame = cap.read()
    if not ret:
        break
    out_writer.write(frame)

cap.release()
out_writer.release()
print(f"✅ Saved trimmed clip as:\n   {out_path}")
