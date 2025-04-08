import cv2
import numpy as np
import os
import time
import tkinter as tk
from tkinter import filedialog
import pandas as pd

# Global variable for mouse click selection.
click_point = None

# Parameters for filtering and smoothing the tracked position.
MIN_CONTOUR_AREA = 18         # Normal mode: keep contours larger than this.
MAX_JUMP_DISTANCE = 13         # Allowed jump (in pixels) for moderate smoothing.
PREDICTION_THRESHOLD = 17      # If candidate jump > this, candidate is suspect.
SMOOTHING_ALPHA = 0.001        # Blending factor when jump is moderate.
LOST_COUNTER_MAX = 5           # Number of consecutive lost frames allowed before update.

# Base fallback detection parameter.
FALLBACK_AREA_FACTOR = 0.3     # Base factor for fallback (effective threshold = MIN_CONTOUR_AREA * factor).

def process_frame(frame, min_area=MIN_CONTOUR_AREA):
    """
    Convert frame to grayscale, blur it, then apply adaptive thresholding
    and morphological closing to get contours.
    Only keep contours with area larger than min_area.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        11, 15
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    return filtered_contours, processed

def get_contour_centroid(contour):
    """
    Compute the centroid (cx, cy) of a contour.
    Returns (cx, cy) or None if the area is zero.
    """
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)

def mouse_callback(event, x, y, flags, param):
    """
    Record the user's click location.
    """
    global click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        click_point = (x, y)
        print("User clicked at:", click_point)

def main():
    global click_point

    try:
        root = tk.Tk()
        root.withdraw()
        video_file = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("MP4 files", "*.mp4"), 
                       ("AVI files", "*.avi"), 
                       ("MOV files", "*.mov"),
                       ("MKV files", "*.mkv"), 
                       ("All Files", "*.*")]
        )
    except Exception as e:
        print("Error using file dialog:", e)
        video_file = ""

    if not video_file:
        video_file = input("File dialog did not return a file. Please enter the full path to your video file: ")

    if not video_file or not os.path.exists(video_file):
        print("No valid video file selected. Exiting.")
        return

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    # Get frame dimensions for clamping.
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Read and process the first frame for contour selection.
    ret, first_frame = cap.read()
    if not ret:
        print("Error reading the first frame.")
        cap.release()
        return

    display_frame = first_frame.copy()
    contours, _ = process_frame(first_frame)
    for i, cnt in enumerate(contours):
        cv2.drawContours(display_frame, [cnt], -1, (0, 255, 0), 2)
        centroid = get_contour_centroid(cnt)
        if centroid:
            cv2.putText(display_frame, str(i), centroid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.namedWindow("Select Contour")
    cv2.setMouseCallback("Select Contour", mouse_callback)
    print("Please click on the contour you want to track. Press ESC to exit if needed.")
    while click_point is None:
        cv2.imshow("Select Contour", display_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            print("No contour selected; exiting.")
            cv2.destroyWindow("Select Contour")
            cap.release()
            return
    cv2.destroyWindow("Select Contour")

    # Choose the contour whose centroid is closest to the clicked point.
    selected_contour = None
    min_distance = float('inf')
    for cnt in contours:
        centroid = get_contour_centroid(cnt)
        if centroid is not None:
            distance = np.linalg.norm(np.array(centroid) - np.array(click_point))
            if distance < min_distance:
                min_distance = distance
                selected_contour = cnt

    if selected_contour is None:
        print("No valid contour found near the selected point.")
        cap.release()
        return

    tracked_centroid = get_contour_centroid(selected_contour)
    print("Selected contour centroid:", tracked_centroid)

    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_filename = os.path.splitext(video_file)[0] + "_processed.mp4"
    out = cv2.VideoWriter(out_filename, fourcc, fps, (frame_width, frame_height))

    location_data = []
    frame_count = 0
    lost_counter = 0  # Count consecutive frames where candidate is too far.

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process normally.
        contours, _ = process_frame(frame)
        valid_centroids = [get_contour_centroid(cnt) for cnt in contours if get_contour_centroid(cnt) is not None]

        # If no valid candidate from normal settings, try fallback with an adaptive threshold.
        if not valid_centroids:
            # Adaptively lower the min area threshold based on lost_counter.
            # The more frames lost, the lower the threshold (but not lower than 10% of MIN_CONTOUR_AREA).
            adaptive_factor = max(0.1, FALLBACK_AREA_FACTOR - lost_counter * 0.05)
            fallback_min_area = MIN_CONTOUR_AREA * adaptive_factor
            contours_fallback, _ = process_frame(frame, min_area=fallback_min_area)
            valid_centroids = [get_contour_centroid(cnt) for cnt in contours_fallback if get_contour_centroid(cnt) is not None]
            if valid_centroids:
                print(f"Frame {frame_count}: Fallback (adaptive factor {adaptive_factor:.2f}) found {len(valid_centroids)} candidate(s).")

        if valid_centroids:
            # Choose the candidate closest to current tracked position.
            distances = [np.linalg.norm(np.array(pt) - np.array(tracked_centroid)) for pt in valid_centroids]
            min_idx = np.argmin(distances)
            candidate = valid_centroids[min_idx]
            jump_distance = np.linalg.norm(np.array(candidate) - np.array(tracked_centroid))

            if jump_distance > PREDICTION_THRESHOLD:
                lost_counter += 1
                print(f"Frame {frame_count}: Candidate jump {jump_distance:.1f} exceeds threshold. Lost counter: {lost_counter}")
                if lost_counter >= LOST_COUNTER_MAX:
                    tracked_centroid = candidate
                    lost_counter = 0
                    print(f"Frame {frame_count}: Updating tracked position after {LOST_COUNTER_MAX} lost frames: {tracked_centroid}")
            else:
                lost_counter = 0  # Reset counter if candidate is acceptable.
                if jump_distance > MAX_JUMP_DISTANCE:
                    tracked_centroid = tuple(np.array(tracked_centroid) + SMOOTHING_ALPHA * (np.array(candidate) - np.array(tracked_centroid)))
                else:
                    tracked_centroid = candidate
        else:
            print(f"Frame {frame_count}: No valid contours detected; keeping previous position.")

        # Draw the tracked centroid.
        if tracked_centroid is not None:
            cv2.circle(frame, (int(tracked_centroid[0]), int(tracked_centroid[1])), 3, (0, 0, 255), -1)

        location_data.append({
            "frame": frame_count,
            "x": tracked_centroid[0],
            "y": tracked_centroid[1]
        })
        out.write(frame)
        frame_count += 1

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Tracking terminated by user.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    location_filename = os.path.splitext(video_file)[0] + "_location.csv"
    pd.DataFrame(location_data).to_csv(location_filename, index=False)
    print("Processed video saved as:", out_filename)
    print("Location data saved as:", location_filename)

if __name__ == "__main__":
    main()
