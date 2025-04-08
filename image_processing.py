import cv2
import numpy as np
import os
import time
import tkinter as tk
from tkinter import filedialog
import pandas as pd

# Global variable to store the mouse click location (for contour selection)
click_point = None

def process_frame(frame):
    """
    Process the frame similarly to the original logic:
      - Convert to grayscale, blur, and apply adaptive thresholding (with blockSize=11 and C=15).
      - Use morphological closing with an elliptical kernel.
      - Detect contours and filter out those smaller than a minimum area.
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
    min_area = 30
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    return filtered_contours, processed

def get_contour_centroid(contour):
    """
    Compute the centroid (center point) of a contour.
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
    Mouse callback that saves the click location.
    """
    global click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        click_point = (x, y)
        print("User clicked at:", click_point)

def main():
    global click_point

    # Try to use Tkinter file dialog to select a video file.
    try:
        root = tk.Tk()
        root.withdraw()  # Hide the main window.
        video_file = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("MP4 files", "*.mp4"),
                ("AVI files", "*.avi"),
                ("MOV files", "*.mov"),
                ("MKV files", "*.mkv"),
                ("All Files", "*.*")
            ]
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

    # Read the first frame for contour selection.
    ret, first_frame = cap.read()
    if not ret:
        print("Error reading the first frame from the video.")
        cap.release()
        return

    display_frame = first_frame.copy()
    contours, _ = process_frame(first_frame)
    for i, cnt in enumerate(contours):
        cv2.drawContours(display_frame, [cnt], -1, (0, 255, 0), 2)
        centroid = get_contour_centroid(cnt)
        if centroid:
            cv2.putText(display_frame, str(i), centroid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Let the user click on the contour they want to track.
    cv2.namedWindow("Select Contour")
    cv2.setMouseCallback("Select Contour", mouse_callback)
    print("Please click on the contour you want to track in the displayed frame. Press ESC to exit if needed.")
    while click_point is None:
        cv2.imshow("Select Contour", display_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            print("No contour selected; exiting.")
            cv2.destroyWindow("Select Contour")
            cap.release()
            return
    cv2.destroyWindow("Select Contour")

    # Find the contour whose centroid is closest to the clicked point.
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

    # Setup the output video writer.
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_filename = os.path.splitext(video_file)[0] + "_processed.mp4"
    out = cv2.VideoWriter(out_filename, fourcc, fps, (frame_width, frame_height))

    # Prepare a list to record location data.
    location_data = []
    frame_count = 0

    # Process the video frame by frame.
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        contours, _ = process_frame(frame)
        # Gather valid centroids.
        valid_centroids = [get_contour_centroid(cnt) for cnt in contours if get_contour_centroid(cnt) is not None]

        # Update tracked_centroid: choose the one closest to the previous tracked centroid.
        if valid_centroids:
            distances = [np.linalg.norm(np.array(pt) - np.array(tracked_centroid)) for pt in valid_centroids]
            min_idx = np.argmin(distances)
            tracked_centroid = valid_centroids[min_idx]
        else:
            print(f"No valid contours detected in frame {frame_count}; using previous centroid.")

        if tracked_centroid is not None:
            cv2.circle(frame, tracked_centroid, 3, (0, 0, 255), -1)

        location_data.append({
            "frame": frame_count,
            "x": tracked_centroid[0] if tracked_centroid is not None else None,
            "y": tracked_centroid[1] if tracked_centroid is not None else None
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
    df = pd.DataFrame(location_data)
    df.to_csv(location_filename, index=False)

    print("Processed video saved as:", out_filename)
    print("Location data saved as:", location_filename)

if __name__ == "__main__":
    main()