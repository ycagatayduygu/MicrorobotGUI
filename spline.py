from pixelinkWrapper import *
from ctypes import *
import numpy as np
import cv2
import os
import time
import socket
import struct
import math
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
import pandas as pd

# Camera constants
MAX_WIDTH = 5000
MAX_HEIGHT = 5000
MAX_BYTES_PER_PIXEL = 3
RESIZED_WIDTH = 512
RESIZED_HEIGHT = 512

# Control parameters
K = 2.508  # Constant for speed calculation
horizon = 5  # MPC prediction horizon
optimization_interval = 1  # Optimization interval in seconds

# Target locations for the spline
target_points = np.array([
    [50, 100],
    [100, 250],
    [200, 100],
    [400, 300]
])

# Create a cubic spline from the target points
t = np.arange(target_points.shape[0])
cs_x = CubicSpline(t, target_points[:, 0])
cs_y = CubicSpline(t, target_points[:, 1])
fine_t = np.linspace(0, target_points.shape[0] - 1, 200)
spline_x = cs_x(fine_t)
spline_y = cs_y(fine_t)

# Initialize control state
last_optimization_time = time.time()
last_Wm, last_theta_dot = 0.0, 0.0
control_data = []  # Array to log control parameters
previous_centroid = None  # Initialize previous centroid

# -------------------------------------------------------------------
# Helper function: Determine pixel type (mono or color)
def getPixelType(hCamera):
    pixelType = PT_OTHERWISE
    savedPixelFormat = 0
    newPixelFormat = 0
    ret = PxLApi.getFeature(hCamera, PxLApi.FeatureId.PIXEL_FORMAT)
    if not PxLApi.apiSuccess(ret[0]):
        return pixelType
    params = ret[2]
    savedPixelFormat = int(params[0])
    # Try setting mono pixel format
    newPixelFormat = PxLApi.PixelFormat.MONO8
    params = [newPixelFormat,]
    ret = PxLApi.setFeature(hCamera, PxLApi.FeatureId.PIXEL_FORMAT, PxLApi.FeatureFlags.MANUAL, params)
    if PxLApi.apiSuccess(ret[0]):
        pixelType = PT_MONO
    else:
        # Try setting color pixel format
        newPixelFormat = PxLApi.PixelFormat.BAYER8
        params = [newPixelFormat,]
        ret = PxLApi.setFeature(hCamera, PxLApi.FeatureId.PIXEL_FORMAT, PxLApi.FeatureFlags.MANUAL, params)
        if PxLApi.apiSuccess(ret[0]):
            pixelType = PT_COLOR
    # Restore the saved pixel format
    params = [savedPixelFormat,]
    PxLApi.setFeature(hCamera, PxLApi.FeatureId.PIXEL_FORMAT, PxLApi.FeatureFlags.MANUAL, params)
    return pixelType

# -------------------------------------------------------------------
# Cost breakdown helper for the spline-following phase
def compute_cost_breakdown(control_sequence, x, y, theta_deg, spline_x, spline_y, dt, horizon, last_Wm, last_theta_dot, lookahead_offset, lookahead_weight):
    breakdown = {
        "total": 0.0,
        "lookahead_cost": 0.0,
        "control_bias_cost": 0.0,
        "first_penalty": 0.0,
        "subsequent_penalty": 0.0,
        "distance_penalty": 0.0
    }
    cost = 0.0
    theta = theta_deg  # in degrees
    for i in range(horizon):
        Wm, theta_dot = control_sequence[i*2], control_sequence[i*2+1]
        Wm = np.clip(Wm, 0.1, 0.5)
        V = K * Wm
        theta += np.degrees(theta_dot * dt)
        theta = theta % 360
        theta_transformed = transform_to_surface_coordinates(theta)
        theta_transformed_rad = np.radians(theta_transformed)
        x += V * np.sin(theta_transformed_rad) * dt
        y += -V * np.cos(theta_transformed_rad) * dt
        distances = np.sqrt((spline_x - x)**2 + (spline_y - y)**2)
        closest_idx = np.argmin(distances)
        lookahead_idx = closest_idx + lookahead_offset
        if lookahead_idx >= len(spline_x):
            lookahead_idx = len(spline_x) - 1
        lookahead_error = np.sqrt((spline_x[lookahead_idx] - x)**2 + (spline_y[lookahead_idx] - y)**2)
        la_cost = lookahead_weight * lookahead_error
        cost += la_cost
        breakdown["lookahead_cost"] += la_cost

        bias_cost = 1.1 * (1 - Wm)**2
        cost += bias_cost
        breakdown["control_bias_cost"] += bias_cost

        if i == 0:
            pen = 1.0 * (Wm - last_Wm)**2 + 100000 * (theta_dot - last_theta_dot)**2
            cost += pen
            breakdown["first_penalty"] += pen
        else:
            prev_Wm = control_sequence[(i-1)*2]
            prev_theta_dot = control_sequence[(i-1)*2+1]
            pen = 0.1 * (Wm - prev_Wm)**2 + 5 * (theta_dot - prev_theta_dot)**2
            cost += pen
            breakdown["subsequent_penalty"] += pen
            dpen = 10 * distances[closest_idx]
            cost += dpen
            breakdown["distance_penalty"] += dpen

    breakdown["total"] = cost
    return breakdown

# (1) For spline-following phase (using the full spline)
def calculate_control_parameters(current_position, spline_x, spline_y, rot_azimuth, last_Wm, last_theta_dot, lookahead_offset=2, lookahead_weight=4.0):
    def mpc_cost_function(control_sequence, x, y, theta_deg, spline_x, spline_y, dt, horizon, last_Wm, last_theta_dot, lookahead_offset, lookahead_weight):
        cost = 0.0
        theta = theta_deg  # in degrees
        for i in range(horizon):
            Wm, theta_dot = control_sequence[i*2], control_sequence[i*2+1]
            Wm = np.clip(Wm, 0.1, 0.5)
            V = K * Wm
            theta += np.degrees(theta_dot * dt)
            theta = theta % 360
            theta_transformed = transform_to_surface_coordinates(theta)
            theta_transformed_rad = np.radians(theta_transformed)
            x += V * np.sin(theta_transformed_rad) * dt
            y += -V * np.cos(theta_transformed_rad) * dt
            distances = np.sqrt((spline_x - x)**2 + (spline_y - y)**2)
            closest_idx = np.argmin(distances)
            lookahead_idx = closest_idx + lookahead_offset
            if lookahead_idx >= len(spline_x):
                lookahead_idx = len(spline_x) - 1
            lookahead_error = np.sqrt((spline_x[lookahead_idx]-x)**2 + (spline_y[lookahead_idx]-y)**2)
            cost += lookahead_weight * lookahead_error
            cost += 1.1 * (1 - Wm)**2
            if i==0:
                cost += 1.0 * (Wm - last_Wm)**2 + 0.5 * (theta_dot - last_theta_dot)**2
            else:
                prev_Wm = control_sequence[(i-1)*2]
                prev_theta_dot = control_sequence[(i-1)*2+1]
                cost += 0.1 * (Wm - prev_Wm)**2 + 5 * (theta_dot - prev_theta_dot)**2
                cost += 1 * distances[closest_idx]  # Penalize distance to the closest point on the spline
        return cost

    x, y = current_position
    dt = optimization_interval
    initial_guess = [last_Wm, 0.0] * horizon
    result = minimize(mpc_cost_function, initial_guess,
                      args=(x, y, rot_azimuth, spline_x, spline_y, dt, horizon, last_Wm, last_theta_dot, lookahead_offset, lookahead_weight),
                      bounds=[(0.1, 1), (-np.pi, np.pi)] * horizon,
                      method='SLSQP')
    Wm_opt, theta_dot_opt = result.x[0], result.x[1]
    distances = np.sqrt((spline_x - x)**2 + (spline_y - y)**2)
    min_dist = np.min(distances)
    # Compute the cost breakdown on the final optimized control sequence:
    cost_breakdown = compute_cost_breakdown(result.x, current_position[0], current_position[1], rot_azimuth, spline_x, spline_y, dt, horizon, last_Wm, last_theta_dot, lookahead_offset, lookahead_weight)
    return Wm_opt, theta_dot_opt, min_dist, cost_breakdown

# (2) For go-to-start phase: use a single target (the start point)
def calculate_control_parameters_single(current_position, target_position, rot_azimuth, last_Wm, last_theta_dot):
    def mpc_cost_function(control_sequence, x, y, theta, target_x, target_y, dt, horizon, last_Wm, last_theta_dot):
        cost = 0.0
        for i in range(horizon):
            Wm, theta_dot = control_sequence[i*2], control_sequence[i*2+1]
            Wm = np.clip(Wm, 0.1, 0.5)
            V = K * Wm
            theta += np.degrees(theta_dot * dt)
            theta = theta % 360
            theta_transformed = transform_to_surface_coordinates(theta)
            theta_transformed_rad = np.radians(theta_transformed)
            x += V * np.sin(theta_transformed_rad) * dt
            y += -V * np.cos(theta_transformed_rad) * dt
            path_dist = (x - target_x)**2 + (y - target_y)**2
            cost += path_dist
        return cost
    x, y = current_position
    target_x, target_y = target_position
    dt = optimization_interval
    initial_guess = [last_Wm, 0.0] * horizon
    result = minimize(mpc_cost_function, initial_guess,
                      args=(x, y, rot_azimuth, target_x, target_y, dt, horizon, last_Wm, last_theta_dot),
                      bounds=[(0.1, 0.5), (-np.pi, np.pi)] * horizon,
                      method='SLSQP')
    Wm_opt, theta_dot_opt = result.x[0], result.x[1]
    error = np.sqrt((x - target_x)**2 + (y - target_y)**2)
    # For the single-target phase you can also add a cost breakdown if needed.
    # (Here, we simply log the final error.)
    cost_breakdown = {"final_error": error}
    return Wm_opt, theta_dot_opt, error, cost_breakdown

def transform_to_surface_coordinates(rot_azimuth):
    if 0 <= rot_azimuth <= 270:
        transformed_angle = 90 - rot_azimuth
    else:
        transformed_angle = 360 - (rot_azimuth - 90)
    return transformed_angle

def send_magnetic_field_data(azimuth, inclination, magnitude, rot_inclination, freq, rot_azimuth):
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = ('127.0.0.1', 12345)
        client_socket.connect(server_address)
        message = struct.pack('dddddd', azimuth, inclination, magnitude, rot_inclination, rot_azimuth, freq)
        client_socket.sendall(message)
        print(f"Sent: Azimuth={azimuth}, Inclination={inclination}, Rot_Inclination={rot_inclination}, Rot_Azimuth={rot_azimuth}")
        client_socket.close()
    except Exception as e:
        print(f"Error while sending magnetic field data: {e}")

def process_frame(frame):
    """
    Processes the frame to detect and find contours of the object to be tracked.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=11, C=15
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    processed_binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(processed_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 30
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    return filtered_contours, processed_binary

def get_contour_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"]/M["m00"])
    cy = int(M["m01"]/M["m00"])
    return (cx, cy)

def get_new_target(previous_target, min_distance=50, max_x=RESIZED_WIDTH, max_y=RESIZED_HEIGHT):
    while True:
        new_target_x = np.random.uniform(0, max_x)
        new_target_y = np.random.uniform(0, max_y)
        distance = np.sqrt((new_target_x - previous_target[0])**2 + (new_target_y - previous_target[1])**2)
        if distance >= min_distance:
            return (int(new_target_x), int(new_target_y))

def main():
    global last_optimization_time, last_Wm, last_theta_dot, previous_centroid
    ret = PxLApi.initialize(0)
    if not PxLApi.apiSuccess(ret[0]):
        print(f"Error: Unable to initialize a camera! rc = {ret[0]}")
        return EXIT_FAILURE
    hCamera = ret[1]
    pixelType = getPixelType(hCamera)
    if PT_OTHERWISE == pixelType:
        print("Error: Can't handle this camera type.")
        PxLApi.uninitialize(hCamera)
        return EXIT_FAILURE
    rot_azimuth = 0.0  # Initial heading angle in degrees
    rawFrame = create_string_buffer(MAX_WIDTH * MAX_HEIGHT * 2)
    ret = PxLApi.setStreamState(hCamera, PxLApi.StreamState.START)
    if not PxLApi.apiSuccess(ret[0]):
        print("Error: Unable to start the stream.")
        PxLApi.uninitialize(hCamera)
        return EXIT_FAILURE
    try:
        start_time = time.time()
        # Capture the first frame to select the contour
        ret = PxLApi.getNextFrame(hCamera, rawFrame)
        if not PxLApi.apiSuccess(ret[0]):
            print("Error capturing the first frame.")
            return
        frameDesc = ret[1]
        formatedImage = PxLApi.formatImage(rawFrame, frameDesc, PxLApi.ImageFormat.RAW_RGB24)[1]
        npFormatedImage = np.frombuffer(formatedImage, dtype=np.uint8).reshape((int(frameDesc.Roi.fHeight), int(frameDesc.Roi.fWidth), 3))
        resized_frame = cv2.resize(npFormatedImage, (RESIZED_WIDTH, RESIZED_HEIGHT))
        mirrored_frame = cv2.flip(resized_frame, 0)

        # Process the frame and display contours for selection
        contours, _ = process_frame(mirrored_frame)
        for i, cnt in enumerate(contours):
            cv2.drawContours(mirrored_frame, [cnt], -1, (0, 255, 0), 2)
            centroid = get_contour_centroid(cnt)
            if centroid:
                cv2.putText(mirrored_frame, str(i), centroid, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        # Draw the full spline on the image
        cv2.polylines(mirrored_frame, [np.array(list(zip(spline_x, spline_y)), dtype=np.int32)], False, (0, 255, 255), 2)
        cv2.imshow("First Frame - Select Contour", mirrored_frame)
        cv2.waitKey(0)
        contour_index = int(input("Enter the number of the contour you want to track: "))
        selected_contour = contours[contour_index]
        tracked_centroid = get_contour_centroid(selected_contour)  # Initialize the tracked_centroid
        cv2.destroyAllWindows()

        # Initialize current_position and previous_centroid
        current_position = tracked_centroid
        previous_centroid = tracked_centroid

        # Go-to-start phase using MPC with single-target optimization
        start_point = (float(spline_x[0]), float(spline_y[0]))
        reached_start = False
        while not reached_start:
            ret = PxLApi.getNextFrame(hCamera, rawFrame)
            if not PxLApi.apiSuccess(ret[0]):
                break
            frameDesc = ret[1]
            formatedImage = PxLApi.formatImage(rawFrame, frameDesc, PxLApi.ImageFormat.RAW_RGB24)[1]
            npFormatedImage = np.frombuffer(formatedImage, dtype=np.uint8).reshape((int(frameDesc.Roi.fHeight), int(frameDesc.Roi.fWidth), 3))
            resized_frame = cv2.resize(npFormatedImage, (RESIZED_WIDTH, RESIZED_HEIGHT))
            mirrored_frame = cv2.flip(resized_frame, 0)

            # Process frame and get contours
            contours, _ = process_frame(mirrored_frame)
            if len(contours) > 0:
                # Find the contour closest to the previous centroid
                closest_contour = min(contours, key=lambda cnt: np.linalg.norm(np.array(get_contour_centroid(cnt)) - np.array(previous_centroid)))
                new_tracked_centroid = get_contour_centroid(closest_contour)

                if new_tracked_centroid is not None:
                    # Check if the distance is within a reasonable threshold (e.g., 20 pixels)
                    distance = np.linalg.norm(np.array(new_tracked_centroid) - np.array(previous_centroid))
                    if distance <= 20:  # Only update if the movement is reasonable
                        tracked_centroid = new_tracked_centroid
                        current_position = tracked_centroid
                        previous_centroid = tracked_centroid  # Update the previous centroid

            distance_to_start = np.sqrt((current_position[0] - start_point[0]) ** 2 + (current_position[1] - start_point[1]) ** 2)
            if distance_to_start < 7:
                reached_start = True
                print("Reached the start point. Starting spline-following phase.")
                break

            current_time = time.time()
            if current_time - last_optimization_time >= optimization_interval:
                Wm, theta_dot, error, cost_breakdown = calculate_control_parameters_single(
                    current_position, start_point, rot_azimuth, last_Wm, last_theta_dot
                )
                rot_azimuth += np.degrees(theta_dot * optimization_interval)
                rot_azimuth %= 360
                rot_azimuth_surface = transform_to_surface_coordinates(rot_azimuth)
                rot_azimuth_surface_rad = np.radians(rot_azimuth_surface)
                freq = Wm
                send_magnetic_field_data(
                    azimuth=0,
                    inclination=0,
                    magnitude=10,
                    rot_inclination=math.radians(90),
                    freq=freq,
                    rot_azimuth=rot_azimuth_surface_rad
                )
                control_data.append({
                    "time": current_time - start_time,
                    "tracked_centroid": tracked_centroid,
                    "Wm": Wm,
                    "theta_dot": theta_dot,
                    "error": error,
                    "rot_azimuth_surface": rot_azimuth_surface,
                    "rot_azimuth_degrees": rot_azimuth,
                    "frequency": freq,
                    "cost_breakdown": cost_breakdown
                })
                last_Wm, last_theta_dot = Wm, theta_dot
                last_optimization_time = current_time

            frame_with_tracking = mirrored_frame.copy()
            cv2.circle(frame_with_tracking, (int(start_point[0]), int(start_point[1])), 5, (0, 255, 0), -1)
            cv2.circle(frame_with_tracking, tracked_centroid, 5, (0, 0, 255), -1)
            cv2.imshow("Go-to-Start Phase", frame_with_tracking)
            cv2.waitKey(1)

        # Spline-following phase
        frame_list = []
        start_time_phase = time.time()
        while True:
            ret = PxLApi.getNextFrame(hCamera, rawFrame)
            if not PxLApi.apiSuccess(ret[0]):
                break
            frameDesc = ret[1]
            formatedImage = PxLApi.formatImage(rawFrame, frameDesc, PxLApi.ImageFormat.RAW_RGB24)[1]
            npFormatedImage = np.frombuffer(formatedImage, dtype=np.uint8).reshape((int(frameDesc.Roi.fHeight), int(frameDesc.Roi.fWidth), 3))
            resized_frame = cv2.resize(npFormatedImage, (RESIZED_WIDTH, RESIZED_HEIGHT))
            mirrored_frame = cv2.flip(resized_frame, 0)

            # Process frame and get contours using full spline for MPC
            contours, _ = process_frame(mirrored_frame)
            if len(contours) > 0:
                # Find the contour closest to the previous centroid
                closest_contour = min(contours, key=lambda cnt: np.linalg.norm(np.array(get_contour_centroid(cnt)) - np.array(previous_centroid)))
                new_tracked_centroid = get_contour_centroid(closest_contour)

                if new_tracked_centroid is not None:
                    # Check if the distance is within a reasonable threshold (e.g., 20 pixels)
                    distance = np.linalg.norm(np.array(new_tracked_centroid) - np.array(previous_centroid))
                    if distance <= 20:  # Only update if the movement is reasonable
                        tracked_centroid = new_tracked_centroid
                        current_position = tracked_centroid
                        previous_centroid = tracked_centroid  # Update the previous centroid

            final_x, final_y = spline_x[-1], spline_y[-1]
            final_target_distance = np.sqrt((tracked_centroid[0] - final_x) ** 2 + (tracked_centroid[1] - final_y) ** 2)
            if final_target_distance < 10:
                print("Reached the end of the spline")
                break

            current_time = time.time()
            if current_time - last_optimization_time >= optimization_interval:
                Wm, theta_dot, error, cost_breakdown = calculate_control_parameters(
                    tracked_centroid, spline_x, spline_y, rot_azimuth, last_Wm, last_theta_dot, lookahead_offset=5, lookahead_weight=1.0
                )
                rot_azimuth += np.degrees(theta_dot * optimization_interval)
                rot_azimuth %= 360
                rot_azimuth_surface = transform_to_surface_coordinates(rot_azimuth)
                rot_azimuth_surface_rad = np.radians(rot_azimuth_surface)
                freq = Wm
                send_magnetic_field_data(
                    azimuth=0,
                    inclination=0,
                    magnitude=10,
                    rot_inclination=math.radians(90),
                    freq=freq,
                    rot_azimuth=rot_azimuth_surface_rad
                )
                control_data.append({
                    "time": current_time - start_time_phase,
                    "tracked_centroid": tracked_centroid,
                    "Wm": Wm,
                    "theta_dot": theta_dot,
                    "error": error,
                    "rot_azimuth_surface": rot_azimuth_surface,
                    "rot_azimuth_degrees": rot_azimuth,
                    "frequency": freq,
                    "cost_breakdown": cost_breakdown
                })
                last_Wm, last_theta_dot = Wm, theta_dot
                last_optimization_time = current_time

            frame_with_tracking = mirrored_frame.copy()
            cv2.polylines(frame_with_tracking, [np.array(list(zip(spline_x, spline_y)), dtype=np.int32)], False, (0, 255, 255), 2)
            cv2.circle(frame_with_tracking, tracked_centroid, 5, (0, 0, 255), -1)
            cv2.imshow("Camera Feed with Spline Tracking", frame_with_tracking)
            frame_list.append(frame_with_tracking)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        end_time = time.time()
        elapsed_time = end_time - start_time_phase
        fps = len(frame_list) / elapsed_time
        print(f"FPS: {fps}")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_filename = f"camera_feed_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (RESIZED_WIDTH, RESIZED_HEIGHT))
        for frame in frame_list:
            video_writer.write(frame)
        video_writer.release()
        data_filename = f"control_data_{timestamp}.csv"
        df = pd.DataFrame(control_data)
        df.to_csv(data_filename, index=False)
        print(f"Control data saved as {data_filename}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        PxLApi.setStreamState(hCamera, PxLApi.StreamState.STOP)
        PxLApi.uninitialize(hCamera)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
