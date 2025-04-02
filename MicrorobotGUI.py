from pixelinkWrapper import*
from ctypes import*
import numpy as np
import cv2
import os
import time
import socket
import struct
import math
from scipy.optimize import minimize

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
TARGET_THRESHOLD = 10  # threshold in pixels

# Global state for mouse interactions and GUI controls
new_target = None         # Updated via right-click to set a new target
new_contour_click = None  # Updated via left-click to reselect the tracked contour
motion_mode = "surface"   # Default motion mode; will be updated by GUI trackbar
paused = False            # Pause flag; when True, robot is commanded to stop

# Initial target location
target_location = (300, 300)  # Example target position (x, y)

# Initialize control state
last_optimization_time = time.time()
last_var, last_theta_dot = 0.0, 0.0
control_data = []  # Array to log control parameters

# Number of targets to reach
max_targets = 5
completed_targets = 0
target_locations = [target_location]

# Global variables for dynamic bounds and magnitude (default values)
surface_lower_bound = 0.1
surface_upper_bound = 2.0
swimming_lower_bound = 0.1
swimming_upper_bound = 0.3
magnitude_value = 10

# Helper function to determine pixel type (mono or color)
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


def calculate_control_parameters(current_position, target_position, rot_azimuth,
                                 last_var, last_theta_dot,
                                 motion_mode="surface"):
    """
    Returns four values:
      Wm_opt, delta_opt, theta_dot_opt, error
    """
    def mpc_cost_function(control_sequence, x, y, theta_degs,
                          target_x, target_y, dt, horizon,
                          last_var, last_theta_dot, motion_mode="surface"):
        cost = 0.0
        C1 = 45.39  # For swimming

        # Example weighting factors
        w_path   = 1.0
        w_Wm_dev = 1.1
        w_change = 1.0
        w_tdot   = 0.5
        w_change_Wm = 0.1
        w_change_tdot = 5.0

        for i in range(horizon):
            var1 = control_sequence[2*i]
            theta_dot = control_sequence[2*i + 1]

            if motion_mode == "surface":
                # Use dynamic bounds for Wm (var1)
                Wm = np.clip(var1, surface_lower_bound, surface_upper_bound)
                V = K * Wm

                theta_degs += np.degrees(theta_dot * dt)
                theta_degs = theta_degs % 360.0
                theta_surf = transform_to_surface_coordinates(theta_degs)
                theta_rad  = np.radians(theta_surf)
                x += V * np.sin(theta_rad) * dt
                y += -V * np.cos(theta_rad) * dt

                cost += w_path * ((x - target_x)**2 + (y - target_y)**2)
                cost += w_Wm_dev * (1.0 - Wm)**2

                if i == 0:
                    cost += w_change * (Wm - last_var)**2
                    cost += w_tdot  * (theta_dot - last_theta_dot)**2
                else:
                    prev_Wm       = control_sequence[2*(i-1)]
                    prev_theta_dot= control_sequence[2*(i-1)+1]
                    cost += w_change_Wm * (Wm - prev_Wm)**2
                    cost += w_change_tdot * (theta_dot - prev_theta_dot)**2

            else:
                # Use dynamic bounds for delta (var1)
                delta = np.clip(var1, swimming_lower_bound, swimming_upper_bound)
                #Wm_ideal = 25.0*delta - 0.67 +8.33
                Wm_ideal = 14
                #Wm = np.clip(Wm_ideal, (25.0*delta - 1.67), 18.0)
                Wm = np.clip(Wm_ideal, 13.00, 15.0)
                theta_degs += np.degrees(theta_dot * dt)
                theta_degs = theta_degs % 360.0
                theta_swim = transform_to_swimming_coordinates(theta_degs)
                theta_rad  = np.radians(theta_swim)
                x += C1 * delta * np.cos(theta_rad) * dt
                y += -C1 * delta * np.sin(theta_rad) * dt

                cost += w_path * ((x - target_x)**2 + (y - target_y)**2)
                cost += w_Wm_dev * (Wm - Wm_ideal)**2
                cost += 0.5 * ((delta - 0.3)**2)

                if i == 0:
                    cost += 1.0 * (delta - last_var)**2
                    cost += 0.5 * (theta_dot - last_theta_dot)**2
                else:
                    prev_delta     = control_sequence[2*(i-1)]
                    prev_theta_dot = control_sequence[2*(i-1)+1]
                    cost += 0.1 * (delta - prev_delta)**2
                    cost += 5.0 * (theta_dot - prev_theta_dot)**2

        return cost

    x, y = current_position
    target_x, target_y = target_position
    dt = optimization_interval

    if motion_mode == "surface":
        bounds = [(surface_lower_bound, surface_upper_bound), (-np.pi, np.pi)] * horizon
        initial_guess = [last_var, 0.0] * horizon
    else:
        bounds = [(swimming_lower_bound, swimming_upper_bound), (-np.pi, np.pi)] * horizon
        initial_guess = [0.3, 0.0] * horizon

    result = minimize(
        mpc_cost_function,
        initial_guess,
        args=(x, y, rot_azimuth, target_x, target_y, dt, horizon,
              last_var, last_theta_dot, motion_mode),
        bounds=bounds,
        method='SLSQP'
    )

    var1_opt, theta_dot_opt = result.x[0], result.x[1]
    error = np.sqrt((x - target_x)**2 + (y - target_y)**2)

    if motion_mode == "surface":
        Wm_opt = var1_opt
        delta_opt = None
    else:
        delta_opt = var1_opt
        Wm_opt = 25.0*delta_opt - 0.67 + 8.17

    return Wm_opt, delta_opt, theta_dot_opt, error


def transform_to_surface_coordinates(rot_azimuth):
    if 0 <= rot_azimuth <= 270:
        transformed_angle = 90 - rot_azimuth
    else:
        transformed_angle = 360 - (rot_azimuth - 90)
    return transformed_angle


def transform_to_swimming_coordinates(rot_azimuth):
    transformed_angle = -rot_azimuth
    if transformed_angle < -180:
        transformed_angle += 360
    elif transformed_angle > 180:
        transformed_angle -= 360
    return transformed_angle


def send_magnetic_field_data(azimuth, inclination, magnitude, rot_inclination, freq, rot_azimuth):
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = ('127.0.0.1', 12345)
        client_socket.connect(server_address)

        message = struct.pack('dddddd', azimuth, inclination, magnitude,
                              rot_inclination, rot_azimuth, freq)
        client_socket.sendall(message)
        print(f"Sent: Azimuth={azimuth}, Inclination={inclination}, "
              f"Rot_Inclination={rot_inclination}, Rot_Azimuth={rot_azimuth}")
        client_socket.close()
    except Exception as e:
        print(f"Error while sending magnetic field data: {e}")
    finally:
        pass


def process_frame(frame):
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


def get_new_target(previous_target, min_distance=50,
                   max_x=RESIZED_WIDTH, max_y=RESIZED_HEIGHT):
    while True:
        new_target_x = np.random.uniform(0, max_x)
        new_target_y = np.random.uniform(0, max_y)
        distance = np.sqrt((new_target_x - previous_target[0])**2 +
                           (new_target_y - previous_target[1])**2)
        if distance >= min_distance:
            return (int(new_target_x), int(new_target_y))

def clear_all_targets():
    global target_locations, completed_targets
    target_locations = []       # Clear all targets
    completed_targets = 0       # Reset completed targets counter
    print("All target locations cleared.")


def mouse_callback(event, x, y, flags, param):
    global new_target, new_contour_click
    if event == cv2.EVENT_RBUTTONDOWN:
        new_target = (x, y)
        print(f"New target selected: {new_target}")
    elif event == cv2.EVENT_LBUTTONDOWN:
        new_contour_click = (x, y)
        print(f"Manual contour selection at: {new_contour_click}")

def main():
    global last_optimization_time, last_var, last_theta_dot, completed_targets
    global target_location, new_target, new_contour_click, motion_mode, paused
    EXIT_FAILURE = 1

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

    rot_azimuth = 0.0
    rawFrame = create_string_buffer(MAX_WIDTH * MAX_HEIGHT * 2)

    ret = PxLApi.setStreamState(hCamera, PxLApi.StreamState.START)
    if not PxLApi.apiSuccess(ret[0]):
        print("Error: Unable to start the stream.")
        PxLApi.uninitialize(hCamera)
        return EXIT_FAILURE

    try:
        # Grab first frame, select contour
        ret = PxLApi.getNextFrame(hCamera, rawFrame)
        if not PxLApi.apiSuccess(ret[0]):
            print("Error capturing the first frame.")
            return

        frameDesc = ret[1]
        formatedImage = PxLApi.formatImage(rawFrame, frameDesc, PxLApi.ImageFormat.RAW_RGB24)[1]
        npFormatedImage = np.frombuffer(formatedImage, dtype=np.uint8).reshape(
            (int(frameDesc.Roi.fHeight), int(frameDesc.Roi.fWidth), 3)
        )

        resized_frame = cv2.resize(npFormatedImage, (RESIZED_WIDTH, RESIZED_HEIGHT))
        mirrored_frame = cv2.flip(resized_frame, 0)

        contours, _ = process_frame(mirrored_frame)
        for i, cnt in enumerate(contours):
            cv2.drawContours(mirrored_frame, [cnt], -1, (0, 255, 0), 2)
            centroid = get_contour_centroid(cnt)
            print(f"Contour {i}: centroid = {centroid}")
            if centroid:
                cv2.putText(mirrored_frame, str(i), centroid,
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("First Frame - Select Contour", mirrored_frame)
        cv2.waitKey(0)
        contour_index = int(input("Enter the number of the contour you want to track: "))
        selected_contour = contours[contour_index]
        tracked_centroid = get_contour_centroid(selected_contour)
        previous_centroid = tracked_centroid  # initialize previous_centroid
        current_position = tracked_centroid
        cv2.destroyAllWindows()

        current_position = tracked_centroid

        # Setup display and control GUI windows
        cv2.namedWindow("Camera Feed with Tracking")
        cv2.setMouseCallback("Camera Feed with Tracking", mouse_callback)
        cv2.namedWindow("Controls")
        cv2.moveWindow("Camera Feed with Tracking", 100, 100)
        cv2.moveWindow("Controls", 700, 100)
        cv2.createTrackbar("Motion Mode", "Controls", 0, 1, lambda x: None)  # 0: surface, 1: swimming
        cv2.createTrackbar("Pause", "Controls", 0, 1, lambda x: None)         # 0: run, 1: pause

        # New trackbars for dynamic parameters
        cv2.createTrackbar("Mag", "Controls", 10, 20, lambda x: None)  # Magnitude (0-20)
        cv2.createTrackbar("Surface LB", "Controls", 10, 100, lambda x: None)  # Surface lower bound (scale: /100)
        cv2.createTrackbar("Surface UB", "Controls", 100, 200, lambda x: None) # Surface upper bound (scale: /100)
        cv2.createTrackbar("Swim LB", "Controls", 10, 35, lambda x: None)      # Swimming lower bound (scale: /100)
        cv2.createTrackbar("Swim UB", "Controls", 30, 35, lambda x: None)        # Swimming upper bound (scale: /100)
        cv2.createTrackbar("Opt Interval", "Controls", 10, 50, lambda x: None)  # Interval in tenths of a second

        frame_list = []
        location_data = []
        start_time = time.time()
        frame_counter = 0

        while True:
            # Update dynamic parameters from trackbars each loop iteration
            surface_lower_bound = cv2.getTrackbarPos("Surface LB", "Controls") / 100.0
            surface_upper_bound = cv2.getTrackbarPos("Surface UB", "Controls") / 100.0
            swimming_lower_bound = cv2.getTrackbarPos("Swim LB", "Controls") / 100.0
            swimming_upper_bound = cv2.getTrackbarPos("Swim UB", "Controls") / 100.0
            magnitude_value = cv2.getTrackbarPos("Mag", "Controls")
            optimization_interval = cv2.getTrackbarPos("Opt Interval", "Controls") / 10.0
            
            opt_val = cv2.getTrackbarPos("Opt Interval", "Controls")
            if opt_val == 0:
                opt_val = 1  # Ensure non-zero value
            optimization_interval = opt_val / 10.0

            ret = PxLApi.getNextFrame(hCamera, rawFrame)
            if not PxLApi.apiSuccess(ret[0]):
                break

            frameDesc = ret[1]
            formatedImage = PxLApi.formatImage(rawFrame, frameDesc, PxLApi.ImageFormat.RAW_RGB24)[1]
            npFormatedImage = np.frombuffer(formatedImage, dtype=np.uint8).reshape(
                (int(frameDesc.Roi.fHeight), int(frameDesc.Roi.fWidth), 3)
            )

            resized_frame = cv2.resize(npFormatedImage, (RESIZED_WIDTH, RESIZED_HEIGHT))
            mirrored_frame = cv2.flip(resized_frame, 0)

            # Process frame and get contours
            contours, _ = process_frame(mirrored_frame)
            valid_contours = [cnt for cnt in contours if get_contour_centroid(cnt) is not None]

            if new_contour_click is not None:
                manual_click_point = new_contour_click
                new_contour_click = None
                if valid_contours:
                    closest_contour = min(
                        valid_contours,
                        key=lambda cnt: np.linalg.norm(np.array(get_contour_centroid(cnt)) - np.array(manual_click_point))
                    )
                    new_tracked_centroid = get_contour_centroid(closest_contour)
                    if new_tracked_centroid is not None:
                        tracked_centroid = new_tracked_centroid
                        current_position = tracked_centroid
                        previous_centroid = tracked_centroid
                        print(f"Updated tracking contour via left click: {tracked_centroid}")
                else:
                    print("No valid contours detected for manual update.")
            else:
                if valid_contours:
                    closest_contour = min(
                        valid_contours,
                        key=lambda cnt: np.linalg.norm(np.array(get_contour_centroid(cnt)) - np.array(previous_centroid))
                    )
                    new_tracked_centroid = get_contour_centroid(closest_contour)
                    if new_tracked_centroid is not None:
                        distance = np.linalg.norm(np.array(new_tracked_centroid) - np.array(previous_centroid))
                        if distance <= 20:
                            tracked_centroid = new_tracked_centroid
                            current_position = tracked_centroid
                            previous_centroid = tracked_centroid
                else:
                    print("No valid contours detected for automatic update.")

            if new_target is not None:
                target_location = new_target
                target_locations.append(target_location)
                new_target = None
                print(f"Target updated via right click: {target_location}")

            distance_to_target = np.linalg.norm(np.array(current_position) - np.array(target_location))
            if distance_to_target < TARGET_THRESHOLD:
                print("Target reached!")
                completed_targets += 1
                if completed_targets < len(target_locations):
                    target_location = target_locations[completed_targets]
                else:
                    target_location = get_new_target(current_position)
                    target_locations.append(target_location)
                    print(f"New target generated: {target_location}")

            frame_with_tracking = mirrored_frame.copy()
            for idx, t_loc in enumerate(target_locations):
                color = (0, 255, 0) if idx < completed_targets else (255, 0, 0)
                cv2.circle(frame_with_tracking, t_loc, 2, color, -1)
            cv2.circle(frame_with_tracking, current_position, 2, (0, 0, 255), -1)

            mode_val = cv2.getTrackbarPos("Motion Mode", "Controls")
            pause_val = cv2.getTrackbarPos("Pause", "Controls")
            motion_mode = "surface" if mode_val == 0 else "swimming"
            paused = True if pause_val == 1 else False

            cv2.putText(frame_with_tracking, f"Mode: {motion_mode}", (10,20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(frame_with_tracking, f"Paused: {paused}", (10,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            current_time = time.time()

            if current_time - last_optimization_time >= optimization_interval:
                if not paused:
                    Wm, delta, theta_dot, error = calculate_control_parameters(
                        current_position, target_location, rot_azimuth,
                        last_var, last_theta_dot, motion_mode=motion_mode
                    )
                    rot_azimuth += np.degrees(theta_dot * optimization_interval)
                    rot_azimuth %= 360
                    freq = Wm
                    # Use the new magnitude value from the trackbar when not paused
                    if motion_mode == "swimming" and delta is not None:
                        alpha_deg = math.degrees(math.atan(1.0 / delta))
                        inclination = math.radians(90 - alpha_deg)
                        send_magnetic_field_data(
                            azimuth=rot_azimuth,
                            inclination=inclination,
                            magnitude=magnitude_value,
                            rot_inclination=math.radians(90),
                            freq=freq,
                            rot_azimuth=np.radians(rot_azimuth)
                        )
                    else:
                        rot_azimuth_surface = transform_to_surface_coordinates(rot_azimuth)
                        rot_azimuth_surface_rad = np.radians(rot_azimuth_surface)
                        send_magnetic_field_data(
                            azimuth=0,
                            inclination=0,
                            magnitude=magnitude_value,
                            rot_inclination=math.radians(90),
                            freq=freq,
                            rot_azimuth=rot_azimuth_surface_rad
                        )
                    last_var = delta if motion_mode == "swimming" else Wm
                    last_theta_dot = theta_dot
                else:
                    # When paused, send a zero-magnitude command to stop the robot
                    rot_azimuth_surface = transform_to_surface_coordinates(rot_azimuth)
                    rot_azimuth_surface_rad = np.radians(rot_azimuth_surface)
                    send_magnetic_field_data(
                        azimuth=0,
                        inclination=0,
                        magnitude=0,
                        rot_inclination=math.radians(90),
                        freq=0,
                        rot_azimuth=rot_azimuth_surface_rad
                    )
                last_optimization_time = current_time
                control_data.append({
                    "time": current_time - start_time,
                    "tracked_centroid": current_position,
                    "target_location": target_location,
                    "Wm": Wm if not paused else 0,
                    "delta": delta if not paused else 0,
                    "theta_dot": theta_dot if not paused else 0,
                    "error": error if not paused else None,
                    "rot_azimuth_degrees": rot_azimuth,
                    "frequency": freq if not paused else 0,
                    "motion_mode": motion_mode,
                    "paused": paused
                })

            cv2.imshow("Camera Feed with Tracking", frame_with_tracking)
            frame_list.append(frame_with_tracking)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                clear_all_targets()

            frame_counter += 1

        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = len(frame_list) / elapsed_time
        print(f"FPS: {fps}")

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_filename = f"camera_feed_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_filename, fourcc, fps,
                                       (RESIZED_WIDTH, RESIZED_HEIGHT))
        for frame in frame_list:
            video_writer.write(frame)
        video_writer.release()

        data_filename = f"control_data_{timestamp}.csv"
        import pandas as pd
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
