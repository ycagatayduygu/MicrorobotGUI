# NOTE: Before running, set FRAME_RATE to your camera's FPS for accurate video timing

from pixelinkWrapper import*
from ctypes import*
from collections import deque
import numpy as np
import cv2
import os
import time
import socket
import struct
import math
from scipy.optimize import minimize
import pandas as pd
import heapq
from scipy.ndimage import binary_dilation, distance_transform_edt

# Camera constants
MAX_WIDTH = 5000
MAX_HEIGHT = 5000
MAX_BYTES_PER_PIXEL = 3
RESIZED_WIDTH = 512
RESIZED_HEIGHT = 512

#ADJUST BEFORE RUNNING!
FRAME_RATE = 8.5

# Control parameters
K = 2.508  # Constant for speed calculation
horizon = 5  # MPC prediction horizon
optimization_interval = 1  # Optimization interval in seconds

TARGET_THRESHOLD = 10  # threshold in pixels
MIN_CONTOUR_AREA = 18         # Normal mode: keep contours larger than this.
MAX_JUMP_DISTANCE    = 15
PREDICTION_THRESHOLD = 15
SMOOTHING_ALPHA      = 0.001
LOST_COUNTER_MAX     = 5
FALLBACK_AREA_FACTOR = 0.3
desired_sharpness    = 112
V_TERM_EXPERIMENT = 5.0   # ← replace 5.0 with your measured vertical-lift term
blur_hist = deque(maxlen=5)

# Path Planning Mode constants
PATH_PLANNING_ASTAR = "A*"
PATH_PLANNING_MPC = "MPC-Direct"
path_planning_mode = PATH_PLANNING_ASTAR  # Default mode

# MPC Obstacle Avoidance Parameters
OBSTACLE_SAFE_DISTANCE = 20  # Default safe distance for MPC obstacle avoidance
w_obstacle = 10.0  # Default weight for obstacle avoidance in MPC cost function

# Global state for mouse interactions and GUI controls
new_target = None         # Updated via right-click to set a new target
new_contour_click = None  # Updated via left-click to reselect the tracked contour
motion_mode = "surface"   # Default motion mode; will be updated by GUI trackbar
paused = False            # Pause flag; when True, robot is commanded to stop
auto_paused = False

# Initial target location
target_location = (300, 300)  # Example target position (x, y)

# Initialize control state
last_optimization_time = time.time()
last_var, last_theta_dot = 0.0, 0.0
last_gamma_dot = 0.0
last_gamma = 0.0
last_z = desired_sharpness
# ADD — needed for signed-blur logic
last_blur        = desired_sharpness
last_blur_slope  = 0.0
last_optimization_blur = desired_sharpness

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

# Waypoint tracking
last_waypoint_time = 0
WAYPOINT_PAUSE_DURATION = 0.5
full_path = None  # Store the complete path for visualization

# -------- A* Planning Setup --------
np.random.seed(42)
N_OBS, R_MIN, R_MAX = 8, 25, 60
BUFFER_PX = 7

# Global variables for obstacle generation
obstacles = []
obs_map = None
occ_buffer = None
dist_to_obs = None
selected_contour = None  # Added for click-to-select functionality

def generate_obstacles(current_position=None, min_distance=70):
    """Generate obstacles, avoiding the current robot position if provided"""
    global obstacles, obs_map, occ_buffer, dist_to_obs, N_OBS, R_MIN, R_MAX
    
    # Generate static obstacle map at 512×512
    obs_map = np.zeros((RESIZED_HEIGHT, RESIZED_WIDTH), dtype=bool)
    obstacles = []
    YY, XX = np.ogrid[:RESIZED_HEIGHT, :RESIZED_WIDTH]
    
    for _ in range(N_OBS):
        cx = np.random.randint(60, RESIZED_WIDTH-60)
        cy = np.random.randint(60, RESIZED_HEIGHT-60)
        r  = np.random.randint(R_MIN, R_MAX)
        
        # Skip this obstacle if it's too close to the robot
        if current_position:
            distance_to_robot = np.hypot(cx - current_position[0], cy - current_position[1])
            if distance_to_robot < (r + min_distance):
                continue
                
        mask = (XX - cx)**2 + (YY - cy)**2 <= r**2
        obs_map[mask] = True
        obstacles.append((cx, cy, r))

    # Inflate for safety buffer
    occ_buffer = binary_dilation(obs_map, iterations=BUFFER_PX)
    dist_to_obs = distance_transform_edt(~obs_map)
    
    print(f"Generated {len(obstacles)} obstacles, avoiding current position")

def initial_selection_callback(event, x, y, flags, param):
    global selected_contour, current_position
    contours, frame = param
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if contours:
            # Find closest contour centroid to click point
            valid_centroids = [get_contour_centroid(cnt) for cnt in contours if get_contour_centroid(cnt) is not None]
            if valid_centroids:
                closest_idx = min(range(len(valid_centroids)), 
                                key=lambda i: np.hypot(valid_centroids[i][0] - x, valid_centroids[i][1] - y))
                
                # Get the corresponding contour
                selected_contour = contours[closest_idx]
                current_position = valid_centroids[closest_idx]
                print(f"Selected contour {closest_idx} at {current_position}")
                
                # Highlight the selected contour and redraw
                highlight_frame = frame.copy()
                cv2.drawContours(highlight_frame, [selected_contour], -1, (0, 255, 255), 2)
                cv2.circle(highlight_frame, current_position, 5, (255, 0, 255), -1)
                cv2.putText(highlight_frame, f"Selected contour {closest_idx}!", current_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                cv2.imshow("First Frame - Select Contour", highlight_frame)

def heuristic(a, b):
    return np.hypot(a[0] - b[0], a[1] - b[1])

def astar(start, goal, occupancy, dist_map, buf=BUFFER_PX, k_pen=1.0):
    moves     = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]
    move_cost = [1,1,1,1,np.sqrt(2)] * 4
    h, w      = occupancy.shape
    open_set  = [(heuristic(start,goal), 0, start, None)]
    came_from = {}
    gscore    = {start: 0}
    while open_set:
        f, g, cur, parent = heapq.heappop(open_set)
        if cur in came_from:
            continue
        came_from[cur] = parent
        if cur == goal:
            break
        for (dx, dy), mc in zip(moves, move_cost):
            nx, ny = cur[0]+dy, cur[1]+dx
            if not (0 <= nx < h and 0 <= ny < w):
                continue
            if occupancy[nx, ny]:
                continue
            d = dist_map[nx, ny]
            soft_pen = (k_pen * max(0, (buf*2 - d))**2 / (buf**2)) if d < buf*3 else 0
            tg = g + mc + soft_pen
            neigh = (nx, ny)
            if tg < gscore.get(neigh, np.inf):
                gscore[neigh] = tg
                heapq.heappush(open_set, (tg + heuristic(neigh, goal), tg, neigh, cur))
    if goal not in came_from:
        return None
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = came_from[node]
    return path[::-1]

def extract_waypoints(path, step=60):
    wpts, acc = [path[0]], 0
    for i in range(1, len(path)):
        acc += np.linalg.norm(np.subtract(path[i], path[i-1]))
        if acc >= step:
            wpts.append(path[i])
            acc = 0
    wpts.append(path[-1])
    return wpts

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
                                 last_var, last_theta_dot, last_gamma, last_gamma_dot, last_z, optimization_interval,
                                 motion_mode="surface", planning_mode=PATH_PLANNING_ASTAR, dist_map=None):
    """
    Returns six values:
      Wm_opt, delta_opt, theta_dot_opt, gamma_dot_opt, error, cost_terms
    """

    # —— NEW 3D STATE INITIALIZATION ——
    # current headings & altitude
    theta_degs = rot_azimuth
    gamma_degs = last_gamma
    z          = last_z  
    blur_slope = last_blur_slope       # ADD
    # your vertical lift term from experiment:
    v_term     = V_TERM_EXPERIMENT  
    
    def mpc_cost_and_breakdown(control_sequence, x, y, theta_degs, z, gamma_degs,
                          target_x, target_y, dt, horizon,
                          last_var, last_theta_dot, last_gamma_dot, motion_mode="surface",
                          planning_mode_internal=PATH_PLANNING_ASTAR, dist_map_internal=None):
        cost = 0.0
        C1 = 45.39  # For swimming
        C_blur = 45.39        # ← put your own calibration here

        target_z = desired_sharpness        
        # initialize per-term accumulators
        path_cost = dev_cost = smooth_cost = tdot_cost = z_cost = gdot_cost = obs_cost = 0.0
        
        # Example weighting factors
        w_path   = 1.0
        w_Wm_dev = 1.1
        w_change = 1.0
        w_tdot   = 0.5
        w_change_Wm = 0.1
        w_change_tdot = 5.0
        w_z = 1.0 
        w_change_gammadot = 3.0    # choose as you like
        # w_obstacle defined globally for tuning via trackbar later
        
        ### ADD
        def signed_blur_error(b , slope , gdot):
            """Return ± error depending on whether γ̇ moves toward or away from focus."""
            if abs(gdot) < 1e-6:            # γ̇ ≈ 0 → ordinary error
                return desired_sharpness - b
            # flip sign if travelling the wrong way
            return (desired_sharpness - b) if slope * gdot >= 0 else -(desired_sharpness - b)

        
        for i in range(horizon):

            if motion_mode == "surface":
                var1 = control_sequence[2*i]
                theta_dot = control_sequence[2*i + 1]
                # Use dynamic bounds for Wm (var1)
                Wm = np.clip(var1, surface_lower_bound, surface_upper_bound)
                V = K * Wm

                theta_degs += np.degrees(theta_dot * dt)
                theta_degs = theta_degs % 360.0
                
                theta_surf = transform_to_surface_coordinates(theta_degs)
                theta_rad  = np.radians(theta_surf)
                x += V * np.sin(theta_rad) * dt
                y += -V * np.cos(theta_rad) * dt

                # Add obstacle avoidance cost for MPC direct planning
                if planning_mode_internal == PATH_PLANNING_MPC and dist_map_internal is not None:
                    # Get integer coordinates for checking distance map
                    pred_x, pred_y = int(np.clip(x, 0, dist_map_internal.shape[1]-1)), int(np.clip(y, 0, dist_map_internal.shape[0]-1))
                    # Get distance to nearest obstacle
                    dist_to_obstacle = dist_map_internal[pred_y, pred_x]
                    # Penalty increases as robot gets closer to obstacles
                    if dist_to_obstacle < OBSTACLE_SAFE_DISTANCE:
                        obstacle_penalty = (OBSTACLE_SAFE_DISTANCE - dist_to_obstacle)**2
                        cost += w_obstacle * obstacle_penalty
                        obs_cost += w_obstacle * obstacle_penalty

                term = (x - target_x)**2 + (y - target_y)**2
                cost += w_path * term
                path_cost += w_path * term                
                
                term = (1.0 - Wm)**2
                cost += w_Wm_dev * term
                dev_cost += w_Wm_dev * term
                
                
                
                
                if i == 0:
                    term = (Wm - last_var)**2
                    cost += w_change * term
                    smooth_cost += w_change * term                    
                    
                    
                    term = (theta_dot - last_theta_dot)**2
                    cost += w_tdot * term
                    tdot_cost += w_tdot * term
                else:
                    prev_Wm       = control_sequence[2*(i-1)]
                    prev_theta_dot= control_sequence[2*(i-1)+1]
                    
                    
                    term = (Wm - prev_Wm)**2
                    cost += w_change_Wm * term
                    smooth_cost += w_change_Wm * term
                
                
                    term = (theta_dot - prev_theta_dot)**2
                    cost += w_change_tdot * term
                    tdot_cost += w_change_tdot * term
                    

            else:
                # Use dynamic bounds for delta (var1)
                Wm_ideal = 14
                Wm = np.clip(Wm_ideal, 13.00, 15.0)
                
                var1     = control_sequence[3*i]
                delta = np.clip(var1, swimming_lower_bound, swimming_upper_bound)
                
                theta_dot = control_sequence[3*i+1]
                gamma_dot = control_sequence[3*i+2]
                 
                theta_degs += np.degrees(theta_dot * dt)
                theta_degs = theta_degs % 360.0
                
                gamma_degs = np.clip(
                    gamma_degs + np.degrees(gamma_dot * dt),
                    0.0,
                    45.0
                )
                gamma_rad = np.radians(gamma_degs)  
                                
                theta_swim = transform_to_swimming_coordinates(theta_degs)
                theta_rad  = np.radians(theta_swim)
                           
                # 3D updates
                x += C1 * delta * np.cos(theta_rad) * dt
                y += C1 * delta * np.sin(theta_rad) * dt
                z += (C_blur * delta * np.sin(gamma_rad) - v_term) * dt
                
                # Add obstacle avoidance cost for MPC direct planning (swimming mode)
                if planning_mode_internal == PATH_PLANNING_MPC and dist_map_internal is not None:
                    # Get integer coordinates for checking distance map
                    pred_x, pred_y = int(np.clip(x, 0, dist_map_internal.shape[1]-1)), int(np.clip(y, 0, dist_map_internal.shape[0]-1))
                    # Get distance to nearest obstacle
                    dist_to_obstacle = dist_map_internal[pred_y, pred_x]
                    # Penalty increases as robot gets closer to obstacles
                    if dist_to_obstacle < OBSTACLE_SAFE_DISTANCE:
                        obstacle_penalty = (OBSTACLE_SAFE_DISTANCE - dist_to_obstacle)**2
                        cost += w_obstacle * obstacle_penalty
                        obs_cost += w_obstacle * obstacle_penalty
                
                #cost updates with saving parts
                #path error
                term = (x - target_x)**2 + (y - target_y)**2
                cost += w_path * term
                path_cost += w_path * term
                
                # speed input error
                term = (Wm - Wm_ideal)**2
                cost += w_Wm_dev * term
                dev_cost += w_Wm_dev * term                
                
                #delta error
                term = (delta - 0.3)**2
                cost += 0.5 * term
                dev_cost += 0.5 * term                
                
                #blurriness error
                signed_err = signed_blur_error(z, blur_slope, gamma_dot)
                term       = signed_err**2
                cost += w_z * term
                z_cost += w_z * term
                
                if i == 0:
                    term = (delta - last_var)**2
                    cost += 1.0 * term
                    smooth_cost += 1.0 * term
                    
                    
                    term = (theta_dot - last_theta_dot)**2
                    cost += 0.5 * term
                    tdot_cost += 0.5 * term
                    
                    term = (gamma_dot - last_gamma_dot)**2
                    cost += w_change_gammadot * term
                    gdot_cost += w_change_gammadot * term
                    
                else:
                    prev_delta     = control_sequence[3*(i-1)]
                    prev_theta_dot = control_sequence[3*(i-1)+1]
                    prev_gamma_dot = control_sequence[3*(i-1) + 2]
                    
                    
                    term = (delta - prev_delta)**2
                    cost += 0.1 * term
                    smooth_cost += 0.1 * term

                    term = (theta_dot - prev_theta_dot)**2
                    cost += 5.0 * term
                    tdot_cost += 5.0 * term
                    
                    term = (gamma_dot - prev_gamma_dot)**2
                    cost += w_change_gammadot * term
                    gdot_cost += w_change_gammadot * term
                    
                    
                    
        breakdown = {
            "path":   path_cost,
            "dev":    dev_cost,
            "smooth": smooth_cost,
            "tdot":   tdot_cost,
            "z":      z_cost,
            "gdot":   gdot_cost,
            "obs":    obs_cost
        }
        return cost, breakdown        

        # right after mpc_cost_and_breakdown
    def mpc_cost_function(control_sequence, *args):
        total, _ = mpc_cost_and_breakdown(control_sequence, *args)
        return total



    x, y = current_position
    target_x, target_y = target_position
    dt = optimization_interval
    gamma_dot_max = np.radians(45) / dt

    if motion_mode == "surface":
        bounds        = [(surface_lower_bound, surface_upper_bound),
                         (-np.pi, np.pi)] * horizon
        initial_guess = [last_var, 0.0] * horizon
    else:
        # now δ, θ̇, γ̇ each step
        bounds        = [(swimming_lower_bound, swimming_upper_bound),
                         (-np.pi, np.pi),
                         (-gamma_dot_max, gamma_dot_max)] * horizon
        initial_guess = [0.3, 0.0, 0.0] * horizon


    result = minimize(
        mpc_cost_function,
        initial_guess,
        args=(x, y, theta_degs, last_z, last_gamma,
              target_x, target_y, dt, horizon,
              last_var, last_theta_dot, last_gamma_dot, motion_mode,
              planning_mode, dist_map),
        bounds=bounds,
        method='SLSQP'
    )

    if not result.success:
        print("MPC failed:", result.message)
        return None


    error     = np.sqrt((x - target_x)**2 + (y - target_y)**2)

    if motion_mode == "surface":
        Wm_opt, theta_dot_opt = result.x[:2]
        delta_opt = None
        gamma_dot_opt = 0.0
    else:
        delta_opt, theta_dot_opt, gamma_dot_opt = result.x[:3]
        Wm_opt = 25.0*delta_opt - 0.67 + 8.17
        
    _, cost_terms = mpc_cost_and_breakdown(
        result.x,
        x, y, theta_degs,               # state
        last_z, last_gamma,  # z, γ, γ̇
        target_x, target_y, dt, horizon,  # target & timing
        last_var, last_theta_dot, last_gamma_dot,         # history
        motion_mode,                      # mode
        planning_mode, dist_map           # planning mode & obstacle map
    )
 

    return Wm_opt, delta_opt, theta_dot_opt, gamma_dot_opt, error, cost_terms


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


def process_frame(frame,
                  min_area=MIN_CONTOUR_AREA,
                  adaptive_C=15,
                  adaptive_blockSize=11,
                  blur_kernel=(5,5)):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        adaptive_blockSize, adaptive_C
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filtered = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    return filtered, processed

def estimate_local_blurriness_fft(gray, centroid, patch_size=64):
    x, y = int(centroid[0]), int(centroid[1])
    half = patch_size//2
    h, w = gray.shape
    x1, x2 = max(0, x-half), min(w, x+half)
    y1, y2 = max(0, y-half), min(h, y+half)
    patch = gray[y1:y2, x1:x2]
    if patch.size<10*10: 
        return 0.0
    f = np.fft.fft2(patch)
    fshift = np.fft.fftshift(f)
    mag = 20*np.log(np.abs(fshift)+1e-8)
    ch, cw = patch.shape
    mask = np.ones((ch,cw),bool)
    c = 5
    mask[ch//2-c:ch//2+c, cw//2-c:cw//2+c] = False
    return float(mag[mask].mean())

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
    global last_optimization_time, last_var, last_theta_dot, last_gamma, completed_targets, last_z, desired_sharpness
    global target_location, new_target, new_contour_click, motion_mode, paused, auto_paused, target_locations
    global surface_lower_bound, surface_upper_bound, swimming_lower_bound, swimming_upper_bound, magnitude_value, optimization_interval
    global last_blur, last_blur_slope, last_optimization_blur, last_gamma_dot
    global last_waypoint_time, WAYPOINT_PAUSE_DURATION, full_path, path_planning_mode, OBSTACLE_SAFE_DISTANCE, w_obstacle
    global current_position, selected_contour

    # A* state
    astar_active = False
    astar_waypoints = []

    EXIT_FAILURE = 1

    video_writer = None
    raw_video_writer = None
    
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
        selection_frame = mirrored_frame.copy()
        for i, cnt in enumerate(contours):
            cv2.drawContours(selection_frame, [cnt], -1, (0, 255, 0), 2)
            centroid = get_contour_centroid(cnt)
            if centroid:
                cv2.putText(selection_frame, str(i), centroid,
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Setup for click-to-select functionality
        cv2.namedWindow("First Frame - Select Contour")
        cv2.setMouseCallback("First Frame - Select Contour", initial_selection_callback, (contours, selection_frame))
        
        # Add instructions for both selection methods
        instruction_frame = selection_frame.copy()
        cv2.putText(instruction_frame, "OPTION 1: Click on a contour to select it", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(instruction_frame, "OPTION 2: Press any key and type the contour number", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(instruction_frame, "Press 'q' to quit", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        for i, cnt in enumerate(contours):
            centroid = get_contour_centroid(cnt)
            if centroid:
                # Print contour numbers for reference
                print(f"Contour {i}: centroid = {centroid}")
        
        # Wait for user to either click or press a key
        current_position = None
        cv2.imshow("First Frame - Select Contour", instruction_frame)
        
        while current_position is None:
            key = cv2.waitKey(100)
            
            # If a key is pressed (except q), switch to manual number entry
            if key > 0 and key != ord('q'):
                cv2.setMouseCallback("First Frame - Select Contour", lambda *args: None)  # Disable mouse callback
                
                # Keep window open but show "waiting for input" message
                input_message = instruction_frame.copy()
                cv2.putText(input_message, "Enter contour number in terminal...", (150, 250),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("First Frame - Select Contour", input_message)
                cv2.waitKey(1)  # Brief update to show message
                
                try:
                    contour_index = int(input("Enter the number of the contour you want to track: "))
                    if 0 <= contour_index < len(contours):
                        selected_contour = contours[contour_index]
                        temp_position = get_contour_centroid(selected_contour)
                        
                        if temp_position is not None:
                            current_position = temp_position
                            print(f"Selected contour {contour_index} via manual entry at {current_position}")
                            
                            # Show confirmation
                            confirm_frame = instruction_frame.copy()
                            cv2.drawContours(confirm_frame, [selected_contour], -1, (0, 255, 255), 2)
                            cv2.circle(confirm_frame, current_position, 5, (255, 0, 255), -1)
                            cv2.putText(confirm_frame, f"Selected contour {contour_index}!", current_position,
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                            cv2.imshow("First Frame - Select Contour", confirm_frame)
                            cv2.waitKey(1)  # Update display
                        else:
                            print(f"Error: Contour {contour_index} has no valid centroid. Please select another.")
                            cv2.imshow("First Frame - Select Contour", instruction_frame)
                            cv2.waitKey(1)  # Refresh display
                    else:
                        print(f"Error: Invalid contour index. Please enter a number between 0 and {len(contours)-1}.")
                        cv2.imshow("First Frame - Select Contour", instruction_frame)
                        cv2.waitKey(1)  # Refresh display
                except ValueError:
                    print("Error: Please enter a valid integer for the contour number.")
                    cv2.imshow("First Frame - Select Contour", instruction_frame)
                    cv2.waitKey(1)  # Refresh display
            
            # If user clicked (callback set current_position), break the loop
            if current_position is not None:
                break
                
            # Allow quitting
            if key == ord('q'):
                return

        cv2.waitKey(500)  # Brief pause to show selection
        
        # Now generate obstacles AFTER contour selection
        cv2.putText(selection_frame, "Press 'g' to generate obstacles", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("First Frame - Select Contour", selection_frame)
        
        # Wait for 'g' key to generate obstacles
        waiting_for_obstacles = True
        while waiting_for_obstacles:
            key = cv2.waitKey(100)
            if key == ord('g'):
                generate_obstacles(current_position)
                waiting_for_obstacles = False
            elif key == ord('q'):
                return
        
        cv2.destroyAllWindows()
        
        # Setup display and control GUI windows
        cv2.namedWindow("Camera Feed with Tracking")
        cv2.setMouseCallback("Camera Feed with Tracking", mouse_callback)
        cv2.namedWindow("Controls")
        cv2.moveWindow("Camera Feed with Tracking", 100, 100)
        cv2.moveWindow("Controls", 700, 100)
        cv2.createTrackbar("Motion Mode", "Controls", 0, 1, lambda x: None)  # 0: surface, 1: swimming
        cv2.createTrackbar("Pause", "Controls", 0, 1, lambda x: None)         # 0: run, 1: pause
        cv2.createTrackbar("Planning Mode", "Controls", 0, 1, lambda x: None) # 0: A*, 1: MPC-Direct
        cv2.createTrackbar("Obstacle Distance", "Controls", OBSTACLE_SAFE_DISTANCE, 40, lambda x: None) # Safe distance
        cv2.createTrackbar("Obstacle Weight", "Controls", int(w_obstacle), 50, lambda x: None) # Weight for obstacle avoidance

        # New trackbars for dynamic parameters
        cv2.createTrackbar("Mag", "Controls", 10, 20, lambda x: None)  # Magnitude (0-20)
        cv2.createTrackbar("Surface LB", "Controls", 10, 100, lambda x: None)  # Surface lower bound (scale: /100)
        cv2.createTrackbar("Surface UB", "Controls", 100, 200, lambda x: None) # Surface upper bound (scale: /100)
        cv2.createTrackbar("Swim LB", "Controls", 10, 35, lambda x: None)      # Swimming lower bound (scale: /100)
        cv2.createTrackbar("Swim UB", "Controls", 30, 35, lambda x: None)        # Swimming upper bound (scale: /100)
        cv2.createTrackbar("Opt Interval", "Controls", 10, 50, lambda x: None)  # Interval in tenths of a second
        cv2.createTrackbar("Target Z", "Controls", desired_sharpness, 200, lambda x: None)

        location_data = []
        start_time = time.time()
        frame_counter = 0
        lost_counter = 0          # consecutive "lost" frames
        
        # — start streaming video to disk immediately —
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer     = cv2.VideoWriter(f"camera_feed_{timestamp}.mp4",     fourcc, FRAME_RATE, (RESIZED_WIDTH, RESIZED_HEIGHT))
        raw_video_writer = cv2.VideoWriter(f"camera_feed_{timestamp}_raw.mp4", fourcc, FRAME_RATE, (RESIZED_WIDTH, RESIZED_HEIGHT))

        if not video_writer.isOpened():
            print("ERROR: main video_writer failed to open—check your codec/container.")
        if not raw_video_writer.isOpened():
            print("ERROR: raw_video_writer failed to open—check your codec/container.")


        ####If either prints, you know you have a codec/container mismatch. In that case try a more universal combo, e.g.   
        #fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #video_writer = cv2.VideoWriter(f"camera_feed_{timestamp}.avi", fourcc, FRAME_RATE, (RESIZED_WIDTH, RESIZED_HEIGHT))


        while True:
            # Update dynamic parameters from trackbars each loop iteration
            surface_lower_bound = cv2.getTrackbarPos("Surface LB", "Controls") / 100.0
            surface_upper_bound = cv2.getTrackbarPos("Surface UB", "Controls") / 100.0
            swimming_lower_bound = cv2.getTrackbarPos("Swim LB", "Controls") / 100.0
            swimming_upper_bound = cv2.getTrackbarPos("Swim UB", "Controls") / 100.0
            magnitude_value = cv2.getTrackbarPos("Mag", "Controls")
            opt_val = cv2.getTrackbarPos("Opt Interval", "Controls")
            desired_sharpness = cv2.getTrackbarPos("Target Z", "Controls")
            
            # Read planning mode from trackbar
            planning_val = cv2.getTrackbarPos("Planning Mode", "Controls")
            current_planning_mode = PATH_PLANNING_ASTAR if planning_val == 0 else PATH_PLANNING_MPC
            
            # Check if planning mode changed
            if current_planning_mode != path_planning_mode:
                print(f"Switching planning mode: {path_planning_mode} -> {current_planning_mode}")
                path_planning_mode = current_planning_mode
                # Reset path planning state
                astar_active = False
                full_path = None
                
            # Update obstacle safe distance
            OBSTACLE_SAFE_DISTANCE = cv2.getTrackbarPos("Obstacle Distance", "Controls")
            
            # Update obstacle avoidance weight
            w_obstacle = float(cv2.getTrackbarPos("Obstacle Weight", "Controls"))
            
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

            
            raw_video_writer.write(mirrored_frame)
            
            
            # Process frame and get contours
            # 1) first get a gray copy to measure blur:
            gray = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2GRAY)
            
            if current_position is None:
                # nothing to track in this frame; skip the rest of the loop
                blur_score = last_z
                continue
            
            raw_blur = estimate_local_blurriness_fft(gray, current_position)
            last_blur = raw_blur
        
            # sliding-window median denoise
            blur_hist.append(raw_blur)
            blur_score = float(sorted(blur_hist)[len(blur_hist)//2])
        
            # low-pass filtered blur for controller
            alpha_blur = 0.6    # tune 0…1 (smaller → smoother)
            last_z = alpha_blur * blur_score + (1 - alpha_blur) * last_z
            
            
            # 2) choose parameters based on blur:
            if blur_score < desired_sharpness:
                adaptive_C = 5
                blur_kernel = (1,1)
            else:
                adaptive_C = 15
                blur_kernel = (5,5)
            
            # 3) now detect contours with your dynamic settings:
            contours, processed_binary = process_frame(
                mirrored_frame,
                min_area=MIN_CONTOUR_AREA,
                adaptive_C=adaptive_C,
                adaptive_blockSize=11,    # you can also make this dynamic if you like
                blur_kernel=blur_kernel
            )
            
            # (you can optionally show processed_binary in a small window for debugging)
            #cv2.imshow("Processed Binary", processed_binary)            
            valid_contours = [cnt for cnt in contours if get_contour_centroid(cnt) is not None]
            valid_centroids = [get_contour_centroid(cnt) for cnt in valid_contours]
            candidate_pt = current_position
            # --------------------------------- new block START ---------------------------------
            # Compute candidate jump distance from the last tracked centroid
            if valid_contours:
                # work directly with centroids
                candidate_pt = min(
                    valid_centroids,
                    key=lambda pt: np.linalg.norm(np.array(pt) - np.array(current_position))
                )

                jump_distance = np.linalg.norm(np.array(candidate_pt) - np.array(current_position))
            else:
                jump_distance = 0
            
            # Blur OR jump too big  →  widen search criteria
            if (jump_distance > PREDICTION_THRESHOLD) or (blur_score < desired_sharpness):
                adaptive_factor   = max(0.1, FALLBACK_AREA_FACTOR - lost_counter * 0.5)
                fallback_min_area = MIN_CONTOUR_AREA * adaptive_factor
                contours_fb, _ = process_frame(mirrored_frame,
                                               min_area=fallback_min_area,
                                               adaptive_C=adaptive_C,
                                               adaptive_blockSize=11,
                                               blur_kernel=blur_kernel)
                # keep only the contour objects, not centroids
                valid_contours = [cnt for cnt in contours_fb
                                  if get_contour_centroid(cnt) is not None]
                # now build a separate list of centroid tuples
                valid_centroids = [get_contour_centroid(cnt) for cnt in valid_contours]
                # update candidate_pt to the new closest centroid
                if valid_centroids:
                    candidate_pt = min(
                        valid_centroids,
                        key=lambda pt: np.linalg.norm(np.array(pt) - np.array(current_position))
                    )

            # ---------------------------------- new block END ----------------------------------
                        
            if new_contour_click is not None:
                manual_click_point = new_contour_click
                new_contour_click = None
                if valid_contours:
                    
                    
                    # pick the centroid list, find the one closest to the click
                    closest_centroid = min(
                        valid_centroids,
                        key=lambda pt: np.linalg.norm(np.array(pt) - np.array(manual_click_point))
                    )
                    current_position = closest_centroid
                    print(f"Updated tracking contour via left click: {current_position}")




            # ---------------- replace simple update with this -------------
            if valid_contours:
                # pick the closest centroid (already have candidate_pt from above)
                jump_distance = np.linalg.norm(np.array(candidate_pt) - np.array(current_position))
            
                if jump_distance > PREDICTION_THRESHOLD:
                    lost_counter += 1
                    if lost_counter >= LOST_COUNTER_MAX:
                        current_position  = candidate_pt
                        lost_counter = 0
                else:
                    lost_counter = 0
                    if jump_distance > MAX_JUMP_DISTANCE:
                        # soft update
                        current_position = tuple(np.array(current_position) +
                                                 SMOOTHING_ALPHA *
                                                 (np.array(candidate_pt) - np.array(current_position)))
                    else:
                        current_position = candidate_pt


            if new_target is not None:
                # Clear previous waypoints when setting a new target
                target_locations = []
                completed_targets = 0
                astar_active = False
                full_path = None

                if current_position is not None:
                    # Check if target is inside an obstacle
                    target_in_obstacle = False
                    for cx, cy, r in obstacles:
                        if np.hypot(new_target[0] - cx, new_target[1] - cy) <= r + BUFFER_PX:
                            target_in_obstacle = True
                            print("Warning: Target point is inside or too close to an obstacle. Please select a different target.")
                            auto_paused = True
                            break
                    
                    if not target_in_obstacle:
                        if path_planning_mode == PATH_PLANNING_ASTAR:
                            # A* planning mode
                            start = (current_position[1], current_position[0])
                            goal = (new_target[1], new_target[0])
                            path = astar(start, goal, occ_buffer, dist_to_obs, buf=BUFFER_PX)
                            if path:
                                full_path = path  # Store the complete path
                                raw_wpts = extract_waypoints(path)
                                astar_waypoints = [(col, row) for (row, col) in raw_wpts]
                                target_locations = list(astar_waypoints)
                                completed_targets = 0
                                astar_active = True
                                auto_paused = False
                                last_waypoint_time = time.time()
                                print(f"A* planned {len(astar_waypoints)} waypoints")
                            else:
                                print("A* planning failed: no valid path to target. Please select a different target.")
                                auto_paused = True
                        else:
                            # MPC Direct planning mode - set target directly
                            target_location = new_target
                            target_locations = [target_location]  # Single target for MPC
                            print(f"MPC Direct target set to {target_location}")
                            auto_paused = False
                new_target = None

            if path_planning_mode == PATH_PLANNING_ASTAR and astar_active:
                curr_wp = target_locations[completed_targets]
                distance_to_target = np.linalg.norm(np.array(current_position) - np.array(curr_wp))
                current_time = time.time()
                
                if distance_to_target < TARGET_THRESHOLD:
                    if completed_targets < len(target_locations) - 1:
                        if current_time - last_waypoint_time >= WAYPOINT_PAUSE_DURATION:
                            completed_targets += 1
                            print(f"Waypoint {completed_targets} reached, moving to next")
                            last_waypoint_time = current_time
                        else:
                            auto_paused = True  # Pause briefly at waypoint
                    else:
                        print("Final goal reached! Pausing.")
                        auto_paused = True
                        astar_active = False
                else:
                    # If we're moving between waypoints, ensure we're not paused
                    if current_time - last_waypoint_time >= WAYPOINT_PAUSE_DURATION:
                        auto_paused = False
            elif path_planning_mode == PATH_PLANNING_MPC and target_locations:
                # For MPC direct, check if we've reached the target
                current_target = target_locations[0]  # Only one target in MPC mode
                distance_to_target = np.linalg.norm(np.array(current_position) - np.array(current_target))
                if distance_to_target < TARGET_THRESHOLD:
                    if not auto_paused:
                        print("MPC Direct target reached! Pausing.")
                        auto_paused = True

            frame_with_tracking = mirrored_frame.copy()
            
            # Draw obstacles first
            for cx, cy, r in obstacles:
                cv2.circle(frame_with_tracking, (cx, cy), r, (0,0,255), 2)
                cv2.circle(frame_with_tracking, (cx, cy), r + BUFFER_PX, (255,0,0), 1)

            # Draw path and waypoints based on planning mode
            if path_planning_mode == PATH_PLANNING_ASTAR:
                if astar_active or target_locations:
                    draw_path_and_waypoints(frame_with_tracking, full_path, target_locations, completed_targets)
            else:  # MPC Direct mode
                if target_locations:
                    # Just show the target
                    cv2.circle(frame_with_tracking, target_locations[0], 5, (0, 255, 255), -1)
                    cv2.putText(frame_with_tracking, "MPC Target", 
                               (target_locations[0][0]+10, target_locations[0][1]),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Draw current position
            cv2.circle(frame_with_tracking, current_position, 3, (255, 255, 0), -1)

            mode_val = cv2.getTrackbarPos("Motion Mode", "Controls")
            manual_pause = True if cv2.getTrackbarPos("Pause", "Controls") == 1 else False
            motion_mode = "surface" if mode_val == 0 else "swimming"
            paused = manual_pause or auto_paused


            cv2.putText(frame_with_tracking, f"Mode: {motion_mode}", (10,20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(frame_with_tracking, f"Paused: {paused}", (10,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(frame_with_tracking,
                        f"Blur: {blur_score:.1f} / {desired_sharpness}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            cv2.putText(frame_with_tracking, f"Planning: {path_planning_mode}", (10,80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)


            current_time = time.time()

            if current_time - last_optimization_time >= optimization_interval:
                if not paused:
                    # Determine goal based on planning mode
                    if path_planning_mode == PATH_PLANNING_ASTAR and astar_active and target_locations:
                        goal_for_control = target_locations[completed_targets]
                    else:  # MPC Direct or no active A* path
                        if target_locations:
                            goal_for_control = target_locations[0]  # Use the direct target
                        else:
                            goal_for_control = target_location
                    
                    out = calculate_control_parameters(
                        current_position, goal_for_control, rot_azimuth,
                        last_var, last_theta_dot, last_gamma, last_gamma_dot, last_z, optimization_interval,
                        motion_mode=motion_mode,
                        planning_mode=path_planning_mode,
                        dist_map=dist_to_obs
                    )
                    # convergence check
                    if out is None:
                        print("MPC failed to converge this cycle, skipping control update")
                        continue
                    
                    Wm, delta, theta_dot, gamma_dot, error, cost_terms = out
                    last_gamma_dot = gamma_dot

                    # Calculate blur slope between optimization times
                    last_blur_slope = (blur_score - last_optimization_blur) / optimization_interval
                    last_optimization_blur = blur_score  # Update for next cycle

                    rot_azimuth += np.degrees(theta_dot * optimization_interval)
                    rot_azimuth %= 360
                    

                    # increment
                    gamma = (last_gamma + np.degrees(gamma_dot * optimization_interval))
                    # now clamp into [0,30]
                    gamma = min(max(gamma, 30.0), 45.0)
                    last_gamma = gamma

                    
                    freq = Wm
                    rot_inclination = math.radians(90) - math.radians(gamma)

                    # Use the new magnitude value from the trackbar when not paused
                    if motion_mode == "swimming" and delta is not None:
                        alpha_deg = math.degrees(math.atan(1.0 / delta))
                        inclination = rot_inclination - math.radians(alpha_deg)
                        # Use transform_to_swimming_coordinates for swimming mode:
                        rot_swim = transform_to_swimming_coordinates(rot_azimuth)
                        send_magnetic_field_data(
                            azimuth=np.radians(rot_swim),
                            inclination=inclination,
                            magnitude=magnitude_value,
                            rot_inclination=rot_inclination,  # Sending as radian; see note below
                            freq=freq,
                            rot_azimuth=np.radians(rot_swim)
                        )
                    else:
                        rot_azimuth_surface = transform_to_surface_coordinates(rot_azimuth)
                        rot_azimuth_surface_rad = np.radians(rot_azimuth_surface)
                        rot_inclination=math.radians(90)
                        inclination=0.0
                        azimuth=0.0
                        send_magnetic_field_data(
                            azimuth=azimuth,
                            inclination=inclination,
                            magnitude=magnitude_value,
                            rot_inclination=rot_inclination,
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
                        rot_inclination=0,
                        freq=0,
                        rot_azimuth=rot_azimuth_surface_rad
                    )
                    # ↓ ensure these exist even when paused ↓
                    Wm        = 0.0
                    delta     = 0.0
                    theta_dot = 0.0
                    gamma_dot = 0.0
                    error     = None  
                    gamma = last_gamma
                    # define cost_terms so data logging won't error when paused
                    cost_terms = {"path":0, "dev":0, "smooth":0, "tdot":0, "z":0, "gdot":0, "obs":0}
                
                last_optimization_time = current_time
                delta_safe = delta if ('delta' in locals() and not paused) else 0

                control_data.append({
                    "time": current_time - start_time,
                    "tracked_centroid": current_position,
                    "target_location": target_location,
                    "Wm": Wm if not paused else 0,
                    "delta": delta_safe,
                    "theta_dot": theta_dot if not paused else 0,
                    "error": error if not paused else None,
                    "rot_azimuth_degrees": rot_azimuth,
                    "frequency": freq if not paused else 0,
                    "motion_mode": motion_mode,
                    "paused": paused,
                    "inclination_deg": math.degrees(inclination) if (not paused and motion_mode=="swimming") else 0,
                    "rot_inclination_deg": math.degrees(rot_inclination) if not paused else 0,
                    "gamma_dot_deg": math.degrees(gamma_dot) if not paused else 0,
                    "gamma_deg":     gamma,
                    "blur_score": blur_score,           # raw FFT-based blur
                    "smoothed_blur": last_z,            # low-pass filtered blur
                    "magnitude": magnitude_value,
                    "planning_mode": path_planning_mode,
                    "obstacle_distance": OBSTACLE_SAFE_DISTANCE,
                    "cost_path":   cost_terms.get("path", 0),
                    "cost_dev":    cost_terms.get("dev", 0),
                    "cost_smooth": cost_terms.get("smooth", 0),
                    "cost_tdot":   cost_terms.get("tdot", 0),
                    "cost_z":      cost_terms.get("z", 0),
                    "cost_gdot":   cost_terms.get("gdot", 0),
                    "cost_obs":    cost_terms.get("obs", 0)
                })

            cv2.imshow("Camera Feed with Tracking", frame_with_tracking)
            
            #frame_list.append(frame_with_tracking)
            video_writer.write(frame_with_tracking)
            
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                clear_all_targets()

            frame_counter += 1


        # compute actual loop FPS from frame_counter
        elapsed_time = time.time() - start_time
        fps = frame_counter / elapsed_time
        print(f"Measured FPS: {fps:.2f}")




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
        # close the on-the-fly video writers
        #video_writer.release()
        #raw_video_writer.release()
        
        # close the on-the-fly video writers if they were created
        if video_writer is not None and video_writer.isOpened():
           video_writer.release()
        if raw_video_writer is not None and raw_video_writer.isOpened():
           raw_video_writer.release()


def draw_path_and_waypoints(frame, path, waypoints, completed_targets):
    """Draw the full path and color-coded waypoints"""
    # Draw full path if available
    if path:
        for i in range(len(path) - 1):
            pt1 = (path[i][1], path[i][0])  # Convert back from (row,col) to (x,y)
            pt2 = (path[i+1][1], path[i+1][0])
            cv2.line(frame, pt1, pt2, (128, 128, 255), 1)  # Light red line for path
    
    # Draw waypoints with color indicating progress
    for idx, t_loc in enumerate(waypoints):
        if idx < completed_targets:
            color = (0, 255, 0)  # Green for completed
        elif idx == completed_targets:
            color = (0, 255, 255)  # Yellow for current target
        else:
            color = (0, 0, 255)  # Red for upcoming
        cv2.circle(frame, t_loc, 3, color, -1)
        # Draw waypoint number
        cv2.putText(frame, str(idx+1), 
                   (t_loc[0]+10, t_loc[1]+10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


if __name__ == "__main__":
    main()