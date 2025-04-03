#!/usr/bin/env python
import socket
import struct
import math
import time

def send_swimming_command(azimuth, inclination, magnitude, rot_inclination, freq, rot_azimuth):
    """Sends the swimming motion command over TCP/IP."""
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = ('127.0.0.1', 12345)
        client_socket.connect(server_address)

        # Pack six double values into a binary message.
        message = struct.pack('dddddd', azimuth, inclination, magnitude,
                              rot_inclination, rot_azimuth, freq)
        client_socket.sendall(message)
        print(f"Sent: Azimuth={azimuth}, Inclination={inclination}, Magnitude={magnitude}, "
              f"Rot_Inclination={rot_inclination}, Rot_Azimuth={rot_azimuth}, Freq={freq}")
        client_socket.close()
    except Exception as e:
        print(f"Error while sending command: {e}")

def main():
    """
    This script simulates sending a swimming motion command.
    
    All values are sent in radians. For example:
      - The swimming branch in your main code calculates:
          alpha_deg = math.degrees(math.atan(1.0 / delta))
          inclination = math.radians(90 - alpha_deg)
      - We then set:
          rot_inclination = math.radians(90)
          and use a transformed azimuth for swimming (e.g. -rot_azimuth).
    Adjust these values as needed.
    """

    # Example parameters for swimming motion
    # You can adjust delta to simulate different conditions.
    delta = 0.2  
    alpha_deg = math.degrees(math.atan(1.0 / delta))
    inclination = math.radians(90 - alpha_deg)  # computed inclination (in rad)
    
    magnitude = 10           # Example magnitude value
    freq = 1.5               # Example frequency
    rot_azimuth_deg = 45     # Example rotation azimuth in degrees
    
    # For swimming, we transform the azimuth.
    # In your code you use: rot_swim = transform_to_swimming_coordinates(rot_azimuth)
    # Here we simulate that by simply inverting the angle.
    rot_swim_deg = -rot_azimuth_deg  
    azimuth = math.radians(rot_swim_deg)        # Send azimuth in rad
    rot_inclination = math.radians(90)           # Fixed value (90° in rad)
    rot_azimuth = math.radians(rot_swim_deg)       # Send rotated azimuth in rad

    while True:
        send_swimming_command(azimuth, inclination, magnitude, rot_inclination, freq, rot_azimuth)
        time.sleep(1)  # Send the command every second

if __name__ == "__main__":
    main()
