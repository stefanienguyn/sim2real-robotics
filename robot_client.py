from argparse import ArgumentParser
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from time import sleep

import cv2
import numpy as np
from PIL import Image
from rpc import RPCClient

from jetbot import Camera, Robot
import time
import csv


class Bot:
    def __init__(self, mtx, dist):
        self.robot = Robot()
        self.camera = Camera.instance(width=224, height=224)
        self.mtx = mtx  # Camera matrix for calibration
        self.dist = dist  # Distortion coefficients for calibration

    def calibrate_image(self, image_filename):
        # Load the image and apply camera calibration
        image = cv2.imread(str(image_filename))
        h, w = image.shape[:2]

        # Get optimal camera matrix
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.mtx, self.dist, (w, h), 0, (w, h)
        )

        # Apply un-distortion
        dst = cv2.undistort(image, self.mtx, self.dist, None, new_camera_matrix)

        # Crop the image to the valid region of interest (ROI)
        x, y, w, h = roi  # Extract ROI coordinates
        dst = dst[y : y + h, x : x + w]  # Crop the image using ROI
        cv2.imwrite(image_filename, dst)  # Save the calibrated image

    def save_image(self, image_filename):
        # Capture image from camera, save it, and calibrate
        image_arr = self.camera.value  # Get current camera frame
        image = Image.fromarray(image_arr)
        image.save(image_filename)
        # TODO: change this so that we don'e save the image and then load it and calibrate it and save it again
        self.calibrate_image(image_filename)

    def execute_command(self, action_to_take, speed):
        # Execute movement commands based on action
        if action_to_take == 0 or action_to_take == "forward":
            self.robot.forward(speed=speed)
        elif action_to_take == 1 or action_to_take == "left":
            self.robot.left(speed=speed*1.01)
        elif action_to_take == 2 or action_to_take == "right":
            self.robot.right(speed=speed*1.01)
        else:
            raise ValueError(f"Unknown action: {action_to_take}")

    def stop(self):
        self.robot.stop()


def calibrate_camera():
    "Calibrate the camera using chessboard images."

    # Termination criteria for corner refinement
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    # Prepare object points in the world space
    object_points = np.zeros((6 * 9, 3), np.float32) * 56

    # Define the grid of object points
    object_points[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    all_object_points = []
    all_image_points = []

    path = Path(__file__).resolve().parent / "camera_calibration" / "images"

    files = Path(path).glob("*.png")

    # TODO: we should only compute the camera calibration once, not every time we run the robot client
    #
    for image_file in files:
        img = cv2.imread(str(image_file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        retval, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if retval:
            all_object_points.append(object_points)

            # Refine corner positions
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            all_image_points.append(corners2)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (9, 6), corners2, retval)

    # Perform camera calibration to get camera matrix and distortion coefficients
    _, mtx, dist, _, _ = cv2.calibrateCamera(
        all_object_points, all_image_points, gray.shape[::-1], None, None
    )

    return mtx, dist


# TODO: look into using a heartbeat instead
def keyboard_kill_switch(q: Queue, done: Event):
    # Monitor for user input to stop the robot
    while not done.is_set():
        user_input = input("Enter x to stop the robot at any point: \n")
        if user_input == "x":
            q.put_nowait("x")
            done.set()


def main():
    parser = ArgumentParser(description="Jetbot Robot Client")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./saved_images",
        help="Directory to save images",
    )
    parser.add_argument(
        "--max_actions", type=int, default=150, help="Maximum number of actions"
    )
    parser.add_argument("--speed", type=float, default=2.5, help="Speed of the robot")
    parser.add_argument(
        "--duration", type=float, default=1.0, help="Duration of each action (seconds)"
    )
    parser.add_argument(
        "--log", type=str, default="inference_log.csv", help="Name of .csv log"
    )
    args = parser.parse_args()

    
    # Establish connection with the RPC server
    server = RPCClient("127.0.0.1", 8033)
    server.connect()

    mtx, dist = calibrate_camera()
    bot = Bot(mtx, dist)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Event to signal when to stop the robot
    done = Event()

    # Queue to handle stop signals
    q = Queue()

    look_thread = Thread(target=keyboard_kill_switch, args=(q, done))
    look_thread.start()

    logfile = open(args.log, "w", newline="")
    logwriter = csv.writer(logfile)
    logwriter.writerow(["step", "image_filename", "action", "inference_time"])

    for action_step in range(args.max_actions):
        # Generate filename for the image
        image_filename = f"{output_dir}/{action_step:04}.png"
        bot.save_image(image_filename)

        # Execute the command from server model
        start = time.time()
        action = server.model_run(image_filename)
        inference_time = time.time() - start

        bot.execute_command(action, args.speed)

        logwriter.writerow([action_step, image_filename, action, inference_time])

        sleep(args.duration)

        # Stop the robot after each action
        bot.stop()

        if not q.empty():
            # Check if stop signal is in the queue
            print("Stopping the robot...")
            bot.stop()  # Stop the robot
            break

    
    logfile.close()


    server.disconnect()  # Disconnect from the server


if __name__ == "__main__":
    main()
