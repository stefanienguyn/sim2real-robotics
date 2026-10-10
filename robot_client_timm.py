from argparse import ArgumentParser
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from time import sleep

import cv2
import numpy as np
from rpc import RPCClient

from jetbot import Camera, Robot
import time
import csv


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------

class Bot:
    def __init__(self):
        self.robot = Robot()
        self.camera = Camera.instance(width=224, height=224)

    def save_image(self, image_filename: str):
        """
        Capture a raw frame and save as JPEG without any correction.
        Correction is applied on the model server side, matching the dataset
        preprocessing pipeline exactly.
        """
        img_rgb = self.camera.value                          # uint8 HxWx3 RGB
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # Crop bottom half — same as model server apply_corrections
        h, w = img_bgr.shape[:2]
        bottom_half = img_bgr[h // 2 : h, 0 : w]
        resized = cv2.resize(bottom_half, (w, h), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(image_filename), resized, [cv2.IMWRITE_JPEG_QUALITY, 80])

    def execute_command(self, action_to_take, speed):
        if action_to_take == 0 or action_to_take == "forward":
            self.robot.forward(speed=speed)
        elif action_to_take == 1 or action_to_take == "left":
            #The physical motors are flipped 
            self.robot.right(speed=speed * 1.01)
        elif action_to_take == 2 or action_to_take == "right":
            #The physical motors are flipped 
            self.robot.left(speed=speed * 1.01)
        else:
            raise ValueError(f"Unknown action: {action_to_take}")


    def stop(self):
        self.robot.stop()
        


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------

def keyboard_kill_switch(q: Queue, done: Event):
    while not done.is_set():
        user_input = input("Enter x to stop the robot at any point: \n")
        if user_input == "x":
            q.put_nowait("x")
            done.set()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = ArgumentParser(description="Jetbot Robot Client")
    parser.add_argument("--output_dir", type=str, default="./saved_images",
                        help="Directory to save images")
    parser.add_argument("--max_actions", type=int, default=999999,
                        help="Maximum number of actions")
    parser.add_argument("--speed", type=float, default=2.5,
                        help="Speed of the robot")
    parser.add_argument("--duration", type=float, default=0.2,
                        help="Duration of each action (seconds)")
    parser.add_argument("--log", type=str, default="inference_log.csv",
                        help="Name of .csv log")
    args = parser.parse_args()

    server = RPCClient("127.0.0.1", 8033)
    server.connect()

    bot = Bot()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    done = Event()
    q = Queue()

    look_thread = Thread(target=keyboard_kill_switch, args=(q, done))
    look_thread.start()
    
    log_dir = Path("inference_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / args.log

    logfile = open(args.log, "w", newline="")
    logwriter = csv.writer(logfile)
    logwriter.writerow(["step", "image_filename", "action", "inference_time"])
    

    for action_step in range(args.max_actions):
        image_filename = str(output_dir / f"{action_step:04}.jpg")
        
        # 1. Stop first
        bot.stop()
        
        # 2. Wait for camera to settle (no motion blur)
        time.sleep(0.3)

        # 3. Save raw JPEG — all correction happens on the model server
        bot.save_image(image_filename)
        
        # 4. Predict on clean frame
        start = time.time()
        
        action = server.model_run(image_filename)
    
        inference_time = time.time() - start

        bot.execute_command(action, args.speed)
        sleep(args.duration)
        
        bot.stop()
        logwriter.writerow([action_step, image_filename, action, inference_time])

        if not q.empty():
            print("Stopping the robot...")
            bot.stop()
            break

    logfile.close()
    server.disconnect()


if __name__ == "__main__":
    main()