#  JetBot Inference

The JetBot SDK uses an older, incompatible version of Python when compared to the latest versions of PyTorch and timm. To overcome this limitation, we must create two separate Python environments: one for JetBot control and another for machine learning. We communicate between these environments using Remote Procedure Calls (RPCs). The JetBot captures an image, applies camera corrections, saves it locally, and calls the model inference function with the image filename. The model loads the corrected image, applies the same preprocessing pipeline, runs the inference, outputs a motion direction, and then calls the robot motion function with that direction.

`rpcs.py` This file contains the RPC server and RPC client classes. The RPC server is responsible for starting the server, receiving RPC calls from the client, executing the requested functions, and sending back the result. The RPC client is responsible for connecting to the server and making RPC calls to the server.

`model_server_timm.py` This file starts the RPC server and is responsible for loading a trained model from a local checkpoint (`.pth`), applying camera undistortion and gray-world color correction to each image, running inference using the same validation transforms used during training, and returning the predicted motion direction.

Dependencies (conda/mamba `arcs-su25` env): `torch`, `torchvision`, `timm`, `opencv-python`, `numpy`, `Pillow`, `tyro`, and the local `rpc` module.

```bash
cd /home/jetbot/arcs/OldenborgAutonomousNavigation/jetbot
mamba activate <environment name>
python model_server_timm.py --checkpoint <model path>.pth
```

`robot_client_timm.py` This file starts the RPC client and is responsible for capturing an image from the JetBot camera, applying camera undistortion and gray-world color correction in-memory, saving the corrected image locally, calling the model inference function with the image filename, and then calling the robot motion function with the motion direction.

Camera calibration constants (matrix and distortion coefficients for 224×224 images) are now hardcoded directly in both files, replacing the previous approach of computing calibration at runtime from chessboard images. You no longer need to copy calibration images from Box.

Dependencies (system Python `/usr/bin/python3`): `jetbot`, `opencv-python`, `numpy`, and the local `rpc` module.

```bash
# In a separate terminal
cd /home/jetbot/arcs/OldenborgAutonomousNavigation/jetbot
/usr/bin/python3 robot_client_timm.py
```

Optional arguments for `robot_client_timm.py`:

```bash
/usr/bin/python3 robot_client_timm.py \
  --output_dir ./saved_images \
  --max_actions 150 \
  --speed 2.5 \
  --duration 1.0 \
  --log inference_log.csv
```