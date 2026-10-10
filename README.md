# sim2real-robotics

Code for **"Real, Simulated, or Mixed? Training-Data Composition, Architecture, and Augmentation for Visual Navigation"**

Ruben Millan Fabian '27, Tung Nguyen '28, Stefanie Nguyen '29, Professor Anthony Clark
ARCS Lab, Department of Computer Science, Pomona College (SURP 2026)

Poster: [`poster.pdf`](poster.pdf)

A JetBot follows a taped line around a tabletop track by choosing one of three discrete actions (`left`, `forward`, `right`) from a single monocular RGB frame. We train image classifiers on real frames, simulated frames (Unreal Engine 5), and hybrids of the two, deploy them on the physical robot, and score the driven trajectory against an OptiTrack motion-capture ground truth at millimetre resolution.

Headline result: simulation-only never completes a lap, real-only overfits, and a hybrid of the full simulated set plus ~90% of the real pool is the only regime that transfers both reliably and accurately (16.0 mm mean absolute error, 100% completion).

## Repository layout

| File | Purpose |
|---|---|
| `color_camera_calibration.py` | Offline dataset preprocessing: undistort real frames with the calibrated intrinsics and apply gray-world colour correction. |
| `train_timm.py` | Train a `timm` classifier (ResNet18, MobileNetV3, ConvNeXt, ...) on an `ImageFolder` dataset with `left/forward/right` classes. |
| `infer_timm.py` | Offline inference: predict a single image or evaluate accuracy on a folder. |
| `rpc.py` | Minimal JSON-over-socket RPC server and client used to bridge the two Python environments on the JetBot. |
| `model_server_timm.py` | RPC server that loads a checkpoint, applies the same corrections as the dataset script, and returns an action for each image. |
| `robot_client_timm.py` | RPC client that runs on the JetBot: capture, ask the model server for an action, drive, repeat. |
| `robot_client.py` | Earlier version of the client that computed camera calibration from chessboard images at runtime. Kept for reference. |
| `trajectory_evaluator.py` | Score an OptiTrack (Motive) recording of a run against a reference centerline and plot the result. |
| `data_compressed.zip` | Real labelled frames, organised per track and per class (not tracked in git). |
| `poster.pdf` | Research poster with methods, results, and figures. |

## Pipeline

```
real frames ──► color_camera_calibration.py ──┐
                                              ├──► train_timm.py ──► best_model.pth
simulated frames (UE5, no correction) ────────┘                          │
                                                                         ▼
JetBot: robot_client_timm.py ◄── RPC ──► model_server_timm.py (loads checkpoint)
         │
         ▼
OptiTrack recording ──► trajectory_evaluator.py ──► MAE (mm), % ≤ 10 mm, completion
```

## 1. Dataset

Real frames live in `data_compressed.zip`, one folder per track and one subfolder per action:

```
data/
  track1-1/{forward,left,right}/
  track1-2/{forward,left,right}/
  track3-1/{forward,left,right}/
  track4/{forward_track4,left_track4,right_track4}/
  ...
```

Tracks 1, 3 and 4 are used for training; track 2 is held out for evaluation. Before training, run the correction script over the raw frames. Edit `INPUT_ROOT` and `OUTPUT_ROOT` at the top of the file, then:

```bash
python color_camera_calibration.py
```

The script undistorts each 224×224 frame with the hardcoded camera matrix and distortion coefficients (`alpha=0`, so no black borders) and applies gray-world colour correction to remove the pink cast from the sensor and lab lighting. Simulated frames are rendered with a pinhole camera and need no correction.

## 2. Training

`train_timm.py` expects either a flat `ImageFolder` with `left/`, `forward/`, `right/` subfolders (a validation split is carved out automatically) or a directory with `train/` and `val/` subfolders. It refuses to run unless exactly those three classes are present.

```bash
python train_timm.py \
  --data-dir datasets/hybrid_90 \
  --output-dir output/line_following \
  --model resnet18 \
  --pretrained \
  --epochs 25 \
  --batch-size 32 \
  --lr 3e-4 \
  --weight-decay 1e-4
```

Training uses AdamW with cosine annealing, random resized cropping (scale 0.7 to 1.0) and colour jitter as online augmentation, and selects the checkpoint with peak validation accuracy. Each run writes a timestamped directory containing `best_model.pth`, the CLI args as JSON, the class mapping, and a per-epoch `training_log.csv` with a confusion matrix.

Useful flags: `--patience N` for early stopping, `--resume path.pth` to continue a run, `--device cpu` to force CPU.

Offline inference and accuracy on a held-out folder:

```bash
python infer_timm.py --checkpoint output/.../best_model.pth --image frame.jpg
python infer_timm.py --checkpoint output/.../best_model.pth --data-dir datasets/track2/val
```

## 3. Deployment on the JetBot

The JetBot SDK runs on an older Python that is incompatible with current PyTorch and `timm`, so deployment uses two processes that talk over a local RPC socket:

- **Model server** (conda/mamba environment with `torch`, `torchvision`, `timm`, `opencv-python`, `numpy`, `Pillow`, `tyro`): loads the checkpoint, applies undistortion and gray-world correction, runs the same validation transform used in training, and returns `left`, `forward` or `right`.
- **Robot client** (system `/usr/bin/python3` with `jetbot`, `opencv-python`, `numpy`): captures a frame, saves it to disk, calls `model_run` on the server with the filename, executes the returned action, and loops.

The control loop is stop-and-go. The robot issues a motion command, stops, lets the camera settle, captures, infers, and repeats, so per-step latency does not affect control stability.

Terminal 1, model server:

```bash
mamba activate <env>
python model_server_timm.py --checkpoint <path>/best_model.pth
```

Terminal 2, robot client:

```bash
/usr/bin/python3 robot_client_timm.py \
  --output_dir ./saved_images \
  --max_actions 150 \
  --speed 2.5 \
  --duration 0.2 \
  --log inference_log.csv
```

Press a key in the client terminal to stop the robot. Camera calibration constants for 224×224 frames are hardcoded in both `model_server_timm.py` and `color_camera_calibration.py`, so the robot and the dataset see an identical preprocessing pipeline.

## 4. Trajectory evaluation

Runs are recorded with an OptiTrack camera tracking a four-marker rigid body on the JetBot. Motive exports X and Z per frame as CSV. A separate recording of the marker square slid along the track centerline provides the reference path.

```bash
python trajectory_evaluator.py \
  --track track2_centerline.csv \
  --robot run_m14_trial1.csv \
  --robot_body Robot \
  --output run_m14_trial1.png
```

The evaluator drops motion-capture dropouts and tracking teleports, trims the robot's lead-in and any parked tail, rigidly aligns the robot path to the centerline by minimising mean nearest-neighbour distance, downsamples by `--stride` (default 60), and reports MAE in millimetres, the share of frames within 10 mm, and a path overlay plot. Use `--start_anchor` to align on the start point instead of a whole-path fit, `--no_align` to skip alignment, and `--light` for a light plot theme. If `robot.csv` is missing, the script generates mock data so the pipeline can be exercised without a recording.

## Requirements

Training and evaluation (any machine with a GPU recommended):

```
torch torchvision timm opencv-python numpy Pillow tyro
pandas scipy matplotlib seaborn   # trajectory_evaluator.py only
```

JetBot client additionally requires the `jetbot` package in the system Python.

## Acknowledgments

Thanks to Francisco Morales Puente and Tommy Ryan. This project was made possible by funding from Mr. & Mrs. Richard N. Frank. Training ran on the Pomona HPC cluster.
