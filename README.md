# FoundationPose Head Tracking + TB Rendering

This package contains the trimmed runtime needed for head pose tracking and TB OpenGL rendering.
All project assets are resolved relative to this folder, so the directory can be moved to another machine/path.

## Main Files

- `redis_track.py`: Orbbec camera sender. Publishes RGB/depth frames over ZMQ at `tcp://*:5555`.
- `online_head_tb_zmq.py`: ZMQ receiver, YOLO mask initialization, FoundationPose tracking, and TB rendering.
- `TB.py`: OpenGL TB renderer.
- `build_all_conda.sh`: Rebuilds C++/CUDA extensions for the active `foundationpose` conda environment.

## Start

Open two terminals.

Terminal 1, camera sender:

```bash
conda activate foundationpose
cd foundationpose-master
python redis_track.py
```

Terminal 2, tracker and renderer:

```bash
conda activate foundationpose
cd foundationpose-master
python online_head_tb_zmq.py
```


## Where To Get Results

All runtime results are produced in `online_head_tb_zmq.py` after each received frame is processed.

### Head Pose

The head pose is the variable `pose` in `online_head_tb_zmq.py`.
It is a `4x4` NumPy matrix representing the head/object pose in the camera coordinate system.
The translation column is in meters at this point:

```python
pose = est.register(...)
pose = est.track_one(...)
```

For downstream code, read or publish `pose` after the jitter filtering block:

```python
if is_pose_significant(pose, prev_pose, TRANSLATION_THRESHOLD, ROTATION_THRESHOLD):
    prev_pose = pose
else:
    pose = prev_pose
```

The TB renderer receives `output_pose`, which is copied from `pose` but converted from meters to millimeters:

```python
output_pose = np.copy(pose)
output_pose[:, 3] = 1000.0 * output_pose[:, 3]
output_pose[3, 3] = 1.0
```

Use `pose` if the next module expects meters. Use `output_pose` if the next module expects millimeters, like `TB.py`.

### 3D Box / Axis Overlay

The 2D visualization with the green 3D bounding box and axis is the variable `vis` in `online_head_tb_zmq.py`:

```python
vis = draw_pose_based_3d_box(reader.K, color_image, pose, bbox_corners, line_color=(0, 255, 0), linewidth=1)
vis, _ = draw_xyz_axis(color_image, ob_in_cam=pose, scale=0.1, K=reader.K, thickness=1, transparency=0, is_input_rgb=True)
cv2.imshow('AR', vis)
```

For downstream code, use `vis` directly before `cv2.imshow('AR', vis)`. It is an image array containing the camera image plus the rendered tracking box/axis.

### TB Rendering

TB rendering is handled by the child process started in `online_head_tb_zmq.py`:

```python
queue = multiprocessing.Queue(maxsize=30)
process_render = TB_processing(queue)
process_render.start()
```

Each frame is sent to TB here:

```python
queue.put_nowait((output_pose, output_color))
```

`TB.py` receives that tuple in its main render loop:

```python
pose_matrix, texture_image = self.queue.get()
```

The TB render result is displayed in the OpenGL window named `Head Tracking` inside `TB.py`. The rendered scene is not currently returned as an image array; it is drawn directly to the OpenGL window and presented by:

```python
glfw.swap_buffers(window)
```

If another module needs the TB render as an image, add the capture/export hook in `TB.py` immediately before `glfw.swap_buffers(window)`.

## Rebuild Extensions

If Python, PyTorch, CUDA, or the conda environment changes, rebuild once:

```bash
conda activate foundationpose
cd foundationpose-master
bash build_all_conda.sh
```

The checked-in `.so` files are built for Python 3.10 in the `foundationpose` environment.

## Required Runtime Data

Keep these folders with the project:

- `weights/`
- `demo_data/demo/`
- `demo_data/Data_1111_9/`
- `mycpp/build/mycpp.cpython-310-x86_64-linux-gnu.so`
- `bundlesdf/mycuda/*.cpython-310-x86_64-linux-gnu.so`

## Orbbec SDK

`redis_track.py` imports `pyorbbecsdk`. Prefer installing it into the conda environment.
If you keep a local SDK build inside the project, place it at `pyorbbecsdk/install/lib`, or set:

```bash
export PYORBBECSDK_LIB=pyorbbecsdk/install/lib
```
