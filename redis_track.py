# import zmq
# import numpy as np
# import time
# import cv2
# import os
# from pyorbbecsdk import Pipeline, FrameSet, Config, OBSensorType, OBFormat, OBAlignMode
# from orbbec import frame_to_bgr_image
# from ultralytics import YOLO
# import torch

# # ✅ 选择设备
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"🔍 设备检测: 运行在 {device}")

# model = YOLO("weights/best.pt").to(device)
# print("✅ YOLO 模型加载完成")

# # ✅ ZeroMQ 设置
# context = zmq.Context()
# socket = context.socket(zmq.PUB)
# socket.bind("tcp://*:5577")

# # ✅ Orbbec 相机初始化
# pipeline = Pipeline()
# config = Config()

# # ✅ 降低相机帧率，减少丢帧
# config.enable_stream(pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
#                      .get_video_stream_profile(640, 480, OBFormat.RGB, 60))
# config.enable_stream(pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
#                      .get_video_stream_profile(640, 480, OBFormat.Y16, 60))

# config.set_align_mode(OBAlignMode.SW_MODE)
# pipeline.enable_frame_sync()
# pipeline.start(config)

# print("✅ 相机已成功启动")

# frame_count = 0  # 帧计数器

# try:
#     while True:
#         frames: FrameSet = pipeline.wait_for_frames(1000)
#         if frames is None:
#             continue

#         color_frame = frames.get_color_frame()
#         depth_frame = frames.get_depth_frame()
#         if not color_frame or not depth_frame:
#             continue

#         frame_count += 1
#         color_image = frame_to_bgr_image(color_frame)  # ✅ 读取原始图像
#         depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((480, 640))

#         if frame_count % 3 == 0:  # ✅ 降低检测频率
#             yolo_input = color_image.copy()  # ✅ 复制一份给 YOLO，确保 `color_image` 不变
#             results = model.track(yolo_input, conf=0.2, max_det=1, persist=True)
#             # ✅ 只有检测到目标才发送帧数据
#             if results and len(results[0].boxes) > 0:
#                 timestamp = time.time()
#                 socket.send_pyobj({
#                     "timestamp": timestamp,
#                     "color": color_image,  # ✅ 发送原始 `color_image`
#                     "depth": depth_data
#                 })
#                 print(f"✅ 目标检测成功，已发送帧 {frame_count} at {timestamp:.6f}")

# except KeyboardInterrupt:
#     print("Stopping...")

# finally:
#     try:
#         pipeline.stop()
#         print("✅ 相机已停止")
#     except Exception as e:
#         print(f"⚠️ 停止相机失败: {e}")



import zmq
import numpy as np
import time
from pyorbbecsdk import Pipeline, FrameSet, Config, OBSensorType, OBFormat, OBAlignMode, DepthFrame, ColorFrame
from orbbec import frame_to_bgr_image



# ZeroMQ 设置
context = zmq.Context()
socket = context.socket(zmq.PUB)  
socket.bind("tcp://*:5555")  

# Orbbec 相机初始化
pipeline = Pipeline()
config = Config()
config.enable_stream(pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
                     .get_video_stream_profile(640, 480, OBFormat.RGB, 60))
config.enable_stream(pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
                     .get_video_stream_profile(640, 480, OBFormat.Y16, 60))
config.set_align_mode(OBAlignMode.SW_MODE)
pipeline.enable_frame_sync()
pipeline.start(config)

print("🚀 ZeroMQ Streaming Server Started at 640x480 @ 60Hz...")

frame_count = 0  # 帧计数器

try:
    
    while True:
        frames: FrameSet = pipeline.wait_for_frames(1000)
        if frames is None:
            continue

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

    

        frame_count += 1

        # **每 2 帧发布 1 帧**
        if frame_count % 3 != 0:
            #print(f"⏩ Skipping frame {frame_count}")
            continue

        # 获取图像数据
        color_image = frame_to_bgr_image(color_frame)
        # 打印彩色图像信息
        print(f"Color image - shape: {color_image.shape}, dtype: {color_image.dtype}, "
              f"min: {np.min(color_image)}, max: {np.max(color_image)}, "
              f"mean: {np.mean(color_image):.2f}")


        #print(f"Depth frame info: {dir(depth_frame)}")  # 打印depth_frame所有属性和方法
        print(f"Depth frame width: {depth_frame.get_width()}, height: {depth_frame.get_height()}")  # 打印宽高
        #print(f"Depth frame format: {depth_frame.get_format()}")  # 打印数据格式

        # 获取深度帧数据
        data = depth_frame.get_data()

        #data = depth_frame.get_data()
        expected_size = 640 * 480 * 2  # 2 bytes per uint16
        #print(f"Data len: {len(data)}, expected_size:{expected_size}")  #

        if len(data) != expected_size:
            raise ValueError(f"Expected {expected_size} bytes, got {len(data)} bytes")


        # 打印数据类型
        print(f"Data type: {type(data)}, Is C-contiguous: {data.flags['C_CONTIGUOUS']}, data len: {len(data)}")

        # 确保数组是C连续的
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)

        # 确保是uint16类型
        if data.dtype != np.uint16:
            data = data.view(np.uint16)  # 将字节数据解释为uint16

        # 确保数组形状正确
        if data.shape != (480, 640):
            data = data.reshape((480, 640))

        print(f"Data type: {type(data)}, Is C-contiguous: {data.flags['C_CONTIGUOUS']}, data len: {data.shape}")

        # 打印reshape后的数据信息
        print(f"After reshape - shape: {data.shape}, dtype: {data.dtype}, min: {np.min(data)}, max: {np.max(data)}, mean: {np.mean(data):.2f}")
        

        depth_data = data
        #depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((480, 640))

        timestamp = time.time()

        # 发送数据
        socket.send_pyobj({
            "timestamp": timestamp,
            "color": color_image,
            "depth": depth_data
        })

        print(f"✅ Published frame {frame_count} at {timestamp:.6f}")

except KeyboardInterrupt:
    print("Stopping...")
    pipeline.stop()








