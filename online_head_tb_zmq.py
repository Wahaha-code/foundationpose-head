import argparse
import logging
import os
import threading
import numpy as np
import trimesh
import cv2
import imageio
import multiprocessing
import queue as queue_module
import zmq
from ultralytics import YOLO
try:
    from pynput import keyboard
except ImportError:
    keyboard = None
from estimater import *
from datareader_online import *
from TB import TB_processing
# **全局变量**
restart_flag = False  # 监听 `t` 键
prev_pose = np.eye(4)  # **初始化上一帧姿态为单位矩阵**

# **抖动阈值**
TRANSLATION_THRESHOLD = 0.02  # **平移抖动阈值**
ROTATION_THRESHOLD = 0.03    # **旋转抖动阈值**

def recv_latest(socket):
    """ 丢弃所有旧消息，只获取最新的一条 """
    latest_msg = None  # 先初始化变量
    while socket.poll(0):  # 检查是否有旧帧
        latest_msg = socket.recv_pyobj()

    if latest_msg is None:  # 如果没有旧消息，就直接阻塞等待一帧
        latest_msg = socket.recv_pyobj()

    return latest_msg


# ZeroMQ 初始化，和 online.py 一样从发布端拿最新一帧 color/depth
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5555")
socket.setsockopt_string(zmq.SUBSCRIBE, "")


# **1️⃣ 监听键盘输入**
def keyboard_listener():
    """监听 `t` 键，重置 `i` 变量"""
    if keyboard is None:
        print("⚠️ pynput 不可用，跳过键盘监听")
        return
    def on_press(key):
        global restart_flag
        try:
            if key.char == 't':
                print("\n检测到 `t`，重置 i = 0")
                restart_flag = True
        except AttributeError:
            pass  # 忽略特殊按键

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()  # **非阻塞监听**

# **2️⃣ 判断姿态变化是否显著**
def is_pose_significant(new_pose, prev_pose, translation_threshold, rotation_threshold):
    """
    判断姿态变化是否显著。
    :param new_pose: 当前帧姿态矩阵
    :param prev_pose: 上一帧姿态矩阵
    :param translation_threshold: 平移变化阈值
    :param rotation_threshold: 旋转变化阈值
    :return: 是否有显著变化
    """
    if prev_pose is None:
        return True  # **第一帧无参考，直接更新**

    # **计算平移变化量**
    translation_diff = np.linalg.norm(new_pose[:3, 3] - prev_pose[:3, 3])

    # **计算旋转变化量**
    rotation_diff = np.linalg.norm(new_pose[:3, :3] - prev_pose[:3, :3])

    # **判断是否超过阈值**
    return translation_diff > translation_threshold or rotation_diff > rotation_threshold


if __name__ == '__main__':
    queue = multiprocessing.Queue(maxsize=30)
    process_render = TB_processing(queue)
    process_render.start()

    # **解析命令行参数**
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/demo/mesh/head_few.obj')
    parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/demo')
    parser.add_argument('--rgb_dir', type=str, default=f'{code_dir}/demo_data/demo/rgb')
    parser.add_argument('--depth_dir', type=str, default=f'{code_dir}/demo_data/demo/depth')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    args = parser.parse_args()

    # **设置日志格式**
    set_logging_format()
    set_seed(0)

    # **加载 Mesh**
    mesh = trimesh.load(args.mesh_file)
    debug_dir = args.debug_dir
    os.makedirs(debug_dir, exist_ok=True)

    # **计算 AABB**
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
    bbox_corners = np.array([
        [mesh.bounds[0, 0], mesh.bounds[0, 1], mesh.bounds[0, 2]],
        [mesh.bounds[1, 0], mesh.bounds[0, 1], mesh.bounds[0, 2]],
        [mesh.bounds[1, 0], mesh.bounds[1, 1], mesh.bounds[0, 2]],
        [mesh.bounds[0, 0], mesh.bounds[1, 1], mesh.bounds[0, 2]],
        [mesh.bounds[0, 0], mesh.bounds[0, 1], mesh.bounds[1, 2]],
        [mesh.bounds[1, 0], mesh.bounds[0, 1], mesh.bounds[1, 2]],
        [mesh.bounds[1, 0], mesh.bounds[1, 1], mesh.bounds[1, 2]],
        [mesh.bounds[0, 0], mesh.bounds[1, 1], mesh.bounds[1, 2]],
    ])

    # **初始化模型**
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    model = YOLO(f'{code_dir}/weights/best.pt')
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, glctx=glctx)
    logging.info("✅ Estimator 初始化完成")

    os.makedirs(args.rgb_dir, exist_ok=True)
    os.makedirs(args.depth_dir, exist_ok=True)
    reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

    # **启动键盘监听线程**
    threading.Thread(target=keyboard_listener, daemon=True).start()

    # **初始化**
    i = 0
    try:
        cv2.namedWindow('AR', cv2.WINDOW_NORMAL)  # ✅ 优化：只执行一次

        while True:
            if restart_flag:
                i = 0
                restart_flag = False
                print("i 已重置为 0")

            # **从 ZeroMQ 获取数据**
            data = recv_latest(socket)  # **始终获取最新帧**
            color_image = data["color"]  # ✅ 直接使用
            depth_image = data["depth"] / 1000.0  # ✅ 单位转换为米


            # **i = 0 时，初始化 `register()`**
            if i == 0:
                results = model(color_image)

                if results:  # ✅ **防止 `results=[]` 时报错**
                    result = results[0]
                    if hasattr(result, "masks") and result.masks is not None:
                        mask_raw = np.sum([mask.cpu().data.numpy().transpose(1, 2, 0) for mask in result.masks], axis=0)
                        mask_scaled = (mask_raw * 255).astype(np.uint8)

                        # **调整 mask**
                        mask = cv2.resize(mask_scaled.astype(np.uint8), (640, 480), interpolation=cv2.INTER_NEAREST).astype(bool)

                        # **初始化 `pose`**
                        pose = est.register(K=reader.K, rgb=color_image, depth=depth_image, ob_mask=mask, iteration=args.est_refine_iter)
                        prev_pose = pose  # **存储初始化 pose**
                        print("✅ `register()` 初始化成功！")
                    else:
                        print("⚠️ 未找到 Mask，跳过 `register()`")
                        continue
                else:
                    print("⚠️ `model(color_image)` 结果为空，跳过 `register()`")
                    continue
            else:
                # **后续帧 `track_one()`**
                try:
                    pose = est.track_one(rgb=color_image, depth=depth_image, K=reader.K, iteration=args.track_refine_iter)
                    # 1️⃣ 计算 mesh 在当前 pose 下的 3D 坐标
                    mesh_vertices_homo = np.hstack((mesh.vertices, np.ones((mesh.vertices.shape[0], 1))))  # 变成齐次坐标
                    mesh_in_cam = (pose @ mesh_vertices_homo.T).T  # 将 mesh 变换到相机坐标系

                    # 2️⃣ 计算投影点
                    projected_points = (reader.K @ mesh_in_cam[:, :3].T).T  # 投影到 2D
                    projected_points[:, 0] /= projected_points[:, 2]  # 归一化 x
                    projected_points[:, 1] /= projected_points[:, 2]  # 归一化 y

                    # 3️⃣ 过滤超出图像范围的点
                    h, w = depth_image.shape
                    valid_mask = (
                        (projected_points[:, 0] >= 0) & (projected_points[:, 0] < w) &
                        (projected_points[:, 1] >= 0) & (projected_points[:, 1] < h) &
                        (projected_points[:, 2] > 0)
                    )

                    valid_points = projected_points[valid_mask]
                    valid_depths = mesh_in_cam[:, 2][valid_mask]  # 预测的深度

                    # 4️⃣ 获取真实深度图中的深度
                    real_depths = depth_image[valid_points[:, 1].astype(int), valid_points[:, 0].astype(int)]

                    # 5️⃣ 计算误差
                    depth_error = np.abs(real_depths - valid_depths)

                    # 6️⃣ 判断追踪是否失败
                    threshold = 0.065  # 设定一个深度误差阈值（单位：米）
                    failure_ratio = (depth_error > threshold).sum() / len(depth_error)

                    if failure_ratio > 0.65:  # 超过 50% 的点误差太大
                        print("⚠️ 追踪失败，重置 i = 0")
                        restart_flag = True  # 触发重置
                        continue  # 跳过当前帧
                except RuntimeError:
                    print("⚠️ `track_one()` 失败，跳过该帧")
                    continue

            # **抖动优化**
            if is_pose_significant(pose, prev_pose, TRANSLATION_THRESHOLD, ROTATION_THRESHOLD):
                prev_pose = pose
            else:
                pose = prev_pose
            try:
                # **优化 `pose` 数据格式**
                output_pose = np.copy(pose)  # ✅ 确保 `pose` 传输前不会被修改
                output_pose[:, 3] = 1000.0 * output_pose[:, 3]  # ✅ 位置转换（m -> mm）
                output_pose[3, 3] = 1.0  # ✅ 保持齐次坐标正确

                # **优化 `color_image` 传输格式**
                output_color = color_image.copy()  # ✅ 确保不会影响原始图像

                if process_render.is_alive():
                    # **放入队列**：TB 渲染慢或已退出时不要卡住跟踪主循环
                    try:
                        queue.put_nowait((output_pose, output_color))  # ✅ 直接传递 `numpy` 数组
                        print("✅ 成功放入队列")
                    except queue_module.Full:
                        print("⚠️ TB 队列已满，丢弃当前帧")
                else:
                    print("❌ TB_processing 已退出，跳过队列输出；请检查 TB.py 依赖的模型/数据文件")

            except Exception as e:
                print(f"❌ `queue.put()` 失败: {e}")  # **错误信息**

            # **可视化**
            vis = draw_pose_based_3d_box(reader.K, color_image, pose, bbox_corners, line_color=(0, 255, 0), linewidth=1)
            vis, _ = draw_xyz_axis(color_image, ob_in_cam=pose, scale=0.1, K=reader.K, thickness=1, transparency=0, is_input_rgb=True)
            cv2.imshow('AR', vis)
            cv2.waitKey(1)

            i += 1  # ✅ **更新帧数**

    finally:
        cv2.destroyAllWindows()
