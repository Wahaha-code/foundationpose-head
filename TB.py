import numpy as np
import multiprocessing
import cv2
import OpenGL.GL as gl
from OpenGL.GL import *
from OpenGL.GL import shaders
# from OpenGL.GLUT import *
import glfw
import pywavefront
import trimesh
import ctypes
import glm

import time
from pathlib import Path

# resolution = [1280,720]
resolution = [640,480]
PROJECT_ROOT = Path(__file__).resolve().parent

def relpath(path):
    return str(PROJECT_ROOT / path)

def framebuffer_size_callback( window, width, height):
        ratio = resolution[0] / resolution[1]
        w = int(height * ratio)
        w_left = int((width - w) / 2)
        glViewport(w_left, 0, w, height)  # 更新视口大小


class TB_processing(multiprocessing.Process):
    def __init__(self, queue):
        multiprocessing.Process.__init__(self)
        self.queue = queue


        self.xh_pose = None

    # def store_pose(self, data):
    #     """Store the pose and coordinates received from Redis"""
    #     # with self.data_lock:
    #         # Extract pose and convert to numpy array
    #     self.xh_pose = np.array(data.get('pose'))  # Convert list back to numpy array
        
    #     # 确保 xh_pose 是一个二维数组，且列数足够
    #     if self.xh_pose.ndim == 2 and self.xh_pose.shape[1] > 0:
    #         # 对最后一列的前三行乘以 1000
    #         self.xh_pose[:3, -1] *= 1000

    #     print("selfpose after scaling:", self.xh_pose)


    def run(self):
        self.data_path = relpath("demo_data/Data_1111_9") + "/"
        Head_file = relpath("demo_data/demo/mesh/head_1121_compressd_1000.obj")  # 替换为你的模型文件路径
        Skull_file = relpath("demo_data/Data_1111_9/mesh/SkullModel_LPS.obj")  # 替换为你的模型文件路径
        Brain_file = relpath("demo_data/Data_1111_9/mesh/BrainModel_LPS.obj")  # 替换为你的模型文件路径
        Ventricle_file = relpath("demo_data/Data_1111_9/mesh/VentricleModel_LPS.obj")  # 替换为你的模型文件路径
        # image_file = "demo_data/Data_1111_9/rgb/0000.png"  # 替换为你的 JPEG 文件路径
        # image = cv2.imread(image_file)
        self.main(Head_file, Skull_file, Brain_file, Ventricle_file)


    def load_obj(self, filename):
        mesh = trimesh.load_mesh(filename)
        vertices = mesh.vertices  # 获取顶点数据 (n x 3)
        normals = -mesh.vertex_normals  # 获取法线数据 (n x 3)
        faces = mesh.faces
        faces = np.array(faces)


        # 将顶点和法线合并为一个数组
        combined_data = np.hstack((vertices, normals))  # (n x 6) 数组
        
        # combined_data = combined_data.reshape(-1, 3)
        # print(combined_data[0])
        # print(combined_data[1])

        return combined_data.astype(np.float32), faces.astype(np.int32)
        # return vertices.astype(np.float32), faces.astype(np.int32)

    def get_bounds_center(self, filename, scale=1.0):
        mesh = trimesh.load_mesh(filename)
        return mesh.bounds.mean(axis=0) * scale

    # def load_obj(filename):
    #     # 使用 Open3D 加载 OBJ 文件
    #     mesh = o3d.io.read_triangle_mesh(filename)

    #     # 检查是否成功加载
    #     if not mesh.is_empty():
    #         vertices = np.asarray(mesh.vertices)  # 获取顶点数据 (n x 3)
    #         normals = np.asarray(mesh.vertex_normals)  # 获取法线数据 (n x 3)
    #         faces = np.asarray(mesh.triangles)  # 获取面信息 (m x 3)
    #     else:
    #         raise ValueError("Failed to load the mesh from the file.")

    #     print(faces)
    #     print(len(faces))

    #     # 将顶点和法线合并为一个数组
    #     combined_data = np.hstack((vertices, normals))  # (n x 6) 数组
    #     return combined_data.astype(np.float32), faces.astype(np.int32)


    # def load_obj(filename):
    #     # 加载 OBJ 文件
    #     scene = pywavefront.Wavefront(filename, collect_faces=True)

    #     # 提取顶点，法线和面信息
    #     vertices = np.array(scene.vertices, dtype=np.float32)  # (n x 3)
        
    #     # 使用 PyWavefront 时，法线可能在场景中并不总是可用，需手动处理
    #     normals = np.array(scene.normals, dtype=np.float32) if hasattr(scene, 'normals') else np.zeros_like(vertices)  # (n x 3)

    #     # 获取面信息
    #     faces = []
    #     for mesh in scene.meshes.values():
    #         faces.extend(mesh.faces)  # 将所有面的索引合并
        
    #     faces = np.array(faces, dtype=np.int32)  # 转换为 (m x 3) array

    #     print(faces)
    #     print(len(faces))

    #     # 将顶点和法线合并为一个数组
    #     combined_data = np.hstack((vertices, normals))  # (n x 6) 数组
    #     return combined_data.astype(np.float32), faces.astype(np.int32)


    # 初始化窗口
    def init_window(self, width, height, title):
        if not glfw.init():
            return None
        window = glfw.create_window(width, height, title, None, None)
        glfw.make_context_current(window)
        glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)

        return window

    # 创建视图矩阵
    def look_at(self, eye, center, up):
        f = (center - eye) / np.linalg.norm(center - eye)
        r = np.cross(up, f)
        u = np.cross(f, r)

        view_matrix = np.array([
            [r[0], u[0], -f[0], 0],
            [r[1], u[1], -f[1], 0],
            [r[2], u[2], -f[2], 0],
            [-np.dot(r, eye), -np.dot(u, eye), np.dot(f, eye), 1]
        ])

        return view_matrix


    # 加载纹理
    def load_texture(self, image):
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)

        # 设置纹理参数
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        # 使用 OpenCV 加载图像
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape

        # 生成纹理
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image)
        glGenerateMipmap(GL_TEXTURE_2D)

        glBindTexture(GL_TEXTURE_2D, 0)

        return texture_id


    # 编译着色器
    def compile_shaders(self, vertex_code, fragment_code):
        vertex_shader = shaders.compileShader(vertex_code, gl.GL_VERTEX_SHADER)
        fragment_shader = shaders.compileShader(fragment_code, gl.GL_FRAGMENT_SHADER)
        shader_program = shaders.compileProgram(vertex_shader, fragment_shader)
        return shader_program

    # 设置投影和视图矩阵
    def set_matrices(self, shader_program, pose_matrix, cam_K):
        project_matrix = np.identity(4, dtype='float32')
        # view = np.identity(4, dtype='float32')
        # pose_matrix = np.identity(4, dtype='float32')

        # 填充投影矩阵（例如，透视投影）
        w = resolution[0]
        h = resolution[1]
        fx = cam_K[0,0]
        fy = cam_K[1,1]
        cx = cam_K[0,2]
        cy = cam_K[1,2]
        znear = 1
        zfar = 2000.0
        project_matrix[0,0] = 2 * fx / w
        project_matrix[1,0] = 0.0
        project_matrix[2,0] = 0.0
        project_matrix[3,0] = 0.0

        project_matrix[0,1] = 0.0
        project_matrix[1,1] = 2 * fy / h
        project_matrix[2,1] = 0.0
        project_matrix[3,1] = 0.0

        project_matrix[0,2] = (w - 2.0 * cx) / w
        project_matrix[1,2] = -(h - 2.0 * cy) / h
        project_matrix[2,2] = (-zfar - znear) / (zfar - znear)
        project_matrix[3,2] = -1.0

        project_matrix[0,3] = 0.0
        project_matrix[1,3] = 0.0
        project_matrix[2,3] = -2.0 * zfar * znear / (zfar - znear)
        project_matrix[3,3] = 0.0

        # eye = np.array([0, 0, 0])
        # center = np.array([0, 0, 1])
        # up = np.array([0, -1, 0])
        # view_matrix = look_at(eye, center, up)
        eye = glm.vec3(0.0, 0.0, 0.0)     # 相机位置
        center = glm.vec3(0.0, 0.0, 1.0)  # 观察点
        up = glm.vec3(0.0, -1.0, 0.0)      # 上向量
        view_matrix = glm.lookAt(eye, center, up)
        view_matrix_array = glm.value_ptr(view_matrix)

        # print(gl.glGetUniformLocation(shader_program, "u_PerspectiveProjMatrix"))
        # print(gl.glGetUniformLocation(shader_program, "u_ViewMatrix"))
        # print(gl.glGetUniformLocation(shader_program, "u_ModelMatrix"))
        # 将矩阵传递给着色器
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(shader_program, "u_PerspectiveProjMatrix"), 1, gl.GL_FALSE, project_matrix.T)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(shader_program, "u_ViewMatrix"), 1, gl.GL_FALSE, view_matrix_array)
        # gl.glUniformMatrix4fv(gl.glGetUniformLocation(shader_program, "u_ViewMatrix"), 1, gl.GL_FALSE, view_matrix.T)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(shader_program, "u_ModelMatrix"), 1, gl.GL_FALSE, pose_matrix.T)


    def render_background(self, texture_id):
        glClear(GL_COLOR_BUFFER_BIT)

        glBindTexture(GL_TEXTURE_2D, texture_id)

        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, -1.0, 1.0)
        glTexCoord2f(1.0, 0.0); glVertex3f( 1.0, -1.0, 1.0)
        glTexCoord2f(1.0, 1.0); glVertex3f( 1.0,  1.0, 1.0)
        glTexCoord2f(0.0, 1.0); glVertex3f(-1.0,  1.0, 1.0)
        glEnd()

    # 主程序
    def main(self, Head_file, Skull_file, Brain_file, Ventricle_file):


        # listener_thread = threading.Thread(target=self.pose_listener)
        # listener_thread.daemon = True  # Allows the program to exit even if thread is running
        # listener_thread.start()



        show_head = True

        show_axes = False

        ambient = 0.2
        diffuse = 1.0
        specular = 0.0
        specularPower = 32.0

        m_LightOri = np.array([0.0, -0.5, -0.866]);

        m_AspectRatio = resolution[0]/resolution[1]

        # Head = trimesh.load(Head_file)
        # Skull = trimesh.load(Skull_file)
        # Brain = trimesh.load(Brain_file)
        # Ventricle = trimesh.load(Ventricle_file)
        # texture_image = image.copy()

        cam_K = np.loadtxt(relpath("demo_data/demo/cam_K_ob.txt"))


        # texture_image = cv2.resize(texture_image, None, fx=3, fy=3, interpolation=cv2.INTER_AREA)
        # height, width, _ = texture_image.shape
        height, width = 1440, 1920

        while(1):
                if not self.queue.empty(): break
        window = self.init_window(width, height, "Head Tracking")

        # 编译着色器
        vertex_shader_source = open(relpath("shaders/Texture.vert")).read()  # 替换为您的顶点着色器路径
        fragment_shader_source = open(relpath("shaders/Texture.frag")).read()  # 替换为您的片段着色器路径

        shader_program_Background = self.compile_shaders(vertex_shader_source, fragment_shader_source)

        # texture_image = cv2.flip(texture_image,0)
        # texture_id = load_texture(texture_image)



        # 设置坐标轴的起始和结束位置
        axes_vertices = np.array([
            # X轴 (红色)
            0.0, 0.0, 0.0,
            20.00, 0.0, 0.0,

            # Y轴 (绿色)
            0.0, 0.0, 0.0,
            0.0, 20.0, 0.0,

            # Z轴 (蓝色)
            0.0, 0.0, 0.0,
            0.0, 0.0, 20.0
        ], dtype='float32')

        xh_end = np.array([
            0.0, 154.5, 0.0,
            0.0, 154.5, 0.0,
            0.0, 154.5, 0.0,
            0.0, 154.5, 0.0,
            0.0, 154.5, 0.0,
            0.0, 154.5, 0.0,], dtype='float32')

        # axes_vertices_transformed = axes_vertices + xh_end

        axes_vertices_transformed = axes_vertices

        colors = np.array([
            # X轴颜色 (红色)
            1.0, 0.0, 0.0, 1.0,
            1.0, 0.0, 0.0, 1.0,
            
            # Y轴颜色 (绿色)
            0.0, 1.0, 0.0, 1.0,
            0.0, 1.0, 0.0, 1.0,
            
            # Z轴颜色 (蓝色)
            0.0, 0.0, 1.0, 1.0,
            0.0, 0.0, 1.0, 1.0,
        ], dtype='float32')


        vertexAttribArray = np.array([
                    -1.0,  1.0, 1.0,   0.0, 1.0,  # TL
                    -1.0, -1.0, 1.0,   0.0, 0.0,  # BL
                    1.0, -1.0, 1.0,   1.0, 0.0,  # BR
                    1.0,  1.0, 1.0,   1.0, 1.0   # TR
                ], dtype=np.float32)

        indices_back = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

        # Create VAO
        vao_back = glGenVertexArrays(1)
        glBindVertexArray(vao_back)

        # Create VBO
        vbo_back = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_back)
        glBufferData(GL_ARRAY_BUFFER, vertexAttribArray.nbytes, vertexAttribArray, GL_STATIC_DRAW)

        # Vertex attribute pointers
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * vertexAttribArray.itemsize, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * vertexAttribArray.itemsize, ctypes.c_void_p(3 * vertexAttribArray.itemsize))
        glEnableVertexAttribArray(1)

        # Create EBO
        ebo_back = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_back)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_back.nbytes, indices_back, GL_STATIC_DRAW)

        ###################################################################################################

        # 设置顶点数据
        vao_axes = gl.glGenVertexArrays(1)
        vbo_axes = gl.glGenBuffers(1)
        cbo_axes = gl.glGenBuffers(1)

        gl.glBindVertexArray(vao_axes)

        # 顶点缓冲
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_axes)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, axes_vertices_transformed.nbytes, axes_vertices_transformed, gl.GL_STATIC_DRAW)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, False, 0, None)
        gl.glEnableVertexAttribArray(0)

        # 颜色缓冲
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, cbo_axes)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, colors.nbytes, colors, gl.GL_STATIC_DRAW)
        gl.glVertexAttribPointer(1, 4, gl.GL_FLOAT, False, 0, None)
        gl.glEnableVertexAttribArray(1)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

        ###################################################################################################
        
        vertices_head, indices_head = self.load_obj(Head_file)  # 替换为您的模型路径

        tracked_head_file = relpath("demo_data/demo/mesh/head_few.obj")
        tracked_head_center = self.get_bounds_center(tracked_head_file, scale=1000.0)
        render_head_center = self.get_bounds_center(Head_file)
        head_model_to_track = np.identity(4)
        head_model_to_track[:3, 3] = tracked_head_center - render_head_center
        print("head_model_to_track offset:", head_model_to_track[:3, 3])

        # 创建 VAO
        vao_head = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(vao_head)

        # 创建 VBO 以存储顶点数据
        vbo_head = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_head)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices_head.nbytes, vertices_head, gl.GL_STATIC_DRAW)

        # 创建 EBO 以存储索引数据
        ebo_head = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo_head)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices_head.nbytes, indices_head, gl.GL_STATIC_DRAW)

        # 设置顶点属性指针
        position_loc = 0

        gl.glVertexAttribPointer(position_loc, 3, gl.GL_FLOAT, False, 6 * ctypes.sizeof(ctypes.c_float), None)
        gl.glEnableVertexAttribArray(position_loc)

        normal_loc = 1
        gl.glVertexAttribPointer(normal_loc, 3, gl.GL_FLOAT, False, 6 * ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(3 * ctypes.sizeof(ctypes.c_float)))
        gl.glEnableVertexAttribArray(normal_loc)

        # 解绑 VAO (可选)
        gl.glBindVertexArray(0)

    ###########################################################################################################
        
        vertices_skull, indices_skull = self.load_obj(Skull_file)  # 替换为您的模型路径

        # 创建 VAO
        vao_skull = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(vao_skull)

        # 创建 VBO 以存储顶点数据
        vbo_skull = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_skull)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices_skull.nbytes, vertices_skull, gl.GL_STATIC_DRAW)

        # 创建 EBO 以存储索引数据
        ebo_skull = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo_skull)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices_skull.nbytes, indices_skull, gl.GL_STATIC_DRAW)

        # 设置顶点属性指针
        position_loc = 0

        gl.glVertexAttribPointer(position_loc, 3, gl.GL_FLOAT, False, 6 * ctypes.sizeof(ctypes.c_float), None)
        gl.glEnableVertexAttribArray(position_loc)

        normal_loc = 1
        gl.glVertexAttribPointer(normal_loc, 3, gl.GL_FLOAT, False, 6 * ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(3 * ctypes.sizeof(ctypes.c_float)))
        gl.glEnableVertexAttribArray(normal_loc)

    ###########################################################################################################

        vertices_ventricle, indices_ventricle = self.load_obj(Ventricle_file)  # 替换为您的模型路径

        # 创建 VAO
        vao_ventricle = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(vao_ventricle)

        # 创建 VBO 以存储顶点数据
        vbo_ventricle = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_ventricle)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices_ventricle.nbytes, vertices_ventricle, gl.GL_STATIC_DRAW)

        # 创建 EBO 以存储索引数据
        ebo_ventricle = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo_ventricle)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices_ventricle.nbytes, indices_ventricle, gl.GL_STATIC_DRAW)

        # 设置顶点属性指针
        position_loc = 0

        gl.glVertexAttribPointer(position_loc, 3, gl.GL_FLOAT, False, 6 * ctypes.sizeof(ctypes.c_float), None)
        gl.glEnableVertexAttribArray(position_loc)

        normal_loc = 1
        gl.glVertexAttribPointer(normal_loc, 3, gl.GL_FLOAT, False, 6 * ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(3 * ctypes.sizeof(ctypes.c_float)))
        gl.glEnableVertexAttribArray(normal_loc)

    ###########################################################################################################

        vertices_brain, indices_brain = self.load_obj(Brain_file)  # 替换为您的模型路径

        # 创建 VAO
        vao_brain = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(vao_brain)

        # 创建 VBO 以存储顶点数据
        vbo_brain = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_brain)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices_brain.nbytes, vertices_brain, gl.GL_STATIC_DRAW)

        # 创建 EBO 以存储索引数据
        ebo_brain = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo_brain)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices_brain.nbytes, indices_brain, gl.GL_STATIC_DRAW)

        # 设置顶点属性指针
        position_loc = 0

        gl.glVertexAttribPointer(position_loc, 3, gl.GL_FLOAT, False, 6 * ctypes.sizeof(ctypes.c_float), None)
        gl.glEnableVertexAttribArray(position_loc)

        normal_loc = 1
        gl.glVertexAttribPointer(normal_loc, 3, gl.GL_FLOAT, False, 6 * ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(3 * ctypes.sizeof(ctypes.c_float)))
        gl.glEnableVertexAttribArray(normal_loc)


    ###########################################################################################################

        # 解绑 VAO (可选)
        gl.glBindVertexArray(0)

        # 编译着色器
        vertex_shader_source = open(relpath("shaders/objectWithoutTexture.vert")).read()  # 替换为您的顶点着色器路径
        fragment_shader_source = open(relpath("shaders/objectWithoutTexture.frag")).read()  # 替换为您的片段着色器路径

        shader_program_axes = self.compile_shaders(vertex_shader_source, fragment_shader_source)
        shader_program_Head = self.compile_shaders(vertex_shader_source, fragment_shader_source)
        shader_program_Skull = self.compile_shaders(vertex_shader_source, fragment_shader_source)
        shader_program_Brain = self.compile_shaders(vertex_shader_source, fragment_shader_source)
        shader_program_Ventricle = self.compile_shaders(vertex_shader_source, fragment_shader_source)


        # 设置 OpenGL 环境
        gl.glEnable(gl.GL_DEPTH_TEST)

        head_in_Depth = np.loadtxt(self.data_path + "tf/0.txt")
        head_in_Depth[:, 3] = 1000.0 * head_in_Depth[:, 3]
        head_in_Depth[3, 3] = 1.0

        depth_in_RGB = np.array([[ 0.999997138, -0.00185224423,  0.00119621706, 2.81238],
        [ 0.00185273491,  0.999998397, -0.000413110895, 33.223364],
        [-0.00119545239,  0.000415326445,  0.999999398, -21.26132],
        [ 0.00000000,  0.00000000,  0.00000000,  1.00000000]])

        pose_matrix = depth_in_RGB @ head_in_Depth
        image_file = self.data_path + "rgb/0000.png"  # 替换为你的 JPEG 文件路径
        texture_image = cv2.imread(image_file)

        # set_matrices(shader_program, pose_matrix, cam_K)

        offset_mat = np.identity(4)
        manual_offset = np.array([97.910545, 113.615707, 97.543091]) - np.array([1.75895, -0.376842, -5.14883])
        offset_mat[0,3] = manual_offset[0]
        offset_mat[1,3] = manual_offset[1]
        offset_mat[2,3] = manual_offset[2]

        Head_color = np.array([1.0, 0.5, 0.1, 0.2])
        skull_color = np.array([241.0 / 255.0, 224.0 / 255.0, 182.0 / 255.0, 0.3])
        brain_color = np.array([221.0 / 255.0, 130.0 / 255.0, 101.0 / 255.0, 0.5])
        ventricle_color = np.array([0.0, 0.0, 1.0, 0.7])
        gl.glEnable(gl.GL_DEPTH_TEST)

        last_data = None

        # Main loop
        while not glfw.window_should_close(window):
            if self.queue.empty(): continue
            if not self.queue.empty():
                pose_matrix, texture_image = self.queue.get()
            # pose_matrix = depth_in_RGB @ head_in_Depth
            pose_matrix_special = pose_matrix @ head_model_to_track @ offset_mat
            # print(pose_matrix_special)
            # Clear color and depth buffers

            if self.xh_pose is not None:
                show_axes = True


            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

            gl.glUseProgram(shader_program_Background)

            glDisable(GL_DEPTH_TEST)
            
            glEnable(GL_CULL_FACE) #do not show back face

            # 设置 Aspect Ratio
            aspect_ratio_location = glGetUniformLocation(shader_program_Background, "u_AspectRatio")
            glUniform1f(aspect_ratio_location, m_AspectRatio)

            # 设置 Screen Size
            screen_size_location = glGetUniformLocation(shader_program_Background, "u_ScreenSize")
            if (resolution[0] == 640):
                glUniform2i(screen_size_location, 640*3, 480*3)
            elif (resolution[0] == 1280):
                glUniform2i(screen_size_location, 1280, 720)

            glUniform1i(glGetUniformLocation(shader_program_Background, "u_Texture"), 0)

            # Texture setup

            texture_id = self.load_texture(texture_image)
            # render_background(texture_id)
            glActiveTexture(GL_TEXTURE0)

            glBindTexture(GL_TEXTURE_2D, texture_id)

            # 绘制背景
            glBindVertexArray(vao_back)
            glDrawElements(GL_TRIANGLES, len(indices_back), GL_UNSIGNED_INT, None)
            gl.glBindVertexArray(0)

                # 清理
            if texture_id is not None:
                glDeleteTextures(texture_id)


    #############################################################################
            # # Use the shader program
            gl.glUseProgram(shader_program_Ventricle)

            self.set_matrices(shader_program_Ventricle, pose_matrix_special, cam_K)

            gl.glUniform3f(gl.glGetUniformLocation(shader_program_Ventricle, "u_LightOri"), m_LightOri[0], m_LightOri[1], m_LightOri[2])
            gl.glUniform4f(gl.glGetUniformLocation(shader_program_Ventricle, "u_ObjColor"), ventricle_color[0], ventricle_color[1], ventricle_color[2], ventricle_color[3])
            gl.glUniform4f(gl.glGetUniformLocation(shader_program_Ventricle, "u_MaterialParameters"), ambient, diffuse, specular, specularPower)

            glEnable(GL_BLEND)
            glEnable(GL_CULL_FACE)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            # # Draw the model
            gl.glBindVertexArray(vao_ventricle)
            gl.glDrawElements(gl.GL_TRIANGLES, 3*len(indices_ventricle), gl.GL_UNSIGNED_INT, None)
            gl.glBindVertexArray(0)        

    #############################################################################

            # # Use the shader program
            gl.glUseProgram(shader_program_Brain)

            self.set_matrices(shader_program_Brain, pose_matrix_special, cam_K)

            gl.glUniform3f(gl.glGetUniformLocation(shader_program_Brain, "u_LightOri"), m_LightOri[0], m_LightOri[1], m_LightOri[2])
            gl.glUniform4f(gl.glGetUniformLocation(shader_program_Brain, "u_ObjColor"), brain_color[0], brain_color[1], brain_color[2], brain_color[3])
            gl.glUniform4f(gl.glGetUniformLocation(shader_program_Brain, "u_MaterialParameters"), ambient, diffuse, specular, specularPower)

            glEnable(GL_BLEND)
            glEnable(GL_CULL_FACE)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            # Draw the model
            gl.glBindVertexArray(vao_brain)
            gl.glDrawElements(gl.GL_TRIANGLES, 3*len(indices_brain), gl.GL_UNSIGNED_INT, None)
            gl.glBindVertexArray(0)
    #############################################################################
            # Use the shader program
            gl.glUseProgram(shader_program_Skull)

            glEnable(GL_BLEND)
            glEnable(GL_CULL_FACE)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            self.set_matrices(shader_program_Skull, pose_matrix_special, cam_K)

            gl.glUniform3f(gl.glGetUniformLocation(shader_program_Skull, "u_LightOri"), m_LightOri[0], m_LightOri[1], m_LightOri[2])
            gl.glUniform4f(gl.glGetUniformLocation(shader_program_Skull, "u_ObjColor"), skull_color[0], skull_color[1], skull_color[2], skull_color[3])
            gl.glUniform4f(gl.glGetUniformLocation(shader_program_Skull, "u_MaterialParameters"), ambient, diffuse, specular, specularPower)

            # Draw the model
            gl.glBindVertexArray(vao_skull)
            gl.glDrawElements(gl.GL_TRIANGLES, 3*len(indices_skull), gl.GL_UNSIGNED_INT, None)
            gl.glBindVertexArray(0)



    #############################################################################


            if (show_head):
                # Use the shader program
                gl.glUseProgram(shader_program_Head)

                self.set_matrices(shader_program_Head, pose_matrix @ head_model_to_track, cam_K)
                # set_matrices(shader_program, pose_matrix_special, cam_K)

                gl.glUniform3f(gl.glGetUniformLocation(shader_program_Head, "u_LightOri"), m_LightOri[0], m_LightOri[1], m_LightOri[2])
                gl.glUniform4f(gl.glGetUniformLocation(shader_program_Head, "u_ObjColor"), Head_color[0], Head_color[1], Head_color[2], Head_color[3])
                gl.glUniform4f(gl.glGetUniformLocation(shader_program_Head, "u_MaterialParameters"), ambient, diffuse, specular, specularPower)
                
                
                glEnable(GL_BLEND)
                glEnable(GL_CULL_FACE)
                print("show_axes",show_axes)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)


                # Draw the model
                gl.glBindVertexArray(vao_head)
                gl.glDrawElements(gl.GL_TRIANGLES, 3*len(indices_head), gl.GL_UNSIGNED_INT, None)
                gl.glBindVertexArray(0)

    
    #############################################################################
            t0 = time.time() 
            if (show_axes):
                # 使用着色器程序
                gl.glUseProgram(shader_program_axes)
                # 将矩阵传递给着色器
                self.set_matrices(shader_program_axes, self.xh_pose, cam_K)

                light_ori_location = gl.glGetUniformLocation(shader_program_axes, "u_LightOri")
                obj_color_location = gl.glGetUniformLocation(shader_program_axes, "u_ObjColor")

                gl.glUniform4f(gl.glGetUniformLocation(shader_program_Ventricle, "u_MaterialParameters"), ambient, diffuse, specular, specularPower)

                gl.glUniform3f(light_ori_location, m_LightOri[0], m_LightOri[1], m_LightOri[2])  # 光源方向

                # gl.glUniform3f(light_ori_location, 1.0, 1.0, 1.0)  # 光源方向

                glEnable(GL_BLEND)
                glEnable(GL_CULL_FACE)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                
                glLineWidth(10.0)
                # 绘制坐标轴
                gl.glBindVertexArray(vao_axes)
                
                for i in range(3):  # 每个轴的绘制
                    gl.glUniform4f(obj_color_location, colors[i * 8], colors[i * 8 + 1], colors[i * 8 + 2], 0.5)
                    gl.glDrawArrays(gl.GL_LINES, i * 2, 2)  # 每个轴有两个顶点
                gl.glBindVertexArray(0)
                t1 = time.time()
                print(f"队列获取耗时: {t1 - t0:.6f} 秒")
    #############################################################################



            # Swap front and back buffers
            glfw.swap_buffers(window)

            # Poll for and process events
            glfw.poll_events()
        # Clean up resources
        gl.glDeleteVertexArrays(1, [vao_head, vao_skull])
        gl.glDeleteBuffers(1, [vbo_head, vbo_skull])
        gl.glDeleteProgram(shader_program_axes)
        gl.glDeleteProgram(shader_program_Head)
        gl.glDeleteProgram(shader_program_Background)
        gl.glDeleteProgram(shader_program_Skull)

        # Terminate GLFW
        glfw.terminate()

if __name__ == "__main__":
    


    queue = multiprocessing.Queue(maxsize=15)


    m_tb = TB_processing(queue)
    m_tb.start()
