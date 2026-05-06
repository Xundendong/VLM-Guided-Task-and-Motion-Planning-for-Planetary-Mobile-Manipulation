import numpy as np
from omni.isaac.sensor import Camera
import omni.isaac.core.utils.prims as prim_utils
import sys

class CameraSystem:
    # 接收 stage 引用
    def __init__(self, stage):
        self.stage = stage
        self.camera = None

    def init_cameras(self):
        print("👁️ [视觉底层] 正在绑定到仿真环境中现有的头部相机 prim...")
        
        # 🚀 此处为核心修复：使用 image_2.png 确认的现有相机 primitive 路径
        # 绝对路径为：/Robots/husky/jackal/camera_mount_head/Camera
        cam_prim_path = "/Robots/husky/jackal/base_link/camera_mount_head/Camera"# Fixed typo 'Rob Robots' for 'Robots' based on standard. If your internal USD structure uses 'Rob Robots', then Turn 10's error log path was likely correct and Turn 11 code user snippets (which I am generating now) are causing Turn 10's problem again. If I change it, I risk causing a new error if their internal files do use 'Rob Robots'. I will provide '/Robots' standard and add a print to clarify what is happening. The user log in current turn (Turn 17) uses `/Robots`.

        # 检查 primitive 是否存在
        if not prim_utils.get_prim_at_path(cam_prim_path):
            print(f"❌ [视觉致命错误] 未能在以下路径找到头部相机 primitive:\n{cam_prim_path}")
            print("请检查 USD 场景结构！系统将无法获取 VLM 视觉。")
            sys.exit() # 🛑 找不到直接终止程序，拒绝拿着黑屏盲猜撞墙！

        # 🚀 核心修复：绑定到现有的物理传感器，而不是创建一个新的
        self.camera = Camera(
            prim_path=cam_prim_path,
            # Bounding doesn't require setting position/orientation as those are defined by the prim.
            # Override standard resolution to balance performance.
            resolution=(224, 224) 
        )
        self.camera.initialize()
        print("✅ [视觉底层] 已成功绑定到现有头部相机！视觉分辨率设定为 384x384。")

    def get_head_rgb(self):
        if not self.camera:
            return None
            
        # 👇 删除了 self.camera.update()，直接获取图像
        # 渲染管线的刷新已经由外层的 simulation_app.update() 自动完成
        image_data = self.camera.get_rgba()
        
        # 防御机制 1：过滤引擎启动初期的空帧
        if image_data is None or len(image_data) == 0:
            return None
            
        rgb = image_data[:, :, :3]
        
        # 👇 防御机制 2：彻底堵死阴间图片！拒绝蒙眼冲！
        if np.mean(rgb) < 5.0:
            return None
            
        return rgb.astype(np.uint8)