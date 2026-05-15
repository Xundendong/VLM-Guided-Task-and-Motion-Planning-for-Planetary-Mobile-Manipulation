__author__ = "Antoine Richard & Customized by You (Ultimate Spatial Director & Fallback Edition)"
__license__ = "BSD-3-Clause"

import os
import sys
import datetime
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
from omegaconf import DictConfig, OmegaConf, ListConfig
import logging
import carb
import hydra

import base64
import json
import io
import requests
from PIL import Image
import re

from src.configurations import configFactory
from src.environments_wrappers import startSim

# ==========================================
# 📝 1. 日志与配置
# ==========================================
run_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.path.join(os.getcwd(), "logs", f"mission_{run_time}")
os.makedirs(LOG_DIR, exist_ok=True)
log_file_path = os.path.join(LOG_DIR, "mission_record.log")
# ==========================================
# 维度二：给日志接管器装上“过滤网”
# ==========================================
class DualLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
        self.log.write(f"\n\n{'='*60}\n🚀 终极空间指挥与硬件深度测距系统启动: {run_time}\n📂 记录: {LOG_DIR}\n{'='*60}\n")
        
    def write(self, message):
        # 拦截特征码：如果底层 C++ 库强行往控制台塞警告，在这里直接拦截丢弃！
        lower_msg = message.lower()
        if "[warning]" in lower_msg or "deprecation" in lower_msg or "futurewarning" in lower_msg:
            return  # 直接 return，不打印也不写入文件
            
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = DualLogger(log_file_path)

# ==========================================
#  维度三：压制底层第三方库的日志输出
# ==========================================
import logging
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("numba").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR) 
logging.getLogger("transformers").setLevel(logging.ERROR) 
logging.getLogger("AutoNode").setLevel(logging.ERROR)
class DualLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
        self.log.write(f"\n\n{'='*60}\n🚀 空间指挥与容错视觉伺服启动: {run_time}\n📂 记录: {LOG_DIR}\n{'='*60}\n")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = DualLogger(log_file_path)
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

def resolve_tuple(*args): return tuple(args)
OmegaConf.register_new_resolver("as_tuple", resolve_tuple)

def omegaconfToDict(d):
    if isinstance(d, DictConfig): return {k: omegaconfToDict(v) for k, v in d.items()}
    elif isinstance(d, ListConfig): return [omegaconfToDict(i) for i in d]
    return d

def instantiateConfigs(cfg):
    configs = configFactory.getConfigs()
    ret = {}
    for k, v in cfg.items():
        if isinstance(v, dict): ret[k] = configFactory(k, **v) if k in configs else instantiateConfigs(v)
        else: ret[k] = v
    return ret

from PIL import Image, ImageDraw
import re

# ==========================================
# 🧠 2A. VLM 认知大脑 (全英文输出 + 智能裁判督导)
# ==========================================
class VLMAgent:
    def __init__(self):
        self.server_url = "http://127.0.0.1:8000/vlm_decide" 
        print(f"✅ [大脑代理] VLM 语义中枢初始化完毕 (全英文空间指挥与裁判模式)")

    def _encode_image(self, rgb_array):
        img = Image.fromarray(rgb_array.astype(np.uint8))
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def extract_target_prompt(self, rgb_image, instruction):
        img_str = self._encode_image(rgb_image)
        prompt = f"""
        You are an Embodied AI visual commander operating a lunar rover. 
        The human instruction is: "{instruction}" 
        (Note: Understand the instruction in Chinese, but your visual targets MUST be output in English).

        CRITICAL RULES:
        1. There is a large white robotic arm in the bottom-left/center of the image. DO NOT mistake it as the target.
        2. Analyze the human instruction to determine the TRUE MAIN TARGET (e.g., is the human asking to find a 'crater', a 'rock', or something else?).
        3. Output a general bounding box (roi_box) covering the area where this target is located [ymin, xmin, ymax, xmax] (0.0 to 1.0). If the target is the crater itself, output the box of the crater.
        4. Output a highly specific ENGLISH prompt for this final target. e.g., 'the crater.' or 'the small rock.'
        
        Return a STRICT JSON object exactly like this:
        {{
            "reasoning": "用中文简短分析人类到底想找什么（坑还是石头？），以及它的位置",
            "roi_box": [0.3, 0.4, 0.8, 0.9],
            "target_prompt": "English phrase for the final target. e.g., 'the crater.'"
        }}
        """
        try:
            payload = {"image_base64": img_str, "instruction": prompt}
            response = requests.post(self.server_url, json=payload, timeout=30)
            if response.status_code == 200:
                text = response.json().get("result", "")
                clean_text = text.replace("```json", "").replace("```", "").strip()
                json_match = re.search(r'\{.*?\}', clean_text, re.DOTALL)
                if json_match:
                    action_dict = json.loads(json_match.group())
                    print(f"🧠 [VLM 空间推理]: {action_dict.get('reasoning', '无')}")
                    
                    roi_box = action_dict.get("roi_box", [0.0, 0.0, 1.0, 1.0])
                    # 修复 Qwen-VL 7B 可能输出 0-1000 绝对坐标的问题
                    if any(v > 1.1 for v in roi_box):
                        roi_box = [max(0.0, min(1.0, v / 1000.0)) for v in roi_box]
                        
                    target = action_dict.get("target_prompt", "the rock.")
                    if not target.endswith("."): target += "."
                    
                    return {"roi_box": roi_box, "target_prompt": target}
            return {"roi_box": [0.0, 0.0, 1.0, 1.0], "target_prompt": "the rock."}
        except Exception as e:
            print(f"❌ [VLM 大脑异常]: {e}")
            return {"roi_box": [0.0, 0.0, 1.0, 1.0], "target_prompt": "the rock."}

    def verify_dino_box(self, rgb_image, box, target_prompt):
        """VLM 裁判机制：在原图上画出 DINO 的框，让 VLM 进行最终裁决"""
        img = Image.fromarray(rgb_image.astype(np.uint8))
        draw = ImageDraw.Draw(img)
        xmin, ymin, xmax, ymax = box
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=6)
        
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        prompt = f"""
        You are a quality inspector for an autonomous rover.
        I have drawn a thick RED BOX on the image. Look closely at the object inside the RED BOX.
        Is the object inside the red box the correct target: "{target_prompt}"?
        CRITICAL WARNING: The large white/grey structure at the bottom is the rover's robotic arm. If the red box is on the robotic arm, it is WRONG (return false).
        
        Return STRICT JSON:
        {{
            "reasoning": "用中文回答：红框里到底是什么？是机械臂还是石头？",
            "is_correct": true or false
        }}
        """
        try:
            payload = {"image_base64": img_str, "instruction": prompt}
            response = requests.post(self.server_url, json=payload, timeout=30)
            if response.status_code == 200:
                text = response.json().get("result", "")
                clean_text = text.replace("```json", "").replace("```", "").strip()
                json_match = re.search(r'\{.*?\}', clean_text, re.DOTALL)
                if json_match:
                    res_dict = json.loads(json_match.group())
                    is_correct = res_dict.get("is_correct", False)
                    print(f"⚖️ [VLM 裁判庭]: {res_dict.get('reasoning', '无')} -> 判决: {'✅ 通过' if is_correct else '❌ 驳回 (判定为干扰物)'}")
                    return is_correct
            return False
        except Exception as e:
            print(f"⚠️ [VLM 裁判异常]: {e}")
            return False

# ==========================================
# 🎯 2B. DINO 小脑追踪器 (防抖记忆跟踪 + 物理深度硬件直连)
# ==========================================
import numpy as np
import math
import io
import base64
import requests
from PIL import Image

class DinoTracker:
    def __init__(self,depth_cam):
        self.server_url = "http://127.0.0.1:8000/dino_detect" 
        # ❌ 已删除 self.depth_url，不再使用 AI 脑补深度
        
        print(f"✅ [小脑代理] DINO 鹰眼接口初始化完毕 (开启防抖记忆跟踪 & 物理测距)")
        
        # ==========================================
        # 📷 核心硬件接入：挂载物理深度相机
        # ==========================================
        self.depth_cam = depth_cam

        self.Kp_yaw = 0.15    
        self.Kp_dist = 0.0015 
        self.target_area = 15000 
        
        self.fx, self.fy = 320.0, 320.0  
        self.cx, self.cy = 320.0, 240.0  

        # 🧠 短期记忆模块
        self.last_u = None
        self.last_v = None

    def _encode_image(self, rgb_array):
        img = Image.fromarray(rgb_array.astype(np.uint8))
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _calc_pid(self, box, img_w):
        xmin, ymin, xmax, ymax = box
        u_center = (xmin + xmax) / 2.0
        error_yaw = u_center - (img_w / 2.0) 
        
        # 💡 [微调] 稍微加回一点点转向灵敏度，解决慢吞吞的问题
        self.Kp_yaw = 0.15
        turn_vel = error_yaw * self.Kp_yaw if abs(error_yaw) > 15 else 0.0

        current_area = (xmax - xmin) * (ymax - ymin)
        error_dist = self.target_area - current_area
        # 💡 [微调] 略微提升基础推力系数
        base_forward_vel = max(0.0, min(error_dist * 0.0025, 30.0)) 

        # =======================================================
        # 🚨 [逻辑疏通] 更加丝滑的偏航惩罚
        # =======================================================
        if abs(error_yaw) > 200:
            # 只有极其偏的时候才原地踏步
            forward_vel = 0.0    
            print(f"🔄 [原地对准] 偏差 {error_yaw:.1f}px, 暂停前进")
        elif abs(error_yaw) > 80:
            # 中等偏差，允许以 40% 的速度慢慢推进
            forward_vel = base_forward_vel * 0.4 
        else:
            # 80像素以内，认为已经对得不错了，全速冲刺！
            forward_vel = base_forward_vel
            
        # 物理限幅：转向上限维持 12-15 即可，保证画面不晃
        turn_vel = max(-15.0, min(turn_vel, 15.0))      
        
        return forward_vel, turn_vel, error_yaw

    def _find_best_box_by_memory(self, boxes):
        """🧠 防抖黑科技：在多个候选框中，寻找距离上一帧记忆最近的框"""
        if not self.last_u or not self.last_v:
            # 如果没有记忆，直接返回第一个（得分最高的）
            best_box = boxes[0]
        else:
            min_dist = float('inf')
            best_box = boxes[0]
            for box in boxes:
                u = (box[0] + box[2]) / 2.0
                v = (box[1] + box[3]) / 2.0
                dist = math.hypot(u - self.last_u, v - self.last_v)
                # 距离惩罚策略
                if dist < min_dist:
                    min_dist = dist
                    best_box = box

        # 更新记忆
        self.last_u = (best_box[0] + best_box[2]) / 2.0
        self.last_v = (best_box[1] + best_box[3]) / 2.0
        return best_box

    def _get_3d_coordinates(self, rgb_image, box):
        """📏 核心黑科技：直连物理硬件深度相机与 5x5 中值滤波防突变"""
        u = int((box[0] + box[2]) / 2.0)
        v = int((box[1] + box[3]) / 2.0)
        
        img_h, img_w = rgb_image.shape[:2]
        if not (0 <= u < img_w and 0 <= v < img_h):
            return 0.0, 0.0, 0.0
            
        try:
            # 1. 获取 Isaac Sim 物理深度缓存
            depth_map = self.depth_cam.get_depth() 
            if depth_map is None or depth_map.size == 0:
                return 0.0, 0.0, 0.0
            
            # 2. 5x5 区域采样防边缘跳变 (Median Pooling)
            u_min, u_max = max(0, u-2), min(img_w-1, u+2)
            v_min, v_max = max(0, v-2), min(img_h-1, v+2)
            region_depths = depth_map[v_min:v_max+1, u_min:u_max+1]
            
            valid_depths = region_depths[np.isfinite(region_depths)]
            if len(valid_depths) > 0:
                z_c = float(np.median(valid_depths))
                
                if z_c > 0:
                    # 3. 针孔逆向投射计算真实物理 X, Y (单位：米)
                    x_c = (u - self.cx) * z_c / self.fx
                    y_c = (v - self.cy) * z_c / self.fy
                    return x_c, y_c, z_c
        except Exception as e:
            print(f"⚠️ [硬件测距异常]: {e}")
        return 0.0, 0.0, 0.0

    def track_with_vlm_roi(self, rgb_image, vlm_task_dict, vlm_brain):
        img_h, img_w = rgb_image.shape[:2]
        target_prompt = vlm_task_dict.get("target_prompt", "rock.")
        ymin, xmin, ymax, xmax = vlm_task_dict.get("roi_box", [0.0, 0.0, 1.0, 1.0])

        margin_y, margin_x = int(img_h * 0.10), int(img_w * 0.10)
        c_ymin = max(0, int(ymin * img_h) - margin_y)
        c_ymax = min(img_h, int(ymax * img_h) + margin_y)
        c_xmin = max(0, int(xmin * img_w) - margin_x)
        c_xmax = min(img_w, int(xmax * img_w) + margin_x)
        
        cropped_img = rgb_image[c_ymin:c_ymax, c_xmin:c_xmax]
        
        # 🟢 方案 A：局部寻找
        if cropped_img.shape[0] > 20 and cropped_img.shape[1] > 20:
            try:
                res_local = requests.post(self.server_url, json={"image_base64": self._encode_image(cropped_img), "text_prompt": target_prompt}, timeout=5).json()
                if res_local.get("found"):
                    local_boxes = res_local["boxes"] # 现在拿到的是多个框
                    # 将所有的局部框映射到全局
                    global_boxes = [[c_xmin + b[0], c_ymin + b[1], c_xmin + b[2], c_ymin + b[3]] for b in local_boxes]
                    
                    # 🔴 动用记忆：挑选出最不闪烁的那一个
                    g_box = self._find_best_box_by_memory(global_boxes)
                    
                    # 只有当这是第一次锁定目标时（没有记忆），才呼叫 VLM 进行重度审批，省算力！
                    if not self.last_u:
                        print(f"👀 首次发现目标，提交 VLM 审批...")
                        if not vlm_brain.verify_dino_box(rgb_image, g_box, target_prompt):
                            self.last_u, self.last_v = None, None # 审批失败，清除记忆
                            return 0.0, 15.0 # 强制转圈
                    
                    # 调用改写后的物理测距
                    x3d, y3d, z3d = self._get_3d_coordinates(rgb_image, g_box)
                    print(f"📍 [稳定锁定] X={x3d:.3f}m, Y={y3d:.3f}m, 物理深度Z={z3d:.3f}m")
                    
                    forward_vel, turn_vel, error_yaw = self._calc_pid(g_box, img_w)
                    return forward_vel, turn_vel
            except Exception: pass

        # 🔴 方案 B：全局盲扫
        print(f"🔄 [丢失目标] 记忆清空，全局重新寻找: [{target_prompt}]")
        self.last_u, self.last_v = None, None # 一旦进入盲扫，必须清除旧记忆
        try:
            res_global = requests.post(self.server_url, json={"image_base64": self._encode_image(rgb_image), "text_prompt": target_prompt}, timeout=5).json()
            if res_global.get("found"):
                g_boxes = res_global["boxes"]
                g_box = self._find_best_box_by_memory(g_boxes)
                
                print(f"👀 全局扫描发现目标，提交 VLM 审批...")
                if vlm_brain.verify_dino_box(rgb_image, g_box, target_prompt):
                    # 调用改写后的物理测距
                    x3d, y3d, z3d = self._get_3d_coordinates(rgb_image, g_box)
                    print(f"📍 [全局稳定锁定] X={x3d:.3f}m, Y={y3d:.3f}m, 物理深度Z={z3d:.3f}m")
                    forward_vel, turn_vel, error_yaw = self._calc_pid(g_box, img_w)
                    return forward_vel, turn_vel
                else:
                    self.last_u, self.last_v = None, None
                    return 0.0, 15.0
            else:
                return 0.0, 15.0
        except Exception: return 0.0, 0.0
# ==========================================
# 🚀 3. 主程序入口与仿真循环
# ==========================================
@hydra.main(config_name="config", config_path="cfg")
def run(cfg: DictConfig):
    cfg_container = OmegaConf.to_container(cfg, resolve=True)
    cfg_inst = instantiateConfigs(cfg_container)
    
    print("\n⏳ Isaac Sim 物理引擎启动中...\n")
    SM, simulation_app = startSim(cfg_inst)

    logging.getLogger("omni.physx.plugin").setLevel(logging.ERROR)
    carb.settings.get_settings().set("/log/level", "error")

    import omni.usd
    from pxr import UsdPhysics, UsdGeom, PhysxSchema
    import omni.timeline
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.prims import RigidPrim
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.extensions import get_extension_path_from_name
    from omni.isaac.sensor import Camera

    ext_name = "isaacsim.robot_motion.motion_generation"
    mg_ext_path = get_extension_path_from_name(ext_name)
    if mg_ext_path is None: 
        ext_name = "omni.isaac.motion_generation"
        mg_ext_path = get_extension_path_from_name(ext_name)

    if "isaacsim" in ext_name:
        from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
    else:
        from omni.isaac.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver

    stage = omni.usd.get_context().get_stage()
    timeline = omni.timeline.get_timeline_interface()

    paths_to_remove = []
    for prim in stage.Traverse():
        if prim.GetTypeName() in ["ActionGraph", "OmniGraphNode"]: paths_to_remove.append(prim.GetPath())
    for p in paths_to_remove: stage.RemovePrim(p)

    wheel_prims = []
    for prim in stage.Traverse():
        if prim.GetTypeName() == "PhysicsRevoluteJoint" and "wheel" in prim.GetName().lower():
            wheel_prims.append(prim)
            drive = UsdPhysics.DriveAPI.Get(prim, "angular") or UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive.GetTargetVelocityAttr().Set(0.0)

    my_robot = Articulation("/Robots/husky/jackal")
    my_robot.initialize()
    
    safe_home_angles = np.array([0.0, -1.0, 1.0, -1.0, -1.57, 0.0])
    joint_indices = [idx for kw in ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"] for idx, name in enumerate(my_robot.dof_names) if kw in name.lower()]

    config_dir = os.path.join(mg_ext_path, "motion_policy_configs")
    urdf_path, yaml_path = "", ""
    for root, dirs, files in os.walk(config_dir):
        if "ur3e" in root.lower():
            for f in files:
                if f.endswith(".urdf") and "ur3e" in f.lower(): urdf_path = os.path.join(root, f)
                if f.endswith(".yaml") and "ur3e" in root.lower(): yaml_path = os.path.join(root, f)

    ik_solver = LulaKinematicsSolver(robot_description_path=yaml_path, urdf_path=urdf_path)
    ee_solver = ArticulationKinematicsSolver(my_robot, ik_solver, "wrist_3_link")

    # ========================================================
    # 📸 4. 挂载 OV9782 高杆相机
    # ========================================================
    real_mast_cam_prim_path = "/Robots/husky/jackal/base_link/mast_base/rsd455/RSD455/Camera_OmniVision_OV9782_Color" 
    mast_cam = Camera(prim_path=real_mast_cam_prim_path)
    mast_cam.initialize() 
    mast_cam.set_resolution((640, 480))
    depth_cam_prim_path = "/Robots/husky/jackal/base_link/mast_base/rsd455/RSD455/Camera_Pseudo_Depth"
    depth_cam = Camera(prim_path=depth_cam_prim_path)
    depth_cam.initialize()
    depth_cam.set_resolution((640,480))
    depth_cam.add_distance_to_image_plane_to_frame()
    print("深度相机已经挂载")
    
    # ========================================================
    # 🎯 5. 不穿模月岩加载
    # ========================================================
    target_x, target_y, drop_z = 2.6049, 2.18118, 0.1 
    rock_prim_path = "/World/SRB_Apollo_Rock"
    apollo_rock_path = os.path.expanduser("~/Luna-VLA/ript-vla/space_robotics_bench/assets/srb_assets/object/rock/apollo_sample1.usdz")
    
    if os.path.exists(apollo_rock_path):
        add_reference_to_stage(usd_path=apollo_rock_path, prim_path=rock_prim_path)
        for prim in stage.Traverse():
            if prim.GetPath().HasPrefix(rock_prim_path) and prim.IsA(UsdGeom.Mesh):
                if prim.HasAPI(UsdPhysics.CollisionAPI): prim.RemoveAPI(UsdPhysics.CollisionAPI)
                UsdPhysics.CollisionAPI.Apply(prim)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("convexDecomposition")
                physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                physx_collision_api.CreateContactOffsetAttr().Set(0.005) 
                physx_collision_api.CreateRestOffsetAttr().Set(0.001)

        test_rock = RigidPrim(prim_path=rock_prim_path, name="srb_rock", position=np.array([target_x, target_y, drop_z]), scale=np.array([1.0, 1.0, 1.0]), mass=0.5)
        test_rock.initialize()
        PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath(rock_prim_path)).CreateEnableCCDAttr().Set(True)

    
    
    # ========================================================
    # 🟢 双模初始化与终极状态机主循环
    # ========================================================
    vlm_brain = VLMAgent()
    dino_tracker = DinoTracker(depth_cam=depth_cam)
    
    base_instruction = "你的右前方有一个圆形的陨石坑，坑的底部中间有一块小小的岩石，看起来很小。请找到这块岩石。"
    
    # 🔴 核心状态机变量
    ROBOT_STATE = "SEARCHING"  # 初始状态：搜索
    locked_roi_box = None      # 锁定后的局部跟踪框 [xmin, ymin, xmax, ymax]
    
    startup_frames, step_counter = 0, 0
    VISION_INTERVAL = 15 
    
    while simulation_app.is_running():
        simulation_app.update()
        if not timeline.is_playing(): timeline.play()

        # 1. 物理自检与机械臂展开 (前90帧)
        if timeline.is_playing() and startup_frames <= 90:
            if startup_frames < 20 and len(joint_indices) == 6:
                my_robot.set_joint_positions(safe_home_angles, joint_indices=joint_indices)
            elif 20 < startup_frames <= 50:
                current_pos, _ = ee_solver.compute_end_effector_pose()
                action, _ = ee_solver.compute_inverse_kinematics(current_pos + np.array([1.0, 0, 0]) * 0.005, np.array([0, 0.7071, 0.7071, 0]))
                my_robot.apply_action(action)
            elif 50 < startup_frames < 90:
                current_pos, _ = ee_solver.compute_end_effector_pose()
                action, _ = ee_solver.compute_inverse_kinematics(current_pos + np.array([-1.0, 0, 0]) * 0.005, np.array([0, 0.7071, 0.7071, 0]))
                my_robot.apply_action(action)
            startup_frames += 1
            continue

        # 2. 视觉伺服与状态机流转
        if timeline.is_playing() and startup_frames > 90:
            if step_counter % VISION_INTERVAL == 0: 
                rgba_data = mast_cam.get_rgba()
                
                if rgba_data is not None and rgba_data.size > 0:
                    rgb_img = rgba_data[:, :, :3]
                    img_h, img_w = rgb_img.shape[:2]
                    
                    # ====================================================
                    # 🟡 状态 1: 搜索与 EQA 终极锁定 (停车思考模式)
                    # ====================================================
                    if ROBOT_STATE == "SEARCHING":
                        # 🚨 停车思考原则：大模型看图时，底盘必须绝对静止！不许乱动！
                        if len(wheel_prims) > 0:
                            for prim in wheel_prims:
                                UsdPhysics.DriveAPI.Get(prim, "angular").GetTargetVelocityAttr().Set(0.0)

                        print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] 🔍 [状态: SEARCHING] 原地停车，启动 VLM+DINO 级联搜索...")
                        task_dict = vlm_brain.extract_target_prompt(rgb_img, base_instruction)
                        
                        if task_dict:
                            target_prompt = task_dict.get('target_prompt', 'rock.')
                            print(f"🎯 VLM 锁定目标特征: {target_prompt}，正在呼叫 DINO 执行全局 Set-of-Mark...")
                            
                            # 强制进行一次全局搜索并让 VLM 裁判拍板
                            res_global = requests.post(dino_tracker.server_url, json={"image_base64": dino_tracker._encode_image(rgb_img), "text_prompt": target_prompt}, timeout=5).json()
                            
                            if res_global.get("found"):
                                # DINO 可能返回了多个框，我们用记忆系统挑出最好的，或者直接让 VLM 裁决第一个
                                best_box = dino_tracker._find_best_box_by_memory(res_global["boxes"])
                                
                                print("⚖️ 提交 VLM 裁判庭进行最终目标确认 (EQA 拍板)...")
                                if vlm_brain.verify_dino_box(rgb_img, best_box, target_prompt):
                                    print("✅ [EQA 确认通过] 目标已绝对锁死！VLM 进入休眠，移交底盘控制权！")
                                    locked_roi_box = best_box
                                    ROBOT_STATE = "TRACKING" # 🔴 状态切换！
                                else:
                                    print("❌ [EQA 驳回] 裁判认为不是目标！继续搜索...")
                                    # 控制底盘原地轻微旋转继续找
                                    if len(wheel_prims) > 0:
                                        for prim in wheel_prims:
                                            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                                            if "left" in prim.GetName().lower(): drive.GetTargetVelocityAttr().Set(15.0)
                                            else: drive.GetTargetVelocityAttr().Set(-15.0)
                    
                    # ====================================================
                    # 🟢 状态 2: 纯净死锁追踪 (剔除 VLM 干扰，专注底盘闭环)
                    # ====================================================
                    elif ROBOT_STATE == "TRACKING":
                        # 1. 提取局部视野 (Local ROI)
                        margin_x = int((locked_roi_box[2] - locked_roi_box[0]) * 0.3)
                        margin_y = int((locked_roi_box[3] - locked_roi_box[1]) * 0.3)
                        
                        c_xmin = max(0, int(locked_roi_box[0]) - margin_x)
                        c_ymin = max(0, int(locked_roi_box[1]) - margin_y)
                        c_xmax = min(img_w, int(locked_roi_box[2]) + margin_x)
                        c_ymax = min(img_h, int(locked_roi_box[3]) + margin_y)
                        
                        local_patch = rgb_img[c_ymin:c_ymax, c_xmin:c_xmax]
                        
                        try:
                            # 局部高速 DINO 追踪
                            res_local = requests.post(dino_tracker.server_url, json={"image_base64": dino_tracker._encode_image(local_patch), "text_prompt": task_dict.get('target_prompt', 'rock.')}, timeout=5).json()
                            
                            if res_local.get("found"):
                                local_box = res_local["boxes"][0] 
                                g_box = [c_xmin + local_box[0], c_ymin + local_box[1], c_xmin + local_box[2], c_ymin + local_box[3]]
                                # ==========================================
                                # 📸 [NEW] 客户端全景 Debug 截图功能
                                # ==========================================
                                try:
                                    # 将当前完整的高清原始画面转为 PIL 图像
                                    debug_pil = Image.fromarray(rgb_img.astype(np.uint8))
                                    draw = ImageDraw.Draw(debug_pil)
                                    
                                    # 在全图上画出这个红色追踪框
                                    draw.rectangle(g_box, outline="red", width=6)
                                    
                                    # 保存到你运行 demo2.py 的目录下
                                    debug_pil.save("debug_full_view.jpg")
                                except Exception as e:
                                    print(f"全图Debug保存失败: {e}")
                                locked_roi_box = g_box 
                                
                                # 获取极其精准的 3D 深度 (来自物理深度相机)
                                x3d, y3d, z3d = dino_tracker._get_3d_coordinates(rgb_img, g_box)
                                print(f"🔒 [静默追踪] X={x3d:.3f}m, Y={y3d:.3f}m, 深度Z={z3d:.3f}m")
                                
                                # ==========================================
                                # 🛑 核心分支 A：到达黄金抓取区，准备交接
                                # ==========================================
                                TARGET_Z_MIN = 0.35  
                                TARGET_Z_MAX = 0.65  
                                TARGET_X_TOL = 0.25  
                                
                                if TARGET_Z_MIN < z3d < TARGET_Z_MAX and abs(x3d) < TARGET_X_TOL:
                                    print("\n🛑 [到达工作空间] 目标已进入黄金抓取区！底盘紧急制动！")
                                    # 底盘电机归零，强制刹车锁死并加阻尼防溜车
                                    for prim in wheel_prims:
                                        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                                        drive.GetTargetVelocityAttr().Set(0.0)
                                        drive.GetDampingAttr().Set(100.0)
                                    
                                    # 保存最后时刻的相机相对坐标，交接状态机
                                    final_cam_pos = np.array([x3d, y3d, z3d])
                                    ROBOT_STATE = "ARRIVED"
                                    
                                # ==========================================
                                # 🚙 核心分支 B：未到达，静默追踪 (拒绝大模型干预)
                                # ==========================================
                                else:
                                    # 1. 正常驱动底盘
                                    v_x, v_yaw, _ = dino_tracker._calc_pid(g_box, img_w)
                                    # 🚨 [核心补丁] 分段降速：如果离目标不到 1.2 米了，强制开启“龟速抵近”模式
                                    if z3d > 0 and z3d < 1.2:
                                        v_x = min(v_x, 15.0)  # 限制最大前进速度 (原来可能是80)
                                        v_yaw = min(max(v_yaw, -10.0), 10.0) # 限制最大转向速度
                                        
                                    if len(wheel_prims) > 0:
                                        for prim in wheel_prims:
                                            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                                            drive.GetDampingAttr().Set(100.0) # 确保正常行驶时阻尼不卡死
                                            if "left" in prim.GetName().lower(): drive.GetTargetVelocityAttr().Set(v_x + v_yaw)
                                            else: drive.GetTargetVelocityAttr().Set(v_x - v_yaw)
                                            
                                    # 🗑️ [已删除] 彻底移除了 VLM 抽查，让底盘专心干活！
                                            
                            else:
                                print("⚠️ 局部视野丢失目标！退回全局搜索模式重新确认。")
                                ROBOT_STATE = "SEARCHING"
                                dino_tracker.last_u, dino_tracker.last_v = None, None
                        except Exception as e:
                            print(f"Tracking Error: {e}")
                    # ====================================================
                    # 🎯 状态 3: 到达交接 (解算绝对坐标，等待外部模型接管)
                    # ====================================================
                    elif ROBOT_STATE == "ARRIVED":
                        from omni.isaac.core.utils.xforms import get_world_pose
                        
                        # 使用你挂载的深度相机的真实 Prim Path
                        cam_prim_path = "/Robots/husky/jackal/base_link/mast_base/rsd455/RSD455/Camera_OmniVision_OV9782_Depth"
                        try:
                            # 获取相机在世界里的绝对位置
                            cam_world_pos, cam_world_quat = get_world_pose(cam_prim_path)
                            
                            # 坐标系翻转：Isaac 四元数 [w,x,y,z] -> Scipy [x,y,z,w]
                            r_matrix = R.from_quat([cam_world_quat[1], cam_world_quat[2], cam_world_quat[3], cam_world_quat[0]])
                            
                            # 仿射变换：计算石头在 Isaac Sim 里的上帝视角(世界)坐标
                            target_world_pos = r_matrix.apply(final_cam_pos) + cam_world_pos
                            
                            print("\n" + "="*50)
                            print(f"📡 [系统交接] 底盘导航任务完美结束！")
                            print(f"🎯 目标的绝对世界坐标为: X={target_world_pos[0]:.3f}, Y={target_world_pos[1]:.3f}, Z={target_world_pos[2]:.3f}")
                            print(f"⏸️ 底盘已休眠。请唤醒您的【抓取模型】接管机械臂！")
                            print("="*50 + "\n")
                            
                            # 切换到 DONE 状态，保持底盘锁死
                            ROBOT_STATE = "DONE"
                                
                        except Exception as e:
                            print(f"❌ 交接坐标系转换失败: {e}")
                            ROBOT_STATE = "DONE"

                    # ====================================================
                    # 💤 状态 4: 任务完成 / 待机休眠
                    # ====================================================
                    elif ROBOT_STATE == "DONE":
                        # 确保所有轮速死死咬在 0，防止仿真里发生溜车
                        for prim in wheel_prims:
                            UsdPhysics.DriveAPI.Get(prim, "angular").GetTargetVelocityAttr().Set(0.0)
                    

            step_counter += 1

    simulation_app.close()

if __name__ == "__main__":
    run()
