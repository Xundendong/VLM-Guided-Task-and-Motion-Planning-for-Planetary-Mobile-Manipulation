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

class DualLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
        self.log.write(f"\n\n{'='*60}\n🚀 终极空间指挥与容错视觉伺服启动: {run_time}\n📂 记录: {LOG_DIR}\n{'='*60}\n")
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
        1. There is a large white robotic arm in the bottom-left/center of the image. DO NOT mistake it or the rocks near it as the target.
        2. Find the correct crater. Output its bounding box (roi_box) as normalized coordinates [ymin, xmin, ymax, xmax] (values from 0.0 to 1.0).
        3. Output a highly specific ENGLISH prompt for the final rock target. e.g., 'the small rock inside the crater.'
        
        Return a STRICT JSON object exactly like this:
        {{
            "reasoning": "用中文简短分析正确的坑在哪，以及最终目标的长相",
            "roi_box": [0.3, 0.4, 0.8, 0.9],
            "target_prompt": "English phrase for the final target. e.g., 'the small rock.'"
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
# 🎯 2B. DINO 小脑追踪器 (配合 VLM 裁判)
# ==========================================
class DinoTracker:
    def __init__(self):
        self.server_url = "http://127.0.0.1:8000/dino_detect" 
        print(f"✅ [小脑代理] DINO 鹰眼接口初始化完毕 (受 VLM 裁判督导)")
        self.Kp_yaw = 0.15    
        self.Kp_dist = 0.0015 
        self.target_area = 15000 

    def _encode_image(self, rgb_array):
        img = Image.fromarray(rgb_array.astype(np.uint8))
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _calc_pid(self, box, img_w):
        """内部抽取的 PID 计算器"""
        xmin, ymin, xmax, ymax = box
        u_center = (xmin + xmax) / 2.0
        error_yaw = u_center - (img_w / 2.0) 
        
        if abs(error_yaw) < 20: turn_vel = 0.0
        else: turn_vel = error_yaw * self.Kp_yaw

        current_area = (xmax - xmin) * (ymax - ymin)
        error_dist = self.target_area - current_area
        forward_vel = max(0.0, min(error_dist * self.Kp_dist, 80.0))
        turn_vel = max(-50.0, min(turn_vel, 50.0))      
        return forward_vel, turn_vel, error_yaw

    def track_with_vlm_roi(self, rgb_image, vlm_task_dict, vlm_brain):
        """主线：在 VLM 指定的框内切图寻找。若找到，交由裁判。若失败，吃后悔药降级全局搜索并交由裁判。"""
        img_h, img_w = rgb_image.shape[:2]
        target_prompt = vlm_task_dict.get("target_prompt", "rock.")
        ymin, xmin, ymax, xmax = vlm_task_dict.get("roi_box", [0.0, 0.0, 1.0, 1.0])

        # 方案 A：VLM 坐标系高精度物理裁剪
        margin_y, margin_x = int(img_h * 0.10), int(img_w * 0.10)
        c_ymin = max(0, int(ymin * img_h) - margin_y)
        c_ymax = min(img_h, int(ymax * img_h) + margin_y)
        c_xmin = max(0, int(xmin * img_w) - margin_x)
        c_xmax = min(img_w, int(xmax * img_w) + margin_x)
        
        cropped_img = rgb_image[c_ymin:c_ymax, c_xmin:c_xmax]
        
        if cropped_img.shape[0] > 20 and cropped_img.shape[1] > 20:
            try:
                res_local = requests.post(self.server_url, json={"image_base64": self._encode_image(cropped_img), "text_prompt": target_prompt}, timeout=5).json()
                if res_local.get("found"):
                    r_xmin, r_ymin, r_xmax, r_ymax = res_local["box"]
                    g_box = [c_xmin + r_xmin, c_ymin + r_ymin, c_xmin + r_xmax, c_ymin + r_ymax]
                    
                    print(f"👀 DINO 在坑内发现疑似目标，提交 VLM 审批...")
                    if vlm_brain.verify_dino_box(rgb_image, g_box, target_prompt):
                        forward_vel, turn_vel, error_yaw = self._calc_pid(g_box, img_w)
                        return forward_vel, turn_vel
                    else:
                        print("🚫 审批被驳回，DINO 找错了。执行降级盲扫...")
            except Exception: pass

        # 方案 B：吃后悔药 (降级为全图盲扫)
        print(f"🔄 [动态回退] 全局寻找: [{target_prompt}]")
        try:
            res_global = requests.post(self.server_url, json={"image_base64": self._encode_image(rgb_image), "text_prompt": target_prompt}, timeout=5).json()
            if res_global.get("found"):
                g_box = res_global["box"]
                print(f"👀 DINO 在全局发现疑似目标，提交 VLM 审批...")
                if vlm_brain.verify_dino_box(rgb_image, g_box, target_prompt):
                    forward_vel, turn_vel, error_yaw = self._calc_pid(g_box, img_w)
                    return forward_vel, turn_vel
                else:
                    print("🚫 审批被驳回！(大概率又框住机械臂了)，原地旋转搜索。")
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
    # 🟢 双模初始化与主循环
    # ========================================================
    vlm_brain = VLMAgent()
    dino_tracker = DinoTracker()
    
    base_instruction = "你的右前方有一个陨石坑，坑的底部中间有一块岩石，请靠近它并停在它旁边。不要识别出机械臂或者坑边的岩石，目标岩石在坑内，并且比较小。如果你看不见目标，你可以适当向右转"
    current_dino_task_dict = None 

    startup_frames, step_counter = 0, 0
    VISION_INTERVAL = 15 
    
    while simulation_app.is_running():
        simulation_app.update()
        if not timeline.is_playing(): timeline.play()

        # 自检
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

        # 视觉伺服
        if timeline.is_playing() and startup_frames > 90:
            if step_counter % VISION_INTERVAL == 0: 
                rgba_data = mast_cam.get_rgba()
                
                if rgba_data is not None and rgba_data.size > 0:
                    rgb_img = rgba_data[:, :, :3]
                    
                    if current_dino_task_dict is None:
                        print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] 🧠 VLM 大脑正在进行全景空间建模...")
                        current_dino_task_dict = vlm_brain.extract_target_prompt(rgb_img, base_instruction)
                        if current_dino_task_dict:
                            print(f"🎯 VLM 下发【空间裁剪任务】: 锁定框 {current_dino_task_dict.get('roi_box')} -> 寻找 {current_dino_task_dict.get('target_prompt')}")
                            print("⚡ DINO 小脑接入，执行局部精确制导！")
                    
                    elif current_dino_task_dict is not None:
                        # 确保调用的是 DinoTracker 中的新方法 track_with_vlm_roi
                        v_x, v_yaw = dino_tracker.track_with_vlm_roi(rgb_img, current_dino_task_dict, vlm_brain)
                        if len(wheel_prims) > 0:
                            for prim in wheel_prims:
                                drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                                if "left" in prim.GetName().lower(): drive.GetTargetVelocityAttr().Set(v_x + v_yaw)
                                else: drive.GetTargetVelocityAttr().Set(v_x - v_yaw)
                        
            step_counter += 1

    simulation_app.close()

if __name__ == "__main__":
    run()