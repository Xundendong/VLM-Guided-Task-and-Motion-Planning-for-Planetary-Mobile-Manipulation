__author__ = "Antoine Richard & Customized by You (Ultimate Spatial Director & Fallback Edition)"
__license__ = "BSD-3-Clause"

import os
import sys
import datetime
import math
import numpy as np
from omegaconf import DictConfig, OmegaConf, ListConfig
import logging
import carb
import hydra

import base64
import json
import io
import requests

vendor_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor_py")
if os.path.isdir(os.path.join(vendor_py, "PIL")) and vendor_py not in sys.path:
    sys.path.append(vendor_py)

from PIL import Image
import re
from dataclasses import dataclass

from src.configurations import configFactory
from src.environments_wrappers import startSim

# ==========================================
# 📝 1. 日志与配置
# ==========================================
run_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.path.join(os.getcwd(), "logs", f"mission_{run_time}")
os.makedirs(LOG_DIR, exist_ok=True)
log_file_path = os.path.join(LOG_DIR, "mission_record.log")
custom_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("logs", exist_ok=True)
MY_PRIVATE_LOG = os.path.join("logs", f"vlm_cover_tracking_{custom_time}.txt")
def my_print(message):
    print(message) # 照常在终端显示
    # 'a' 模式代表追加，绝对不覆盖！
    with open(MY_PRIVATE_LOG, "a", encoding="utf-8") as f:
        f.write(message + "\n")
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

def quat_wxyz_normalize(q):
    q = np.array(q, dtype=np.float64)
    return q / max(np.linalg.norm(q), 1e-12)

def quat_wxyz_multiply(q1, q2):
    w1, x1, y1, z1 = np.array(q1, dtype=np.float64)
    w2, x2, y2, z2 = np.array(q2, dtype=np.float64)
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)

def quat_wxyz_apply(q, vec):
    q = quat_wxyz_normalize(q)
    v = np.array([0.0, vec[0], vec[1], vec[2]], dtype=np.float64)
    q_inv = np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)
    return quat_wxyz_multiply(quat_wxyz_multiply(q, v), q_inv)[1:]

def quat_wxyz_from_yaw(yaw_rad):
    half = 0.5 * float(yaw_rad)
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)

def usd_camera_local_to_optical(local_pos):
    """USD Camera looks along local -Z; our pinhole math uses optical +Z forward, +Y down."""
    local_pos = np.array(local_pos, dtype=np.float64)
    return np.array([local_pos[0], -local_pos[1], -local_pos[2]], dtype=np.float64)

def optical_to_usd_camera_local(optical_pos):
    optical_pos = np.array(optical_pos, dtype=np.float64)
    return np.array([optical_pos[0], -optical_pos[1], -optical_pos[2]], dtype=np.float64)

def optical_camera_to_world(cam_world_pos, cam_world_quat, optical_pos):
    local_pos = optical_to_usd_camera_local(optical_pos)
    return quat_wxyz_apply(cam_world_quat, local_pos) + np.array(cam_world_pos, dtype=np.float64)

def world_to_optical_camera(cam_world_pos, cam_world_quat, world_pos):
    cam_q = quat_wxyz_normalize(cam_world_quat)
    cam_q_inv = np.array([cam_q[0], -cam_q[1], -cam_q[2], -cam_q[3]], dtype=np.float64)
    cam_local = quat_wxyz_apply(cam_q_inv, np.array(world_pos, dtype=np.float64) - np.array(cam_world_pos, dtype=np.float64))
    return usd_camera_local_to_optical(cam_local)

def rot_matrix_to_quat_wxyz(R):
    """Convert a 3x3 rotation matrix to a wxyz quaternion.

    Uses the trace-based algorithm that is numerically stable across all
    rotation angles.  Gracefully degrades to identity when the input is
    not a proper rotation matrix.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    det = np.linalg.det(R)
    if not (0.75 < det < 1.35):
        # Not a rotation — return identity.
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / max(np.linalg.norm(q), 1e-12)


def quat_wxyz_to_rot_matrix(q):
    """Convert a wxyz quaternion to a 3x3 rotation matrix.

    The matrix form matches the ``quat_wxyz_apply`` convention used
    throughout this file (Hamilton product q * v * q⁻¹).
    """
    q = quat_wxyz_normalize(q)
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z),     2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y),       2.0 * (y * z + w * x),       1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def anygrasp_rotation_to_world_quat(R_optical, cam_world_quat):
    """Transform an AnyGrasp rotation matrix from camera optical frame to a
    world-frame wxyz quaternion.

    Frame chain (see the optical ↔ USD helpers above):
      Camera optical  (+X right, +Y **down**, +Z forward)
        → T = diag(1, −1, −1)
      USD camera local (+X right, +Y up,     +Z backward)
        → R_cam  (world rotation of the camera prim)
      World frame     → wxyz quaternion

    Returns the **full** 6-DOF orientation.  For lunar top-down grasping
    this must be projected to a strict vertical approach afterwards (see
    ``grasp_quat_to_topdown_yaw`` inside AnyGraspWristGraspExecutor).
    """
    R_opt = np.asarray(R_optical, dtype=np.float64)
    if R_opt.shape != (3, 3) or not np.all(np.isfinite(R_opt)):
        return None  # caller falls back to base_down_quat

    # 1.  Optical → USD camera local:
    #     T = diag(1, -1, -1)   and   T⁻¹ = T,  so  R_local = T @ R_opt.
    T = np.diag([1.0, -1.0, -1.0])
    R_local = T @ R_opt

    # 2.  Apply the camera's world rotation.
    R_cam = quat_wxyz_to_rot_matrix(cam_world_quat)
    R_world = R_cam @ R_local

    # 3.  Matrix → wxyz quaternion (full 6-DOF, not yet top-down-projected).
    return rot_matrix_to_quat_wxyz(R_world)


def grasp_quat_to_topdown_yaw(q_world, base_down_quat):
    """Project a world-frame grasp quaternion to a **strict top-down** orientation.

    Lunar rock sampling requires the gripper to approach vertically (world −Z).
    This function extracts only the *yaw* (rotation around world Z) that
    AnyGrasp recommends and composes it with ``base_down_quat`` so that:

    * approach  → straight down  (world −Z)
    * yaw       → from AnyGrasp  (finger closing direction in the horizontal plane)

    Parameters
    ----------
    q_world : (4,) ndarray
        AnyGrasp full world-frame wxyz quaternion.
    base_down_quat : (4,) ndarray
        The hardcoded top-down wxyz quaternion.

    Returns
    -------
    (4,) ndarray  –  unit wxyz quaternion: ``q_yaw * base_down_quat``.
    """
    R_world = quat_wxyz_to_rot_matrix(q_world)

    # ── 1.  Identify the approach axis (column closest to world ±Z) ──
    z_scores = [abs(R_world[2, i]) for i in range(3)]
    approach_col = int(np.argmax(z_scores))

    # ── 2.  Among the two remaining columns, pick the more horizontal one
    #        as the closing-direction reference for yaw. ──
    other = [i for i in range(3) if i != approach_col]
    close_col = other[0] if abs(R_world[2, other[0]]) <= abs(R_world[2, other[1]]) else other[1]

    # ── 3.  Yaw = angle of the closing axis projected onto the XY plane ──
    yaw = math.atan2(float(R_world[1, close_col]), float(R_world[0, close_col]))

    # ── 4.  Compose: yaw around Z, then base_down (tool flange → down) ──
    q_yaw = quat_wxyz_from_yaw(yaw)
    return quat_wxyz_normalize(quat_wxyz_multiply(q_yaw, base_down_quat))


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
                    my_print(f"🧠 [VLM 空间推理]: {action_dict.get('reasoning', '无')}")
                    
                    roi_box = action_dict.get("roi_box", [0.0, 0.0, 1.0, 1.0])
                    # 修复 Qwen-VL 7B 可能输出 0-1000 绝对坐标的问题
                    if any(v > 1.1 for v in roi_box):
                        roi_box = [max(0.0, min(1.0, v / 1000.0)) for v in roi_box]
                        
                    target = action_dict.get("target_prompt", "the rock.")
                    if not target.endswith("."): target += "."
                    
                    return {"roi_box": roi_box, "target_prompt": target}
            return {"roi_box": [0.0, 0.0, 1.0, 1.0], "target_prompt": "the rock."}
        except Exception as e:
            my_print(f"❌ [VLM 大脑异常]: {e}")
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
                    my_print(f"⚖️ [VLM 裁判庭]: {res_dict.get('reasoning', '无')} -> 判决: {'✅ 通过' if is_correct else '❌ 驳回 (判定为干扰物)'}")
                    return is_correct
            return False
        except Exception as e:
            my_print(f"⚠️ [VLM 裁判异常]: {e}")
            return False

    def verify_wrist_candidate(self, mast_reference_rgb, wrist_rgb, wrist_box, target_prompt):
        """比较高杆目标参考图和手腕候选框，防止把车体/机械臂误识别为岩石。"""
        if mast_reference_rgb is None or wrist_rgb is None or wrist_box is None:
            return False, "missing_reference_or_wrist_image"

        ref_img = Image.fromarray(mast_reference_rgb.astype(np.uint8)).resize((320, 240))
        wrist_img = Image.fromarray(wrist_rgb.astype(np.uint8)).resize((640, 480))
        draw = ImageDraw.Draw(wrist_img)
        h, w = wrist_rgb.shape[:2]
        sx, sy = 640.0 / max(w, 1), 480.0 / max(h, 1)
        x0, y0, x1, y1 = [float(v) for v in wrist_box]
        draw.rectangle([x0 * sx, y0 * sy, x1 * sx, y1 * sy], outline="red", width=6)

        canvas = Image.new("RGB", (960, 480), (20, 20, 20))
        canvas.paste(ref_img, (0, 120))
        canvas.paste(wrist_img, (320, 0))
        cdraw = ImageDraw.Draw(canvas)
        cdraw.text((12, 92), "MAST TARGET REFERENCE", fill=(255, 255, 255))
        cdraw.text((332, 12), "WRIST CANDIDATE IN RED BOX", fill=(255, 255, 255))

        buffered = io.BytesIO()
        canvas.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        prompt = f"""
        You are a strict visual safety checker for a lunar sample collection robot.

        The left panel is the target reference crop from the mast camera.
        The right panel is the wrist camera image. A thick RED BOX marks the candidate detected by DINO.
        The intended target is: "{target_prompt}".

        Decide whether the RED BOX contains the same kind of compact loose lunar rock as the left reference.

        Hard rejection rules:
        - Return false if the red box contains the rover chassis, yellow/white/black robot body, robotic arm, gripper, fingers, wheel, shadow, ground texture, or a huge image region.
        - Return false if the candidate is not a compact standalone rock.
        - Return false if you are uncertain.

        Return STRICT JSON:
        {{
            "reasoning": "用中文说明红框里是岩石还是机器人自体/地面/阴影",
            "is_same_target": true or false
        }}
        """
        try:
            response = requests.post(self.server_url, json={"image_base64": img_str, "instruction": prompt}, timeout=30)
            if response.status_code == 200:
                text = response.json().get("result", "")
                clean_text = text.replace("```json", "").replace("```", "").strip()
                json_match = re.search(r'\{.*?\}', clean_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    ok = bool(result.get("is_same_target", False))
                    reason = result.get("reasoning", "无")
                    my_print(f"🧑‍⚖️ [WristVision VLM复核]: {reason} -> {'通过' if ok else '拒绝'}")
                    return ok, reason
        except Exception as e:
            my_print(f"⚠️ [WristVision VLM复核异常]: {e}")
        return False, "vlm_check_failed"

    def propose_grasp_plan(self, rgb_image, box, target_prompt):
        """VLM 负责语义抓取决策；深度和 IK 负责把它落成物理动作。"""
        img = Image.fromarray(rgb_image.astype(np.uint8))
        draw = ImageDraw.Draw(img)
        xmin, ymin, xmax, ymax = [int(v) for v in box]
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=6)

        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        prompt = f"""
        You are a zero-shot robotic grasp director for a lunar mobile manipulator.
        The target is: "{target_prompt}". A thick RED BOX marks the object to collect.

        Context:
        - The object is a small lunar rock lying inside or near a crater.
        - The mast RGB-D camera sees the rock; the wrist camera may not see it because the terrain is sloped.
        - You must choose a robust grasp point visible from this mast camera.
        - DO NOT choose rover arm parts, gripper parts, shadows, crater soil, or the red box border.

        Return STRICT JSON only:
        {{
            "reasoning": "用中文简短说明抓取点为什么稳，是否避开了月壤/阴影/机械臂",
            "target_confirmed": true,
            "grasp_pixel": [u, v],
            "approach": "top_down",
            "grasp_type": "pinch",
            "preferred_wrist_yaw_deg": 0,
            "gripper_width_m": 0.055,
            "confidence": 0.0,
            "retry_hint": "move the wrist slightly front/back/left/right before descending"
        }}

        Rules:
        - grasp_pixel must be an absolute image coordinate [u, v] inside the RED BOX.
        - approach must be "top_down" unless the rock is visibly better grasped from the front.
        - gripper_width_m must be between 0.025 and 0.100 for a small lunar rock.
        - target_confirmed must be false if the red box contains the rover arm, gripper, soil only, or a crater without a rock.
        """
        fallback = {
            "target_confirmed": True,
            "grasp_pixel": [0.5 * (xmin + xmax), ymin + 0.45 * (ymax - ymin)],
            "approach": "top_down",
            "grasp_type": "pinch",
            "preferred_wrist_yaw_deg": 0.0,
            "gripper_width_m": 0.055,
            "confidence": 0.0,
            "retry_hint": "fallback center grasp",
        }
        try:
            payload = {"image_base64": img_str, "instruction": prompt}
            response = requests.post(self.server_url, json=payload, timeout=30)
            if response.status_code == 200:
                text = response.json().get("result", "")
                clean_text = text.replace("```json", "").replace("```", "").strip()
                json_match = re.search(r'\{.*?\}', clean_text, re.DOTALL)
                if json_match:
                    plan = json.loads(json_match.group())
                    pixel = plan.get("grasp_pixel", fallback["grasp_pixel"])
                    if not (isinstance(pixel, list) and len(pixel) == 2):
                        pixel = fallback["grasp_pixel"]
                    my_print(f"🧠 [VLM 抓取规划]: {plan.get('reasoning', '无')} -> pixel={pixel}")
                    return {
                        "target_confirmed": bool(plan.get("target_confirmed", True)),
                        "grasp_pixel": pixel,
                        "approach": plan.get("approach", "top_down"),
                        "grasp_type": plan.get("grasp_type", "pinch"),
                        "preferred_wrist_yaw_deg": float(plan.get("preferred_wrist_yaw_deg", 0.0)),
                        "gripper_width_m": float(plan.get("gripper_width_m", 0.055)),
                        "confidence": float(plan.get("confidence", 0.0)),
                        "retry_hint": plan.get("retry_hint", ""),
                    }
        except Exception as e:
            my_print(f"⚠️ [VLM 抓取规划异常]: {e}")
        return fallback

    def verify_grasp_success(self, rgb_image, target_prompt):
        """抓取后让 VLM 判断岩石是否被夹爪抬起；失败时给出恢复原因。"""
        img_str = self._encode_image(rgb_image)
        prompt = f"""
        You are the post-grasp inspector for a lunar sample collection robot.
        The intended target is "{target_prompt}".

        Inspect whether a small rock sample is visibly held between the gripper fingers and lifted above the crater floor.
        Be very strict. Return false if the image only shows the robot body, gripper, ground, shadow, or a rock still on the ground.
        Return STRICT JSON:
        {{
            "reasoning": "用中文说明是否抓到了岩石，是否只是夹到了土壤/空气/机械臂",
            "is_grasped": true,
            "failure_mode": "none"
        }}

        failure_mode must be one of: "none", "missed", "slipped", "grasped_soil", "occluded", "uncertain".
        """
        try:
            payload = {"image_base64": img_str, "instruction": prompt}
            response = requests.post(self.server_url, json=payload, timeout=30)
            if response.status_code == 200:
                text = response.json().get("result", "")
                clean_text = text.replace("```json", "").replace("```", "").strip()
                json_match = re.search(r'\{.*?\}', clean_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    ok = bool(result.get("is_grasped", False))
                    my_print(f"🔎 [VLM 抓取验收]: {result.get('reasoning', '无')} -> {'成功' if ok else '失败'} ({result.get('failure_mode', 'uncertain')})")
                    return ok, result.get("failure_mode", "uncertain")
        except Exception as e:
            my_print(f"⚠️ [VLM 抓取验收异常]: {e}")
        return False, "unchecked"

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
        
        my_print(f"✅ [小脑代理] DINO 鹰眼接口初始化完毕 (开启防抖记忆跟踪 & 物理测距)")
        
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
        
        # 💡 [恢复基础灵敏度]
        self.Kp_yaw = 0.15 
        
        # =======================================================
        # 🚨 [终极解法：暴力坦克掉头模式 (In-Place Tank Turn)]
        # =======================================================
        # 1. 只有极其微小的偏差，才允许边走边微调
        if abs(error_yaw) < 20: 
            turn_vel = 0.0
        else: 
            turn_vel = error_yaw * self.Kp_yaw
            
        current_area = (xmax - xmin) * (ymax - ymin)
        error_dist = self.target_area - current_area
        base_forward_vel = max(0.0, min(error_dist * 0.0025, 30.0)) 

        # 2. 极其严苛的对准逻辑
        if abs(error_yaw) > 100:  # 💡 阈值收紧到 100 像素！
            # 绝对不允许往前走一毫米！
            forward_vel = 0.0    
            
            # 🚨 核心爆发力：强行赋予能撕裂月壤摩擦力的起步扭矩！
            # 之前 18 不够，我们直接给到 50！
            min_breakout_turn = 50.0 if error_yaw > 0 else -50.0
            if abs(turn_vel) < abs(min_breakout_turn):
                turn_vel = min_breakout_turn
                
            my_print(f"🔄 [坦克掉头] 偏差 {error_yaw:.1f}px, 满功率原地转向中...")
            
        elif abs(error_yaw) > 40:
            # 轻微偏差，降速 70%，边走边小心微调
            forward_vel = base_forward_vel * 0.3 
            my_print(f"🔄 [边走边调] 偏差 {error_yaw:.1f}px...")
        else:
            # 准星重合，全速突击！
            forward_vel = base_forward_vel
            
        # 💡 释放野兽：转向限幅拉满到 80！
        turn_vel = max(-80.0, min(turn_vel, 80.0))      
        
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
            my_print(f"⚠️ [硬件测距异常]: {e}")
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
                        my_print(f"👀 首次发现目标，提交 VLM 审批...")
                        
                        if not vlm_brain.verify_dino_box(rgb_image, g_box, target_prompt):
                            self.last_u, self.last_v = None, None # 审批失败，清除记忆
                            return 0.0, 15.0 # 强制转圈
                    
                    # 调用改写后的物理测距
                    x3d, y3d, z3d = self._get_3d_coordinates(rgb_image, g_box)
                    my_print(f"📍 [稳定锁定] X={x3d:.3f}m, Y={y3d:.3f}m, 物理深度Z={z3d:.3f}m")
                    
                    forward_vel, turn_vel, error_yaw = self._calc_pid(g_box, img_w)
                    return forward_vel, turn_vel
            except Exception: pass

        # 🔴 方案 B：全局盲扫
        my_print(f"🔄 [丢失目标] 记忆清空，全局重新寻找: [{target_prompt}]")
        self.last_u, self.last_v = None, None # 一旦进入盲扫，必须清除旧记忆
        try:
            res_global = requests.post(self.server_url, json={"image_base64": self._encode_image(rgb_image), "text_prompt": target_prompt}, timeout=5).json()
            if res_global.get("found"):
                g_boxes = res_global["boxes"]
                g_box = self._find_best_box_by_memory(g_boxes)
                
                my_print(f"👀 全局扫描发现目标，提交 VLM 审批...")
                if vlm_brain.verify_dino_box(rgb_image, g_box, target_prompt):
                    # 调用改写后的物理测距
                    x3d, y3d, z3d = self._get_3d_coordinates(rgb_image, g_box)
                    my_print(f"📍 [全局稳定锁定] X={x3d:.3f}m, Y={y3d:.3f}m, 物理深度Z={z3d:.3f}m")
                    forward_vel, turn_vel, error_yaw = self._calc_pid(g_box, img_w)
                    return forward_vel, turn_vel
                else:
                    self.last_u, self.last_v = None, None
                    return 0.0, 15.0
            else:
                return 0.0, 15.0
        except Exception: return 0.0, 0.0


@dataclass
class GraspPose:
    position: np.ndarray
    orientation: np.ndarray
    width: float = 0.055
    score: float = 0.0
    source: str = "vlm_rgbd"
    pixel: object = None


@dataclass
class GraspCandidate:
    label: str
    scan_pose: GraspPose
    pre_pose: GraspPose
    grasp_pose: GraspPose
    lift_pose: GraspPose
    score: float = 0.0


class VLMGraspPlanner:
    """高杆相机可见、手眼相机不可见时的 VLM+RGB-D 抓取规划器。"""

    def __init__(self, vlm_brain, depth_cam, fx=320.0, fy=320.0, cx=320.0, cy=240.0):
        self.vlm_brain = vlm_brain
        self.depth_cam = depth_cam
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.scan_z = 0.30
        self.pre_z = 0.22
        self.grasp_z = 0.12
        self.lift_z = 0.34
        my_print("⚠️ [VLM-Grasp] 旧RGB-D反投影模块仅保留备用；demo5主流程使用WristVision无坐标抓取。")

    def _median_depth(self, depth_map, u, v, roi_box=None):
        h, w = depth_map.shape[:2]
        u = int(np.clip(u, 0, w - 1))
        v = int(np.clip(v, 0, h - 1))
        for win in [4, 8, 14]:
            u0, u1 = max(0, u - win), min(w, u + win + 1)
            v0, v1 = max(0, v - win), min(h, v + win + 1)
            patch = depth_map[v0:v1, u0:u1]
            valid = patch[np.isfinite(patch) & (patch > 0.15) & (patch < 2.5)]
            if len(valid) > 0:
                return float(np.median(valid)), u, v

        if roi_box is not None:
            x0, y0, x1, y1 = [int(x) for x in roi_box]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            roi_depth = depth_map[y0:y1, x0:x1]
            valid = roi_depth[np.isfinite(roi_depth) & (roi_depth > 0.15) & (roi_depth < 2.5)]
            if len(valid) > 0:
                u = int(0.5 * (x0 + x1))
                v = int(y0 + 0.45 * (y1 - y0))
                return float(np.median(valid)), u, v
        raise ValueError("no valid depth for grasp point")

    def _pixel_to_camera(self, rgb_img, pixel, fallback_cam_pos, roi_box):
        depth_map = self.depth_cam.get_depth()
        if depth_map is None or depth_map.size == 0:
            my_print("⚠️ [VLM-Grasp] 深度图为空，退回导航交接点。")
            return np.array(fallback_cam_pos, dtype=np.float64)

        img_h, img_w = rgb_img.shape[:2]
        u = int(np.clip(pixel[0], 0, img_w - 1))
        v = int(np.clip(pixel[1], 0, img_h - 1))
        try:
            z_c, u, v = self._median_depth(depth_map, u, v, roi_box=roi_box)
            x_c = (u - self.cx) * z_c / self.fx
            y_c = (v - self.cy) * z_c / self.fy
            return np.array([x_c, y_c, z_c], dtype=np.float64)
        except Exception as e:
            my_print(f"⚠️ [VLM-Grasp] VLM像素深度不可用，退回导航交接点: {e}")
            return np.array(fallback_cam_pos, dtype=np.float64)

    def plan_camera_grasp_point(self, rgb_img, roi_box, final_cam_pos, target_prompt):
        plan = self.vlm_brain.propose_grasp_plan(rgb_img, roi_box, target_prompt)
        if not plan.get("target_confirmed", True):
            my_print("❌ [VLM-Grasp] VLM认为目标框不是可抓取岩石，拒绝抓取。")
            return None

        x0, y0, x1, y1 = [int(v) for v in roi_box]
        u, v = plan.get("grasp_pixel", [0.5 * (x0 + x1), y0 + 0.45 * (y1 - y0)])
        if not (x0 <= u <= x1 and y0 <= v <= y1):
            u = 0.5 * (x0 + x1)
            v = y0 + 0.45 * (y1 - y0)
            my_print(f"⚠️ [VLM-Grasp] VLM抓取点落在目标框外，强制拉回ROI: pixel={[u, v]}")

        cam_pos = self._pixel_to_camera(rgb_img, [u, v], final_cam_pos, roi_box)
        width = float(np.clip(plan.get("gripper_width_m", 0.055), 0.025, 0.100))
        score = float(plan.get("confidence", 0.0))
        return GraspPose(
            position=cam_pos,
            orientation=np.array([0.0, 0.7071, 0.7071, 0.0], dtype=np.float64),
            width=width,
            score=score,
            source="vlm_pixel_rgbd",
            pixel=np.array([u, v], dtype=np.float64),
        )

    def camera_to_world_pose(self, grasp_cam, cam_world_pos, cam_world_quat):
        world_pos = optical_camera_to_world(cam_world_pos, cam_world_quat, grasp_cam.position)
        return GraspPose(
            position=world_pos,
            orientation=grasp_cam.orientation,
            width=grasp_cam.width,
            score=grasp_cam.score,
            source=grasp_cam.source,
        )

    def _top_down_orientation(self, yaw_deg):
        # Keep the exact wrist-down convention used by run.py:
        # base_down_quat = [0, 0.7071, 0.7071, 0] in wxyz order.
        base = np.array([0.0, 0.7071, 0.7071, 0.0], dtype=np.float64)
        yaw = quat_wxyz_from_yaw(math.radians(float(yaw_deg)))
        return quat_wxyz_normalize(quat_wxyz_multiply(yaw, base))

    def build_arm_scan_candidates(self, grasp_world, robot_world_pos):
        """围绕目标表面点生成末端前后左右主动扫描候选。"""
        target = np.array(grasp_world.position, dtype=np.float64)
        robot_xy = np.array(robot_world_pos[:2], dtype=np.float64)
        target_xy = np.array(target[:2], dtype=np.float64)
        toward_robot = robot_xy - target_xy
        norm = np.linalg.norm(toward_robot)
        if norm < 1e-6:
            toward_robot = np.array([1.0, 0.0], dtype=np.float64)
        else:
            toward_robot = toward_robot / norm
        lateral = np.array([-toward_robot[1], toward_robot[0]], dtype=np.float64)

        offsets = [
            ("center", np.array([0.0, 0.0], dtype=np.float64), 1.00),
            ("front", toward_robot * 0.045, 0.92),
            ("back", -toward_robot * 0.045, 0.80),
            ("left", lateral * 0.045, 0.86),
            ("right", -lateral * 0.045, 0.86),
            ("front_left", toward_robot * 0.035 + lateral * 0.035, 0.78),
            ("front_right", toward_robot * 0.035 - lateral * 0.035, 0.78),
        ]
        yaws = [0.0, 20.0, -20.0]

        candidates = []
        for name, offset_xy, offset_score in offsets:
            for yaw in yaws:
                orient = self._top_down_orientation(yaw)
                p = target.copy()
                p[0] += offset_xy[0]
                p[1] += offset_xy[1]
                surface_z = max(float(target[2]), 0.03)

                scan = p.copy(); scan[2] = surface_z + self.scan_z
                pre = p.copy(); pre[2] = surface_z + self.pre_z
                grasp = p.copy(); grasp[2] = surface_z + self.grasp_z
                lift = p.copy(); lift[2] = surface_z + self.lift_z

                score = offset_score - abs(yaw) * 0.002 + grasp_world.score * 0.05
                label = f"{name}_yaw{yaw:+.0f}"
                candidates.append(
                    GraspCandidate(
                        label=label,
                        scan_pose=GraspPose(scan, orient, grasp_world.width, score, "active_scan"),
                        pre_pose=GraspPose(pre, orient, grasp_world.width, score, "pre_grasp"),
                        grasp_pose=GraspPose(grasp, orient, grasp_world.width, score, "grasp"),
                        lift_pose=GraspPose(lift, orient, grasp_world.width, score, "lift"),
                        score=score,
                    )
                )
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def filter_reachable_candidates(self, candidates, ee_solver, limit=6):
        reachable = []
        for cand in candidates:
            try:
                _, ok_scan = ee_solver.compute_inverse_kinematics(cand.scan_pose.position, cand.scan_pose.orientation)
                _, ok_pre = ee_solver.compute_inverse_kinematics(cand.pre_pose.position, cand.pre_pose.orientation)
                _, ok_grasp = ee_solver.compute_inverse_kinematics(cand.grasp_pose.position, cand.grasp_pose.orientation)
                my_print(
                    f"🧪 [ArmScan IK] {cand.label}: scan={ok_scan}, pre={ok_pre}, grasp={ok_grasp}, "
                    f"grasp=({cand.grasp_pose.position[0]:.3f},{cand.grasp_pose.position[1]:.3f},{cand.grasp_pose.position[2]:.3f})"
                )
                if ok_scan and ok_pre and ok_grasp:
                    reachable.append(cand)
                    if len(reachable) >= limit:
                        break
            except Exception as e:
                my_print(f"⚠️ [ArmScan IK异常] {cand.label}: {e}")
        return reachable

    def build_relative_arm_scan_candidates(self, ee_pos, grasp_cam):
        """使用 run.py 同款相对末端控制思路，避免错误相机外参把目标送到不可达世界点。"""
        ee_pos = np.array(ee_pos, dtype=np.float64)
        cam_pos = np.array(grasp_cam.position, dtype=np.float64)

        # 图像右侧对应机械臂坐标 -Y，图像左侧对应 +Y；只作为小偏移提示。
        lateral_hint = float(np.clip(-cam_pos[0] * 0.35, -0.10, 0.10))
        forward_offsets = [0.08, 0.12, 0.16, 0.20, 0.24, 0.28]
        lateral_offsets = [
            lateral_hint,
            lateral_hint + 0.04,
            lateral_hint - 0.04,
            lateral_hint + 0.08,
            lateral_hint - 0.08,
        ]
        descend_offsets = [0.10, 0.16, 0.22, 0.28, 0.34]
        yaws = [0.0, 20.0, -20.0]

        candidates = []
        for fwd in forward_offsets:
            for lat in lateral_offsets:
                for descend in descend_offsets:
                    for yaw in yaws:
                        orient = self._top_down_orientation(yaw)
                        scan = ee_pos + np.array([fwd * 0.55, lat, 0.05], dtype=np.float64)
                        pre = ee_pos + np.array([fwd, lat, -max(descend - 0.10, 0.02)], dtype=np.float64)
                        grasp = ee_pos + np.array([fwd + 0.02, lat, -descend], dtype=np.float64)
                        lift = ee_pos + np.array([fwd * 0.75, lat, 0.12], dtype=np.float64)

                        score = 1.0 - abs(lat - lateral_hint) * 2.0 - abs(yaw) * 0.002 - abs(fwd - 0.18) * 0.8
                        label = f"rel_f{fwd:.2f}_y{lat:+.2f}_d{descend:.2f}_yaw{yaw:+.0f}"
                        candidates.append(
                            GraspCandidate(
                                label=label,
                                scan_pose=GraspPose(scan, orient, grasp_cam.width, score, "relative_active_scan"),
                                pre_pose=GraspPose(pre, orient, grasp_cam.width, score, "relative_pre_grasp"),
                                grasp_pose=GraspPose(grasp, orient, grasp_cam.width, score, "relative_grasp"),
                                lift_pose=GraspPose(lift, orient, grasp_cam.width, score, "relative_lift"),
                                score=score,
                            )
                        )

        candidates.sort(key=lambda c: c.score, reverse=True)
        my_print(
            f"🦾 [ArmScan 相对规划] 以当前末端为原点生成候选: ee=({ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f}), "
            f"lateral_hint={lateral_hint:+.3f}, total={len(candidates)}"
        )
        return candidates


class RobotiqGripperDriver:
    def __init__(self, robot, action_cls=None):
        self.robot = robot
        self.action_cls = action_cls
        self.close_effort = float(os.environ.get("OMNILRS_GRIPPER_CLOSE_EFFORT", "0.0"))
        self.hold_effort = float(os.environ.get("OMNILRS_GRIPPER_HOLD_EFFORT", "0.0"))
        self.drive_max_force = float(os.environ.get("OMNILRS_GRIPPER_DRIVE_MAX_FORCE", "0.0"))
        self.drive_stiffness = float(os.environ.get("OMNILRS_GRIPPER_DRIVE_STIFFNESS", "0.0"))
        self.drive_damping = float(os.environ.get("OMNILRS_GRIPPER_DRIVE_DAMPING", "0.0"))
        effort_requested = os.environ.get("OMNILRS_GRIPPER_USE_EFFORT", "0").lower() not in ["0", "false", "no", "off"]
        effort_unlocked = os.environ.get("OMNILRS_GRIPPER_EXPERIMENTAL_EFFORT", "0").lower() not in ["0", "false", "no", "off"]
        self.enable_effort = effort_requested and effort_unlocked
        drive_requested = os.environ.get("OMNILRS_GRIPPER_CONFIGURE_DRIVE", "0").lower() not in ["0", "false", "no", "off"]
        drive_unlocked = os.environ.get("OMNILRS_GRIPPER_EXPERIMENTAL_DRIVE", "0").lower() not in ["0", "false", "no", "off"]
        self.configure_drive = drive_requested and drive_unlocked
        force_joint_names = os.environ.get("OMNILRS_GRIPPER_FORCE_JOINTS", "finger_joint")
        self.force_joint_names = [name.strip().lower() for name in force_joint_names.split(",") if name.strip()]
        self._effort_warning_printed = False
        self.joint_indices = [
            idx for idx, name in enumerate(robot.dof_names)
            if any(k in name.lower() for k in ["finger", "knuckle"])
        ]
        self.force_joint_indices = [
            idx for idx, name in enumerate(robot.dof_names)
            if name.lower() in self.force_joint_names
        ]
        if not self.force_joint_indices and self.joint_indices:
            self.force_joint_indices = [self.joint_indices[0]]
        my_print(f"✅ [Robotiq] 识别夹爪关节数量: {len(self.joint_indices)}")
        if self.configure_drive:
            self._configure_gripper_drives()
        else:
            my_print("🟢 [Robotiq] 使用夹爪原始 drive 参数；未启用额外 drive 强化。")
        if self.enable_effort:
            active_force_names = [self.robot.dof_names[idx] for idx in self.force_joint_indices]
            my_print(
                f"💪 [Robotiq] 已启用主驱动关节夹持力: force_joints={active_force_names}, "
                f"close_effort={self.close_effort:.1f}, hold_effort={self.hold_effort:.1f}"
            )
        elif effort_requested:
            my_print("🟢 [Robotiq] 已忽略 effort 请求；当前只用闭合宽度调夹持，不启用 joint_efforts。")
        if drive_requested and not self.configure_drive:
            my_print("🟢 [Robotiq] 已忽略 drive 强化请求；当前只用闭合宽度调夹持，不改 USD drive。")

    def _configure_gripper_drives(self):
        if not self.force_joint_indices:
            return
        try:
            import omni.usd
            from pxr import UsdPhysics

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                return
            joint_names = {self.robot.dof_names[idx].lower() for idx in self.force_joint_indices}
            configured = 0
            for prim in stage.Traverse():
                prim_name = prim.GetName().lower()
                prim_path = prim.GetPath().pathString.lower()
                prim_type = str(prim.GetTypeName()).lower()
                if "joint" not in prim_name and "joint" not in prim_type:
                    continue
                if prim_name not in joint_names and not any(joint_name in prim_path for joint_name in joint_names):
                    continue
                drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                if not drive:
                    drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                if self.drive_max_force > 0.0:
                    drive.GetMaxForceAttr().Set(self.drive_max_force)
                if self.drive_stiffness > 0.0:
                    drive.GetStiffnessAttr().Set(self.drive_stiffness)
                if self.drive_damping > 0.0:
                    drive.GetDampingAttr().Set(self.drive_damping)
                configured += 1
            my_print(
                f"💪 [Robotiq] 主驱动关节 drive 已增强: joints={configured}/{len(self.force_joint_indices)}, "
                f"maxForce={self.drive_max_force:.0f}, stiffness={self.drive_stiffness:.0f}, "
                f"damping={self.drive_damping:.0f}, close_effort={self.close_effort:.1f}, "
                f"hold_effort={self.hold_effort:.1f}, effort={int(self.enable_effort)}"
            )
        except Exception as e:
            my_print(f"⚠️ [Robotiq] 夹爪驱动力配置失败，继续使用默认驱动: {e}")

    def _joint_close_multiplier(self, idx):
        joint_name = self.robot.dof_names[idx].lower()
        multiplier = 1.0
        if "right" in joint_name:
            multiplier *= -1.0
        if "inner_finger" in joint_name and "knuckle" not in joint_name:
            multiplier *= -1.0
        return multiplier

    def _apply_force_efforts(self, force_efforts):
        if not force_efforts:
            return
        efforts_np = np.array(force_efforts, dtype=np.float64)
        if hasattr(self.robot, "set_joint_efforts"):
            try:
                self.robot.set_joint_efforts(efforts_np, joint_indices=self.force_joint_indices)
                return
            except Exception as e:
                if not self._effort_warning_printed:
                    my_print(f"⚠️ [Robotiq] set_joint_efforts 失败，尝试 ArticulationAction effort: {e}")
                    self._effort_warning_printed = True
        if self.action_cls is not None:
            try:
                action = self.action_cls(joint_efforts=efforts_np, joint_indices=self.force_joint_indices)
                self.robot.apply_action(action)
            except TypeError:
                if not self._effort_warning_printed:
                    my_print("⚠️ [Robotiq] 当前 ArticulationAction 不支持 joint_efforts，主驱动关节 effort 未生效。")
                    self._effort_warning_printed = True

    def command(self, opening, effort=None):
        if not self.joint_indices:
            return
        close_ratio = float(np.clip(1.0 - opening / 0.14, 0.0, 1.0))
        target_val = close_ratio * 0.80
        target_positions = []
        force_efforts = []
        for idx in self.joint_indices:
            multiplier = self._joint_close_multiplier(idx)
            target_positions.append(target_val * multiplier)
        if effort is not None and self.enable_effort and abs(float(effort)) > 0.0:
            for idx in self.force_joint_indices:
                force_efforts.append(float(effort) * self._joint_close_multiplier(idx))

        if self.action_cls is not None:
            target_positions_np = np.array(target_positions, dtype=np.float64)
            action = self.action_cls(joint_positions=target_positions_np, joint_indices=self.joint_indices)
            self.robot.apply_action(action)
            self._apply_force_efforts(force_efforts)
        else:
            current = self.robot.get_joint_positions()
            if current is None:
                return
            full_target = np.array(current, dtype=np.float64)
            for idx, pos in zip(self.joint_indices, target_positions):
                full_target[idx] = pos
            self.robot.set_joint_positions(full_target)
            self._apply_force_efforts(force_efforts)

    def open(self):
        self.command(0.14, effort=None)

    def close(self, width=0.025):
        self.command(width, effort=self.close_effort)

    def hold(self, width=0.025):
        self.command(width, effort=self.hold_effort)


class ActiveArmScanGraspExecutor:
    """先让末端围绕目标前后左右锁定，再执行下探夹取。"""

    def __init__(self, robot, ee_solver, gripper):
        self.robot = robot
        self.ee_solver = ee_solver
        self.gripper = gripper
        self.state = "IDLE"
        self.frame_count = 0
        self.candidates = []
        self.scan_index = 0
        self.active = None

    def start(self, candidates):
        self.candidates = list(candidates)
        self.scan_index = 0
        self.active = self.candidates[0] if self.candidates else None
        self.frame_count = 0
        self.state = "OPEN" if self.active is not None else "FAILED"
        if self.active is not None:
            my_print(f"🦾 [ArmScan] 开始末端主动锁定，候选数量={len(self.candidates)}，首选={self.active.label}")

    def _go_to(self, pose):
        action, success = self.ee_solver.compute_inverse_kinematics(pose.position, pose.orientation)
        if success:
            self.robot.apply_action(action)
        return bool(success)

    def _select_next_candidate(self):
        self.scan_index += 1
        if self.scan_index >= len(self.candidates):
            return False
        self.active = self.candidates[self.scan_index]
        self.frame_count = 0
        my_print(f"↪️ [ArmScan] 切换候选: {self.active.label}")
        return True

    def step(self):
        if self.state in ["IDLE", "DONE", "FAILED"]:
            return self.state

        if self.state == "OPEN":
            self.gripper.open()
            self.state = "SCAN"
            self.frame_count = 0

        elif self.state == "SCAN":
            ok = self._go_to(self.active.scan_pose)
            self.frame_count += 1
            if self.frame_count == 1:
                my_print(f"🔭 [ArmScan] 末端试探位: {self.active.label}")
            if not ok and not self._select_next_candidate():
                self.state = "FAILED"
            elif self.frame_count > 35:
                if self.scan_index + 1 < min(len(self.candidates), 5):
                    self._select_next_candidate()
                else:
                    self.active = self.candidates[0]
                    self.state = "PRE_GRASP"
                    self.frame_count = 0
                    my_print(f"🎯 [ArmScan] 锁定候选: {self.active.label}，开始抓取下探。")

        elif self.state == "PRE_GRASP":
            ok = self._go_to(self.active.pre_pose)
            self.frame_count += 1
            if self.frame_count > 70:
                self.state = "DESCEND" if ok else "FAILED"
                self.frame_count = 0

        elif self.state == "DESCEND":
            ok = self._go_to(self.active.grasp_pose)
            self.frame_count += 1
            if self.frame_count > 80:
                self.state = "CLOSE" if ok else "FAILED"
                self.frame_count = 0

        elif self.state == "CLOSE":
            self.gripper.close(self.active.grasp_pose.width * 0.40)
            self.frame_count += 1
            if self.frame_count > 45:
                self.state = "LIFT"
                self.frame_count = 0

        elif self.state == "LIFT":
            ok = self._go_to(self.active.lift_pose)
            self.frame_count += 1
            if self.frame_count > 90:
                self.state = "DONE" if ok else "FAILED"

        return self.state


class MastVisualServoGraspExecutor:
    """高杆相机闭环视觉伺服抓取：夹爪中心投影对齐目标像素后再下探夹取。"""

    def __init__(
        self,
        robot,
        ee_solver,
        gripper,
        cam_prim_path,
        get_world_pose_fn,
        fx=320.0,
        fy=320.0,
        cx=320.0,
        cy=240.0,
        stage=None,
        robot_root_path="/Robots/husky/jackal",
    ):
        self.robot = robot
        self.ee_solver = ee_solver
        self.gripper = gripper
        self.cam_prim_path = cam_prim_path
        self.get_world_pose = get_world_pose_fn
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.stage = stage
        self.robot_root_path = robot_root_path
        self.tool_point_paths = self._discover_tool_point_paths()
        self.state = "IDLE"
        self.frame_count = 0
        self.target_pixel = None
        self.target_cam_pos = None
        self.grasp_width = 0.055
        self.start_ee_pos = None
        self.descent_limit = 0.26
        self.base_down_quat = np.array([0.0, 0.7071, 0.7071, 0.0], dtype=np.float64)

    def _discover_tool_point_paths(self):
        if self.stage is None:
            return []
        fingertip_paths = []
        tool0_paths = []
        flange_paths = []
        for prim in self.stage.Traverse():
            path = prim.GetPath().pathString
            if self.robot_root_path not in path:
                continue
            name = prim.GetName().lower()
            lower_path = path.lower()
            if ("fingertip" in name or "fingertip" in lower_path) and "material" not in lower_path and "finger" in lower_path:
                fingertip_paths.append(path)
            elif name == "tool0":
                tool0_paths.append(path)
            elif name == "flange":
                flange_paths.append(path)

        fingertip_paths = sorted(fingertip_paths)[:2]
        if len(fingertip_paths) >= 2:
            my_print(f"✅ [MastServo] 使用两指尖中心作为视觉伺服控制点: {fingertip_paths}")
            return fingertip_paths
        fallback_paths = sorted(tool0_paths) or sorted(flange_paths)
        if fallback_paths:
            fallback_paths = sorted(fallback_paths)[:1]
            my_print(f"⚠️ [MastServo] 未找到双指尖，退回工具法兰投影: {fallback_paths}")
            return fallback_paths
        my_print("⚠️ [MastServo] 未找到夹爪指尖/tool0 prim，退回 wrist_3_link 投影。")
        return []

    def _control_point_world(self):
        points = []
        for path in self.tool_point_paths:
            try:
                pos, _ = self.get_world_pose(path)
                points.append(np.array(pos, dtype=np.float64))
            except Exception:
                pass
        if points:
            return np.mean(np.stack(points, axis=0), axis=0), "fingertip_center"
        ee_pos, _ = self.ee_solver.compute_end_effector_pose()
        return np.array(ee_pos, dtype=np.float64), "wrist_3_link"

    def start(self, target_pixel, grasp_width=0.055, target_cam_pos=None):
        self.target_pixel = np.array(target_pixel, dtype=np.float64)
        self.target_cam_pos = np.array(target_cam_pos, dtype=np.float64) if target_cam_pos is not None else None
        self.grasp_width = float(np.clip(grasp_width, 0.025, 0.10))
        self.frame_count = 0
        self.start_ee_pos = None
        self.state = "OPEN"
        depth_msg = f", target_depth={self.target_cam_pos[2]:.3f}m" if self.target_cam_pos is not None else ""
        my_print(
            f"🦾 [MastServo] 启动高杆视觉伺服抓取: target_pixel=({self.target_pixel[0]:.1f},{self.target_pixel[1]:.1f}), "
            f"width={self.grasp_width:.3f}{depth_msg}"
        )

    def _project_world_to_mast(self, world_pos):
        cam_world_pos, cam_world_quat = self.get_world_pose(self.cam_prim_path)
        cam_q = quat_wxyz_normalize(cam_world_quat)
        cam_q_inv = np.array([cam_q[0], -cam_q[1], -cam_q[2], -cam_q[3]], dtype=np.float64)
        cam_local = quat_wxyz_apply(cam_q_inv, np.array(world_pos, dtype=np.float64) - np.array(cam_world_pos, dtype=np.float64))
        cam_pos = usd_camera_local_to_optical(cam_local)
        if cam_pos[2] <= 0.05:
            return None, cam_pos
        u = self.cx + cam_pos[0] * self.fx / cam_pos[2]
        v = self.cy + cam_pos[1] * self.fy / cam_pos[2]
        return np.array([u, v], dtype=np.float64), cam_pos

    def _image_jacobian_step(self, ee_pos, pixel_error, max_step=0.012):
        axes = np.eye(3, dtype=np.float64)
        base_uv, _ = self._project_world_to_mast(ee_pos)
        if base_uv is None:
            return np.zeros(3, dtype=np.float64)

        eps = 0.012
        cols = []
        for axis in axes:
            uv_eps, _ = self._project_world_to_mast(ee_pos + axis * eps)
            if uv_eps is None:
                cols.append(np.zeros(2, dtype=np.float64))
            else:
                cols.append((uv_eps - base_uv) / eps)
        jac = np.stack(cols, axis=1)  # 2x3, pixel/meter
        try:
            delta = np.linalg.pinv(jac).dot(pixel_error * 0.45)
        except Exception:
            delta = np.zeros(3, dtype=np.float64)

        # 对齐阶段不要让深度方向乱跳，下降交给 DESCEND 状态。
        delta[2] = np.clip(delta[2], -0.004, 0.004)
        norm = np.linalg.norm(delta)
        if norm > max_step:
            delta = delta / norm * max_step
        return delta

    def _go_delta(self, delta):
        ee_pos, _ = self.ee_solver.compute_end_effector_pose()
        target_pos = np.array(ee_pos, dtype=np.float64) + np.array(delta, dtype=np.float64)
        action, success = self.ee_solver.compute_inverse_kinematics(
            target_position=target_pos,
            target_orientation=self.base_down_quat,
        )
        if success:
            self.robot.apply_action(action)
        return bool(success), ee_pos, target_pos

    def step(self):
        if self.state in ["IDLE", "DONE", "FAILED"]:
            return self.state

        if self.state == "OPEN":
            self.gripper.open()
            ee_pos, _ = self.ee_solver.compute_end_effector_pose()
            self.start_ee_pos = np.array(ee_pos, dtype=np.float64)
            cp_pos, cp_name = self._control_point_world()
            cp_uv, cp_cam = self._project_world_to_mast(cp_pos)
            if cp_uv is not None:
                my_print(
                    f"👁️ [MastServo] 初始控制点={cp_name}, pixel=({cp_uv[0]:.1f},{cp_uv[1]:.1f}), "
                    f"cam_z={cp_cam[2]:.3f}"
                )
            self.state = "SERVO_ABOVE"
            self.frame_count = 0
            return self.state

        cp_pos, cp_name = self._control_point_world()
        ee_uv, ee_cam = self._project_world_to_mast(cp_pos)
        if ee_uv is None:
            my_print("❌ [MastServo] 当前夹爪控制点不在高杆相机前方，无法视觉伺服。")
            self.state = "FAILED"
            return self.state

        pixel_error = self.target_pixel - ee_uv
        err_norm = float(np.linalg.norm(pixel_error))

        if self.state == "SERVO_ABOVE":
            if self.frame_count % 12 == 0:
                my_print(
                    f"🎯 [MastServo] 对齐中: {cp_name}=({ee_uv[0]:.1f},{ee_uv[1]:.1f}), "
                    f"target=({self.target_pixel[0]:.1f},{self.target_pixel[1]:.1f}), err={err_norm:.1f}px, cam_z={ee_cam[2]:.3f}"
                )
            if err_norm < 24.0:
                my_print(f"✅ [MastServo] 图像对齐完成，进入下探: err={err_norm:.1f}px")
                self.state = "DESCEND"
                self.frame_count = 0
                return self.state
            if self.frame_count > 180:
                if err_norm < 55.0:
                    my_print(f"⚠️ [MastServo] 对齐未完全收敛但误差可接受，进入下探: err={err_norm:.1f}px")
                    self.state = "DESCEND"
                else:
                    my_print(f"❌ [MastServo] 对齐失败，拒绝抓空气: err={err_norm:.1f}px")
                    self.state = "FAILED"
                self.frame_count = 0
                return self.state

            delta = self._image_jacobian_step(cp_pos, pixel_error, max_step=0.010)
            ok, _, _ = self._go_delta(delta)
            self.frame_count += 1
            if not ok:
                my_print("⚠️ [MastServo] 对齐IK失败，缩小步长重试。")
                ok, _, _ = self._go_delta(delta * 0.35)
                if not ok:
                    self.state = "FAILED"

        elif self.state == "DESCEND":
            align_delta = self._image_jacobian_step(cp_pos, pixel_error, max_step=0.006)
            align_delta[2] = 0.0
            descend = np.array([0.0, 0.0, -0.0045], dtype=np.float64)
            delta = align_delta + descend
            ok, current_pos, _ = self._go_delta(delta)
            self.frame_count += 1
            descended = abs(float(current_pos[2] - self.start_ee_pos[2])) if self.start_ee_pos is not None else 0.0
            depth_err = None
            if self.target_cam_pos is not None:
                depth_err = float(ee_cam[2] - self.target_cam_pos[2])
            if self.frame_count % 12 == 0:
                depth_msg = f", depth_err={depth_err:.3f}m" if depth_err is not None else ""
                my_print(f"⬇️ [MastServo] 下探中: err={err_norm:.1f}px, descended={descended:.3f}m{depth_msg}")
            if not ok:
                my_print("⚠️ [MastServo] 下探IK失败，停止下探并尝试闭合。")
                self.state = "CLOSE"
                self.frame_count = 0
            elif (depth_err is not None and abs(depth_err) < 0.045 and descended > 0.04) or descended > self.descent_limit or self.frame_count > 90:
                self.state = "CLOSE"
                self.frame_count = 0

        elif self.state == "CLOSE":
            self.gripper.close(self.grasp_width * 0.40)
            self.frame_count += 1
            if self.frame_count > 50:
                self.state = "LIFT"
                self.frame_count = 0

        elif self.state == "LIFT":
            ok, _, _ = self._go_delta(np.array([0.0, 0.0, 0.006], dtype=np.float64))
            self.frame_count += 1
            if not ok:
                my_print("⚠️ [MastServo] 抬升IK失败，结束执行进入验收。")
                self.state = "DONE"
            elif self.frame_count > 80:
                self.state = "DONE"

        return self.state


class WristVisionSweepGraspExecutor:
    """高杆只给粗方向；手腕相机看到目标后用纯图像闭环抓取。"""

    def __init__(self, robot, ee_solver, gripper, wrist_cam, dino_tracker, vlm_brain):
        self.robot = robot
        self.ee_solver = ee_solver
        self.gripper = gripper
        self.wrist_cam = wrist_cam
        self.dino_tracker = dino_tracker
        self.vlm_brain = vlm_brain
        self.base_down_quat = np.array([0.0, 0.7071, 0.7071, 0.0], dtype=np.float64)
        self.state = "IDLE"
        self.frame_count = 0
        self.detect_count = 0
        self.lost_count = 0
        self.search_index = 0
        self.search_phase_frame = 0
        self.search_pattern = []
        self.target_prompt = "rock."
        self.grasp_width = 0.055
        self.last_box = None
        self.filtered_box = None
        self.last_area = 0.0
        self.area_history = []
        self.err_history = []
        self.reference_rgb = None
        self.target_verified = False
        self.closed_on_valid_target = False
        self.reject_log_count = 0
        self.close_area = 7000.0
        self.close_ready_count = 0
        self.last_motion_delta = np.zeros(3, dtype=np.float64)
        self.world_align_fn = None
        self.world_align_log_count = 0
        self.world_align_tolerance = float(os.environ.get("OMNILRS_WORLD_ALIGN_TOL", "0.030"))
        self.near_field_area = float(os.environ.get("OMNILRS_WRIST_NEAR_FIELD_AREA", "15000"))
        self.near_field_center_px = float(os.environ.get("OMNILRS_WRIST_NEAR_FIELD_CENTER_PX", "55"))
        self.near_descend_frames = int(os.environ.get("OMNILRS_WRIST_NEAR_DESCEND_FRAMES", "32"))
        self.near_descend_step_z = float(os.environ.get("OMNILRS_WRIST_NEAR_DESCEND_Z_STEP", "0.0020"))
        self.image_size = np.array([640.0, 480.0], dtype=np.float64)
        self.desired_pixel_offset = np.array([
            float(os.environ.get("OMNILRS_WRIST_TARGET_U_OFFSET", "0.0")),
            float(os.environ.get("OMNILRS_WRIST_TARGET_V_OFFSET", "0.0")),
        ], dtype=np.float64)
        self.image_jacobian_xy = None
        self.calib_phase = "IDLE"
        self.calib_axis_index = 0
        self.calib_wait = 0
        self.calib_base_uv = None
        self.calib_cols = []
        self.centered_count = 0

    def set_world_alignment_callback(self, fn):
        self.world_align_fn = fn

    def _reset_run_state(self, target_prompt, grasp_width):
        self.target_prompt = target_prompt or "rock."
        self.grasp_width = float(np.clip(grasp_width, 0.025, 0.10))
        self.frame_count = 0
        self.detect_count = 0
        self.lost_count = 0
        self.search_index = 0
        self.search_phase_frame = 0
        self.last_box = None
        self.filtered_box = None
        self.last_area = 0.0
        self.area_history = []
        self.err_history = []
        self.closed_on_valid_target = False
        self.reject_log_count = 0
        self.close_ready_count = 0
        self.last_motion_delta = np.zeros(3, dtype=np.float64)
        self.world_align_log_count = 0
        self._reset_visual_servo_calibration()
        self.centered_count = 0

    def start(self, mast_pixel, roi_box, target_prompt, grasp_width=0.055, mast_rgb=None):
        self._reset_run_state(target_prompt, grasp_width)
        self.reference_rgb = self._crop_reference(mast_rgb, roi_box)
        self.target_verified = False
        self.search_pattern = self._build_search_pattern(mast_pixel, roi_box)
        self.state = "OPEN"
        my_print(
            f"🦾 [WristVision] 启动无坐标抓取: mast_pixel=({mast_pixel[0]:.1f},{mast_pixel[1]:.1f}), "
            f"prompt={self.target_prompt}, width={self.grasp_width:.3f}"
        )

    def start_wrist_first(self, target_prompt, grasp_width=0.055, wrist_rgb=None, wrist_box=None):
        self._reset_run_state(target_prompt, grasp_width)
        self.reference_rgb = self._crop_reference(wrist_rgb, wrist_box) if wrist_rgb is not None and wrist_box is not None else None
        # 手腕相机已经近距离看到目标，调试模式下不再要求高杆参考裁判，否则会被远处岩石带偏。
        self.target_verified = True
        self.search_pattern = [
            ("wrist_hold", np.array([0.0, 0.0, 0.0], dtype=np.float64), 8),
            ("micro_forward", np.array([0.0030, 0.0, -0.0004], dtype=np.float64), 28),
            ("micro_left", np.array([0.0, 0.0030, 0.0], dtype=np.float64), 24),
            ("micro_right", np.array([0.0, -0.0030, 0.0], dtype=np.float64), 48),
            ("micro_raise", np.array([0.0, 0.0, 0.0020], dtype=np.float64), 20),
        ]
        self.state = "OPEN"
        if wrist_box is not None:
            x0, y0, x1, y1 = [float(v) for v in wrist_box]
            my_print(
                f"🦾 [WristVision] 启动手腕优先抓取: wrist_box={[round(x0,1), round(y0,1), round(x1,1), round(y1,1)]}, "
                f"prompt={self.target_prompt}, width={self.grasp_width:.3f}"
            )
        else:
            my_print(f"🦾 [WristVision] 启动手腕优先抓取: prompt={self.target_prompt}, width={self.grasp_width:.3f}")

    def detect_wrist_target_once(self, target_prompt):
        old_prompt = self.target_prompt
        old_verified = self.target_verified
        self.target_prompt = target_prompt or "rock."
        self.target_verified = True
        box, rgb = self._detect_wrist_target()
        self.target_prompt = old_prompt
        self.target_verified = old_verified
        return box, rgb

    def _crop_reference(self, rgb, roi_box):
        if rgb is None or roi_box is None:
            return None
        h, w = rgb.shape[:2]
        x0, y0, x1, y1 = [int(v) for v in roi_box]
        bw, bh = max(1, x1 - x0), max(1, y1 - y0)
        margin_x, margin_y = int(0.35 * bw), int(0.35 * bh)
        x0 = max(0, x0 - margin_x)
        y0 = max(0, y0 - margin_y)
        x1 = min(w, x1 + margin_x)
        y1 = min(h, y1 + margin_y)
        crop = rgb[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        try:
            Image.fromarray(crop.astype(np.uint8)).save("debug_mast_target_reference.jpg")
        except Exception:
            pass
        return crop.astype(np.uint8)

    def _build_search_pattern(self, mast_pixel, roi_box):
        u = float(mast_pixel[0])
        side = -1.0 if u > self.image_size[0] * 0.5 else 1.0
        if roi_box is not None:
            roi_u = 0.5 * (float(roi_box[0]) + float(roi_box[2]))
            side = -1.0 if roi_u > self.image_size[0] * 0.5 else 1.0

        return [
            ("forward_to_workspace", np.array([0.0060, 0.0020 * side, -0.0005], dtype=np.float64), 45),
            ("lateral_to_mast_hint", np.array([0.0000, 0.0050 * side, 0.0000], dtype=np.float64), 32),
            ("lower_camera_view", np.array([0.0020, 0.0000, -0.0035], dtype=np.float64), 34),
            ("cross_sweep", np.array([0.0000, -0.0050 * side, 0.0000], dtype=np.float64), 72),
            ("raise_recover", np.array([0.0000, 0.0000, 0.0030], dtype=np.float64), 24),
            ("forward_deeper", np.array([0.0050, 0.0000, -0.0010], dtype=np.float64), 42),
            ("reverse_cross_sweep", np.array([0.0000, 0.0045 * side, 0.0000], dtype=np.float64), 72),
        ]

    def _go_delta(self, delta, smooth=True):
        delta = np.array(delta, dtype=np.float64)
        if np.linalg.norm(delta) < 0.00025:
            return True
        if smooth:
            delta = 0.55 * self.last_motion_delta + 0.45 * delta
            norm = np.linalg.norm(delta)
            if norm < 0.00020:
                return True
            if norm > 0.0048:
                delta = delta / norm * 0.0048
            self.last_motion_delta = delta.copy()
        else:
            self.last_motion_delta = np.zeros(3, dtype=np.float64)
        ee_pos, _ = self.ee_solver.compute_end_effector_pose()
        target_pos = np.array(ee_pos, dtype=np.float64) + delta
        action, success = self.ee_solver.compute_inverse_kinematics(
            target_position=target_pos,
            target_orientation=self.base_down_quat,
        )
        if success:
            self.robot.apply_action(action)
        return bool(success)

    def _detect_wrist_target(self):
        if self.wrist_cam is None:
            return None, None
        rgba = self.wrist_cam.get_rgba()
        if rgba is None or rgba.size == 0:
            return None, None
        rgb = rgba[:, :, :3].astype(np.uint8)
        self.image_size = np.array([rgb.shape[1], rgb.shape[0]], dtype=np.float64)

        try:
            Image.fromarray(rgb).save("debug_wrist_view.jpg")
        except Exception:
            pass

        try:
            payload = {
                "image_base64": self.dino_tracker._encode_image(rgb),
                "text_prompt": self.target_prompt,
            }
            res = requests.post(self.dino_tracker.server_url, json=payload, timeout=5).json()
            if not res.get("found"):
                return None, rgb
            boxes = res.get("boxes", [])
            if not boxes:
                return None, rgb
            box = self._choose_wrist_box(boxes, rgb)
            if box is None:
                return None, rgb
            if not self.target_verified:
                ok, _ = self.vlm_brain.verify_wrist_candidate(self.reference_rgb, rgb, box, self.target_prompt)
                if not ok:
                    return None, rgb
                self.target_verified = True
            return box, rgb
        except Exception as e:
            my_print(f"⚠️ [WristVision] 手腕相机DINO检测异常: {e}")
            return None, rgb

    def _choose_wrist_box(self, boxes, rgb):
        cx, cy = self.image_size[0] * 0.5, self.image_size[1] * 0.5
        best_box, best_score = None, -1e9
        for box in boxes:
            x0, y0, x1, y1 = [float(v) for v in box]
            reason = self._reject_box_reason(rgb, [x0, y0, x1, y1])
            if reason is not None:
                if self.reject_log_count < 8:
                    my_print(f"🚫 [WristVision] 拒绝手腕候选框: {reason}, box={[round(x0,1), round(y0,1), round(x1,1), round(y1,1)]}")
                    self.reject_log_count += 1
                continue
            area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
            bx, by = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            center_penalty = 4.0 * math.hypot(bx - cx, by - cy)
            score = area - center_penalty
            if score > best_score:
                best_box, best_score = box, score
        return np.array(best_box, dtype=np.float64) if best_box is not None else None

    def _box_center(self, box):
        x0, y0, x1, y1 = [float(v) for v in box]
        return np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)], dtype=np.float64)

    def _target_center(self):
        return np.array([
            self.image_size[0] * 0.5 + self.desired_pixel_offset[0],
            self.image_size[1] * 0.5 + self.desired_pixel_offset[1],
        ], dtype=np.float64)

    def _reset_visual_servo_calibration(self):
        self.image_jacobian_xy = None
        self.calib_phase = "IDLE"
        self.calib_axis_index = 0
        self.calib_wait = 0
        self.calib_base_uv = None
        self.calib_cols = []

    def _step_visual_servo_calibration(self, box):
        axes = [
            np.array([0.0045, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0045, 0.0], dtype=np.float64),
        ]

        if self.calib_phase == "IDLE":
            self.calib_phase = "PROBE"
            self.calib_axis_index = 0
            self.calib_cols = []
            my_print("🧭 [WristVision] 开始手腕图像伺服自校准：用小步探针学习末端XY到像素UV的映射。")

        if self.calib_axis_index >= len(axes):
            if len(self.calib_cols) == 2:
                jac = np.stack(self.calib_cols, axis=1)
                col_norms = np.linalg.norm(jac, axis=0)
                if np.all(col_norms > 200.0) and np.all(col_norms < 12000.0):
                    self.image_jacobian_xy = jac
                    my_print(
                        f"✅ [WristVision] 图像雅可比标定完成: "
                        f"J=[[{jac[0,0]:.1f},{jac[0,1]:.1f}],[{jac[1,0]:.1f},{jac[1,1]:.1f}]] px/m"
                    )
                elif np.any(col_norms >= 12000.0):
                    my_print(f"⚠️ [WristVision] 图像雅可比异常过大，判定为检测框跳变，退回保守固定映射: col_norms={col_norms}")
                    self.image_jacobian_xy = None
                else:
                    my_print(f"⚠️ [WristVision] 图像雅可比太弱，退回保守固定映射: col_norms={col_norms}")
                    self.image_jacobian_xy = None
                self.calib_phase = "DONE"
            return False

        axis = axes[self.calib_axis_index]
        eps = float(np.linalg.norm(axis))

        if self.calib_phase == "PROBE":
            self.calib_base_uv = self._box_center(box)
            if not self._go_delta(axis, smooth=False):
                self._go_delta(axis * 0.35, smooth=False)
            self.calib_wait = 2
            self.calib_phase = "MEASURE"
            return True

        if self.calib_phase == "MEASURE":
            if self.calib_wait > 0:
                self.calib_wait -= 1
                return True
            current_uv = self._box_center(box)
            col = (current_uv - self.calib_base_uv) / max(eps, 1e-6)
            self.calib_cols.append(col)
            if not self._go_delta(-axis, smooth=False):
                self._go_delta(-axis * 0.35, smooth=False)
            self.calib_wait = 2
            self.calib_phase = "RETURN"
            return True

        if self.calib_phase == "RETURN":
            if self.calib_wait > 0:
                self.calib_wait -= 1
                return True
            self.calib_axis_index += 1
            self.calib_phase = "PROBE"
            return True

        return False

    def _reject_box_reason(self, rgb, box):
        h, w = rgb.shape[:2]
        x0, y0, x1, y1 = [float(v) for v in box]
        bw, bh = max(0.0, x1 - x0), max(0.0, y1 - y0)
        area = bw * bh
        img_area = max(1.0, float(w * h))
        area_ratio = area / img_area
        width_ratio = bw / max(1.0, float(w))
        height_ratio = bh / max(1.0, float(h))

        if area_ratio > 0.18:
            return f"面积过大({area_ratio:.2f})，大概率是车体/地面"
        if width_ratio > 0.55 or height_ratio > 0.55:
            return f"框跨度过大(w={width_ratio:.2f},h={height_ratio:.2f})"
        if area_ratio < 0.00015:
            return f"面积过小({area_ratio:.5f})"
        touches = [
            x0 <= 3.0,
            y0 <= 3.0,
            x1 >= w - 3.0,
            y1 >= h - 3.0,
        ]
        if sum(1 for v in touches if v) >= 2 and area_ratio > 0.015:
            return "候选框贴住多个图像边界，像自体/画面边缘误检"
        if x0 < 0.18 * w and y1 > 0.55 * h and area_ratio > 0.010:
            return "左下区域大框，符合底座/车体误检"

        ix0, iy0 = int(max(0, x0)), int(max(0, y0))
        ix1, iy1 = int(min(w, x1)), int(min(h, y1))
        crop = rgb[iy0:iy1, ix0:ix1].astype(np.float32)
        if crop.size == 0:
            return "空裁剪"
        r, g, b = crop[..., 0], crop[..., 1], crop[..., 2]
        yellow_ratio = float(np.mean((r > 150) & (g > 130) & (b < 80)))
        white_robot_ratio = float(np.mean((r > 185) & (g > 185) & (b > 170)))
        black_robot_ratio = float(np.mean((r < 35) & (g < 35) & (b < 35)))
        if yellow_ratio > 0.06:
            return f"黄色车体占比过高({yellow_ratio:.2f})"
        if white_robot_ratio > 0.42 and area_ratio > 0.008:
            return f"白色机械臂/车体占比过高({white_robot_ratio:.2f})"
        if black_robot_ratio > 0.55 and area_ratio > 0.008:
            return f"黑色夹爪/阴影占比过高({black_robot_ratio:.2f})"
        return None

    def _box_error(self, box):
        x0, y0, x1, y1 = [float(v) for v in box]
        bx, by = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        target_uv = self._target_center()
        err_u = bx - target_uv[0]
        err_v = by - target_uv[1]
        return err_u, err_v, area

    def _remember_alignment_stats(self, err_norm, area):
        self.err_history.append(float(err_norm))
        self.area_history.append(float(area))
        self.err_history = self.err_history[-8:]
        self.area_history = self.area_history[-8:]

    def _alignment_is_stable(self):
        if len(self.err_history) < 5 or len(self.area_history) < 5:
            return False
        recent_err = np.array(self.err_history[-5:], dtype=np.float64)
        recent_area = np.array(self.area_history[-5:], dtype=np.float64)
        area_mean = max(1.0, float(np.mean(recent_area)))
        if area_mean > self.near_field_area:
            return (
                float(np.max(recent_err)) < self.near_field_center_px + 4.0
                and float(np.std(recent_err)) < 4.5
                and float(np.std(recent_area) / area_mean) < 0.12
            )
        return (
            float(np.max(recent_err)) < 30.0
            and float(np.std(recent_err)) < 5.5
            and float(np.std(recent_area) / area_mean) < 0.18
        )

    def _world_alignment_correction(self):
        if self.world_align_fn is None:
            return True
        try:
            info = self.world_align_fn()
        except Exception as e:
            if self.world_align_log_count < 4:
                my_print(f"⚠️ [WristVision] 世界坐标对齐回调异常，跳过几何校正: {e}")
                self.world_align_log_count += 1
            return True
        if not info:
            return True

        delta = np.array(info.get("delta", [0.0, 0.0, 0.0]), dtype=np.float64)
        err_norm = float(info.get("err_norm", np.linalg.norm(delta[:2])))
        source = info.get("source", "unknown")
        if err_norm <= self.world_align_tolerance:
            if self.world_align_log_count < 8:
                my_print(f"✅ [WristVision] 世界坐标辅助确认对齐: err={err_norm*1000:.1f}mm, source={source}")
                self.world_align_log_count += 1
            return True

        delta[2] = 0.0
        norm = np.linalg.norm(delta[:2])
        if norm > 0.006:
            delta[:2] = delta[:2] / max(norm, 1e-6) * 0.006
        if self.world_align_log_count < 16:
            rock = info.get("rock", None)
            tool = info.get("tool", None)
            if rock is not None and tool is not None:
                my_print(
                    f"🧭 [WristVision] 世界坐标末端校正: err={err_norm*1000:.1f}mm, "
                    f"delta=({delta[0]:+.4f},{delta[1]:+.4f},0), "
                    f"rock=({rock[0]:.3f},{rock[1]:.3f}), tool=({tool[0]:.3f},{tool[1]:.3f})"
                )
            else:
                my_print(f"🧭 [WristVision] 世界坐标末端校正: err={err_norm*1000:.1f}mm, source={source}")
            self.world_align_log_count += 1
        self._go_delta(delta)
        return False

    def _servo_delta_from_box(self, box, approach=False):
        err_u, err_v, area = self._box_error(box)
        delta = np.zeros(3, dtype=np.float64)

        err_vec = np.array([err_u, err_v], dtype=np.float64)
        if self.image_jacobian_xy is not None:
            try:
                delta_xy = np.linalg.pinv(self.image_jacobian_xy).dot(-0.70 * err_vec)
                delta[0] = delta_xy[0]
                delta[1] = delta_xy[1]
            except Exception:
                self.image_jacobian_xy = None

        if self.image_jacobian_xy is None:
            if abs(err_u) > 18.0:
                delta[1] = -np.sign(err_u) * min(0.0035, 0.0010 + abs(err_u) / 90000.0)
            if abs(err_v) > 22.0:
                delta[0] = np.sign(err_v) * min(0.0030, 0.0010 + abs(err_v) / 100000.0)

        if approach:
            err_norm = math.hypot(err_u, err_v)
            near_field_ok = area > self.near_field_area and err_norm < self.near_field_center_px
            if (abs(err_u) < 30.0 and abs(err_v) < 30.0) or near_field_ok:
                delta[0] += 0.0022
                delta[2] -= 0.0015

        norm = np.linalg.norm(delta)
        if norm > 0.0055:
            delta = delta / norm * 0.0055
        return delta, err_u, err_v, area

    def _run_search_step(self):
        if not self.search_pattern:
            self.state = "FAILED"
            return
        name, delta, frames = self.search_pattern[self.search_index]
        if self.search_phase_frame == 0:
            my_print(f"🔭 [WristVision] 手腕相机未见目标，执行粗扫描: {name}")
        ok = self._go_delta(delta)
        self.search_phase_frame += 1
        if not ok:
            ok = self._go_delta(delta * 0.35)
        if not ok:
            my_print(f"⚠️ [WristVision] 粗扫描IK失败: {name}")
            self.search_index += 1
            self.search_phase_frame = 0
        elif self.search_phase_frame >= frames:
            self.search_index += 1
            self.search_phase_frame = 0
        if self.search_index >= len(self.search_pattern):
            my_print("❌ [WristVision] 粗扫描结束仍未在手腕相机中发现目标。")
            self.state = "FAILED"

    def step(self):
        if self.state in ["IDLE", "DONE", "FAILED"]:
            return self.state

        if self.state == "OPEN":
            self.gripper.open()
            if self.wrist_cam is None:
                my_print("❌ [WristVision] 没有可用手腕相机，不能执行纯视觉抓取。")
                self.state = "FAILED"
                return self.state
            self.state = "WRIST_SEARCH"
            self.frame_count = 0
            return self.state

        should_detect = self.frame_count % 6 == 0 or self.state in ["WRIST_SERVO", "VISUAL_APPROACH"]
        box, _ = self._detect_wrist_target() if should_detect else (self.last_box, None)
        self.frame_count += 1

        if box is not None:
            if self.filtered_box is None:
                self.filtered_box = np.array(box, dtype=np.float64)
            else:
                self.filtered_box = 0.70 * self.filtered_box + 0.30 * np.array(box, dtype=np.float64)
            self.last_box = self.filtered_box.copy()
            self.lost_count = 0
        else:
            self.lost_count += 1
            if self.lost_count > 4:
                self.filtered_box = None

        if self.state == "WRIST_SEARCH":
            if self.last_box is not None:
                err_u, err_v, area = self._box_error(self.last_box)
                my_print(f"✅ [WristVision] 手腕相机锁定目标: err=({err_u:.1f},{err_v:.1f})px, area={area:.0f}")
                self._reset_visual_servo_calibration()
                self.centered_count = 0
                self.close_ready_count = 0
                self.area_history = []
                self.err_history = []
                self.state = "WRIST_SERVO"
                self.frame_count = 0
                return self.state
            self._run_search_step()

        elif self.state == "WRIST_SERVO":
            if self.last_box is None or self.lost_count > 10:
                my_print("⚠️ [WristVision] 手腕相机丢失目标，回到粗扫描。")
                self.state = "WRIST_SEARCH"
                self.last_box = None
                return self.state
            err_u, err_v, area = self._box_error(self.last_box)
            err_norm = math.hypot(err_u, err_v)
            self._remember_alignment_stats(err_norm, area)
            if self.frame_count % 8 == 0:
                jac_msg = "J=calib" if self.image_jacobian_xy is not None else "J=fallback"
                my_print(f"🎯 [WristVision] 2D居中: err={err_norm:.1f}px, area={area:.0f}, {jac_msg}")
            near_field_centered = area > self.near_field_area and err_norm < self.near_field_center_px
            very_close_target = area > 24000.0 and err_norm < 75.0
            if near_field_centered or very_close_target:
                my_print(
                    f"✅ [WristVision] 近场目标满足直接抓取条件(area={area:.0f}, err={err_norm:.1f}px)，"
                    f"停止居中/标定，固定下探 {self.near_descend_frames} 帧后闭合。"
                )
                self.state = "NEAR_DESCEND"
                self.frame_count = 0
                self.close_ready_count = 0
                return self.state
            if err_norm < 26.0:
                self.centered_count += 1
            else:
                self.centered_count = 0
            if self.image_jacobian_xy is None and self.calib_phase != "DONE":
                if self._step_visual_servo_calibration(self.last_box):
                    return self.state
            delta, err_u, err_v, area = self._servo_delta_from_box(self.last_box, approach=False)
            if self.centered_count >= 3:
                my_print(f"✅ [WristVision] 手腕图像居中，开始视觉靠近: area={area:.0f}")
                self.state = "VISUAL_APPROACH"
                self.frame_count = 0
                self.last_area = area
                self.close_ready_count = 0
                self.area_history = []
                self.err_history = []
                return self.state
            if self.frame_count > 90 and err_norm > 55.0:
                my_print("🔁 [WristVision] 居中长期不收敛，重新标定图像雅可比。")
                self._reset_visual_servo_calibration()
                self.frame_count = 0
            if np.linalg.norm(delta) > 1e-6 and not self._go_delta(delta):
                self._go_delta(delta * 0.35)

        elif self.state == "NEAR_DESCEND":
            if self.frame_count == 1:
                my_print(
                    f"⬇️ [WristVision] 近场固定下探: step_z={self.near_descend_step_z:.4f}m, "
                    f"frames={self.near_descend_frames}"
                )
            if self.frame_count <= self.near_descend_frames:
                ok = self._go_delta(np.array([0.0, 0.0, -self.near_descend_step_z], dtype=np.float64), smooth=False)
                if not ok:
                    self._go_delta(np.array([0.0, 0.0, -0.0007], dtype=np.float64), smooth=False)
            else:
                my_print("🤏 [WristVision] 近场固定下探完成，直接闭合夹爪。")
                self.closed_on_valid_target = True
                self.state = "CLOSE"
                self.frame_count = 0
                return self.state

        elif self.state == "VISUAL_APPROACH":
            if self.last_box is None or self.lost_count > 8:
                my_print("⚠️ [WristVision] 靠近阶段目标丢失，回到粗扫描。")
                self.state = "WRIST_SEARCH"
                self.last_box = None
                return self.state
            delta, err_u, err_v, area = self._servo_delta_from_box(self.last_box, approach=True)
            err_norm = math.hypot(err_u, err_v)
            self._remember_alignment_stats(err_norm, area)
            if self.frame_count % 8 == 0:
                my_print(f"⬇️ [WristVision] 视觉靠近: err={err_norm:.1f}px, area={area:.0f}")
            if not self.target_verified:
                my_print("⚠️ [WristVision] 候选目标未通过高杆参考图复核，拒绝闭合夹爪。")
                self.state = "WRIST_SEARCH"
                self.last_box = None
                return self.state
            near_field_centered = area > self.near_field_area and err_norm < self.near_field_center_px
            if err_norm > 45.0 and not near_field_centered:
                my_print("↩️ [WristVision] 靠近阶段偏离中心，退回2D居中后再继续。")
                self.state = "WRIST_SERVO"
                self.centered_count = 0
                self.close_ready_count = 0
                return self.state
            close_candidate = (
                (area > self.close_area and err_norm < 30.0)
                or (area > self.near_field_area and err_norm < self.near_field_center_px and self.frame_count > 8)
                or (self.frame_count > 150 and err_norm < 20.0)
            )
            if close_candidate and self._alignment_is_stable():
                if not self._world_alignment_correction():
                    self.close_ready_count = 0
                    return self.state
                self.close_ready_count += 1
            else:
                self.close_ready_count = 0
            if self.close_ready_count >= 4:
                my_print(
                    f"🤏 [WristVision] 图像/几何对齐连续稳定，闭合夹爪: "
                    f"area={area:.0f}, err={err_norm:.1f}px, stable_frames={self.close_ready_count}"
                )
                self.closed_on_valid_target = True
                self.state = "CLOSE"
                self.frame_count = 0
                return self.state
            if np.linalg.norm(delta) > 1e-6 and not self._go_delta(delta):
                self._go_delta(delta * 0.35)
            self.last_area = area

        elif self.state == "CLOSE":
            self.gripper.close(self.grasp_width * 0.38)
            self.frame_count += 1
            if self.frame_count > 45:
                self.state = "LIFT"
                self.frame_count = 0

        elif self.state == "LIFT":
            if self.frame_count == 0:
                my_print("⬆️ [WristVision] 夹爪闭合完成，只沿世界Z方向垂直上抬。")
            ok = self._go_delta(np.array([0.0, 0.0, 0.006], dtype=np.float64))
            self.frame_count += 1
            if not ok:
                self._go_delta(np.array([0.0, 0.0, 0.0025], dtype=np.float64))
            if self.frame_count > 80:
                self.state = "DONE"

        return self.state


class AnyGraspWristGraspExecutor:
    """手腕RGB-D -> AnyGrasp RPC -> 6D抓取候选 -> IK执行。

    AnyGrasp 在 ript_vla 环境的独立服务中运行；Isaac 进程只负责采集RGB-D和执行机械臂动作。
    """

    def __init__(
        self,
        robot,
        ee_solver,
        gripper,
        wrist_cam,
        get_camera_pose_fn,
        get_tool_center_fn,
        detector=None,
        service_url=None,
    ):
        self.robot = robot
        self.ee_solver = ee_solver
        self.gripper = gripper
        self.wrist_cam = wrist_cam
        self.get_camera_pose_fn = get_camera_pose_fn
        self.get_tool_center_fn = get_tool_center_fn
        self.detector = detector
        self.service_url = service_url or os.environ.get("OMNILRS_ANYGRASP_URL", "http://127.0.0.1:8777/grasp")
        self.health_url = self.service_url.rsplit("/", 1)[0] + "/health"
        self.base_down_quat = np.array([0.0, 0.7071, 0.7071, 0.0], dtype=np.float64)
        self.grasp_orientation = None  # world wxyz quaternion from AnyGrasp rotation_matrix
        self.state = "IDLE"
        self.frame_count = 0
        self.target_prompt = "rock."
        self.roi_box = None
        self.grasp_width = 0.055
        self.selected_grasp = None
        self.grasp_world = None
        self.pre_pos = None
        self.grasp_pos = None
        self.lift_pos = None
        self.closed_on_valid_target = False
        self.local_execution = False
        self.local_approach_failures = 0
        self.world_align_fn = None
        self.orientation_fallback_active = False
        self.plan_attempts = 0
        self.service_checked = False
        self.service_ready = False
        self.known_target_world_fn = None
        self.z_bias = float(os.environ.get("OMNILRS_ANYGRASP_Z_BIAS", "0.000"))
        self.pre_height = float(os.environ.get("OMNILRS_ANYGRASP_PRE_HEIGHT", "0.105"))
        self.lift_height = float(os.environ.get("OMNILRS_ANYGRASP_LIFT_HEIGHT", "0.180"))
        self.plan_timeout = float(os.environ.get("OMNILRS_ANYGRASP_TIMEOUT", "18.0"))
        self.z_min = float(os.environ.get("OMNILRS_ANYGRASP_Z_MIN", "0.04"))
        self.z_max = float(os.environ.get("OMNILRS_ANYGRASP_Z_MAX", "1.20"))
        self.max_points = int(os.environ.get("OMNILRS_ANYGRASP_MAX_POINTS", "60000"))
        self.min_grasp_score = float(os.environ.get("OMNILRS_ANYGRASP_MIN_SCORE", "-1.0"))
        self.debug_top_grasps = int(os.environ.get("OMNILRS_ANYGRASP_DEBUG_TOP_GRASPS", "5"))
        self.allow_depth_offset = os.environ.get("OMNILRS_ANYGRASP_ALLOW_DEPTH_OFFSET", "0").lower() in ["1", "true", "yes", "on"]
        self.tool_center_z_offset = float(os.environ.get("OMNILRS_ANYGRASP_TOOL_CENTER_Z_OFFSET", "0.100"))
        self.anchor_known_xy = os.environ.get("OMNILRS_ANYGRASP_ANCHOR_KNOWN_XY", "1").lower() not in ["0", "false", "no", "off"]
        self.use_anygrasp_yaw = os.environ.get("OMNILRS_ANYGRASP_USE_YAW", "1").lower() in ["1", "true", "yes", "on"]
        self.allow_orientation_fallback = os.environ.get("OMNILRS_ANYGRASP_ALLOW_ORIENTATION_FALLBACK", "0").lower() in ["1", "true", "yes", "on"]
        self.allow_local_fallback = os.environ.get("OMNILRS_ANYGRASP_ALLOW_LOCAL_FALLBACK", "0").lower() in ["1", "true", "yes", "on"]
        self.local_first = os.environ.get("OMNILRS_ANYGRASP_LOCAL_FIRST", "0").lower() in ["1", "true", "yes", "on"]
        self.world_solver_map = os.environ.get("OMNILRS_ANYGRASP_WORLD_SOLVER_MAP", "root").strip().lower()
        self.calibration_path = os.environ.get(
            "OMNILRS_IK_CALIBRATION_PATH",
            os.path.join(os.getcwd(), "ik_calibration_latest.json"),
        )
        self.calibrated_world_to_solver_matrix = None
        self.calibrated_solver_to_world_matrix = None
        self.allow_full_image = os.environ.get("OMNILRS_ANYGRASP_ALLOW_FULL_IMAGE", "0").lower() in ["1", "true", "yes", "on"]
        self.direct_world_grasp = os.environ.get("OMNILRS_ANYGRASP_DIRECT_WORLD_GRASP", "0").lower() in ["1", "true", "yes", "on"]
        self.direct_world_width = float(os.environ.get("OMNILRS_ANYGRASP_DIRECT_WORLD_WIDTH", "0.085"))
        self.direct_world_offset = np.array([
            float(os.environ.get("OMNILRS_ANYGRASP_DIRECT_WORLD_X_OFFSET", "0.000")),
            float(os.environ.get("OMNILRS_ANYGRASP_DIRECT_WORLD_Y_OFFSET", "0.000")),
            float(os.environ.get("OMNILRS_ANYGRASP_DIRECT_WORLD_Z_OFFSET", "0.000")),
        ], dtype=np.float64)
        self.align_frames = int(os.environ.get("OMNILRS_ANYGRASP_ALIGN_FRAMES", "150"))
        self.align_step = float(os.environ.get("OMNILRS_ANYGRASP_ALIGN_STEP", "0.0060"))
        self.align_stop_xy = float(os.environ.get("OMNILRS_ANYGRASP_ALIGN_STOP_XY", "0.085"))
        self.align_ready_xy = float(os.environ.get("OMNILRS_ANYGRASP_ALIGN_READY_XY", "0.150"))
        self.align_plateau_xy = float(os.environ.get("OMNILRS_ANYGRASP_ALIGN_PLATEAU_XY", "0.160"))
        self.align_worse_xy = float(os.environ.get("OMNILRS_ANYGRASP_ALIGN_WORSE_XY", "0.018"))
        self.align_best_xy = float("inf")
        self.align_worse_count = 0
        self._depth_fail_streak = 0
        self.align_edge_margin_px = float(os.environ.get("OMNILRS_ANYGRASP_ALIGN_EDGE_MARGIN_PX", "65"))
        self.roi_validate_px = float(os.environ.get("OMNILRS_ANYGRASP_ROI_VALIDATE_PX", "95"))
        self.depth_consistency_m = float(os.environ.get("OMNILRS_ANYGRASP_DEPTH_CONSISTENCY", "0.12"))
        self.pixel_servo_step = float(os.environ.get("OMNILRS_ANYGRASP_PIXEL_SERVO_STEP", "0.0045"))
        self.target_sector = os.environ.get("OMNILRS_TARGET_SECTOR", "front").strip().lower()
        self.sector_probe_frames = int(os.environ.get("OMNILRS_ANYGRASP_SECTOR_PROBE_FRAMES", "34"))
        self.sector_front_step = float(os.environ.get("OMNILRS_ANYGRASP_SECTOR_FRONT_STEP", "0.0048"))
        self.sector_right_step = float(os.environ.get("OMNILRS_ANYGRASP_SECTOR_RIGHT_STEP", "0.0042"))
        self.pregrasp_realign_limit = int(os.environ.get("OMNILRS_ANYGRASP_PREGRASP_REALIGN_LIMIT", "3"))
        self.pregrasp_realigns = 0
        self.local_approach_frames = int(os.environ.get("OMNILRS_ANYGRASP_LOCAL_APPROACH_FRAMES", "140"))
        self.local_approach_step = float(os.environ.get("OMNILRS_ANYGRASP_LOCAL_APPROACH_STEP", "0.0015"))
        self.local_approach_stop_xy = float(os.environ.get("OMNILRS_ANYGRASP_LOCAL_APPROACH_STOP_XY", "0.018"))
        self.local_descend_frames = int(os.environ.get("OMNILRS_ANYGRASP_LOCAL_DESCEND_FRAMES", "260"))
        self.local_descend_step = float(os.environ.get("OMNILRS_ANYGRASP_LOCAL_DESCEND_STEP", "0.0010"))
        self.local_descend_clearance = float(os.environ.get("OMNILRS_ANYGRASP_LOCAL_DESCEND_CLEARANCE", "0.030"))
        self.local_descend_max_total = float(os.environ.get("OMNILRS_ANYGRASP_LOCAL_DESCEND_MAX_TOTAL", "0.520"))
        self.local_descend_xy_correction_step = float(os.environ.get("OMNILRS_ANYGRASP_LOCAL_DESCEND_XY_CORRECTION_STEP", "0.0004"))
        self.local_descend_xy_deadband = float(os.environ.get("OMNILRS_ANYGRASP_LOCAL_DESCEND_XY_DEADBAND", "0.010"))
        self.local_descend_abort_xy = float(os.environ.get("OMNILRS_ANYGRASP_LOCAL_DESCEND_ABORT_XY", "0.030"))
        self.local_descend_forward_total = float(os.environ.get("OMNILRS_ANYGRASP_LOCAL_DESCEND_FORWARD_TOTAL", "0.140"))
        self.local_descend_right_total = float(os.environ.get("OMNILRS_ANYGRASP_LOCAL_DESCEND_RIGHT_TOTAL", "0.030"))
        self.local_lift_frames = int(os.environ.get("OMNILRS_ANYGRASP_LOCAL_LIFT_FRAMES", "72"))
        self.local_lift_step = float(os.environ.get("OMNILRS_ANYGRASP_LOCAL_LIFT_STEP", "0.0045"))
        self.lift_max_step = float(os.environ.get("OMNILRS_ANYGRASP_LIFT_MAX_STEP", "0.0045"))
        self.close_width_ratio = float(os.environ.get("OMNILRS_ANYGRASP_CLOSE_WIDTH_RATIO", "0.68"))
        self.close_width_margin = float(os.environ.get("OMNILRS_ANYGRASP_CLOSE_WIDTH_MARGIN", "0.030"))
        self.close_min_width = float(os.environ.get("OMNILRS_ANYGRASP_CLOSE_MIN_WIDTH", "0.030"))
        self.close_max_width = float(os.environ.get("OMNILRS_ANYGRASP_CLOSE_MAX_WIDTH", "0.090"))
        self.close_frames = int(os.environ.get("OMNILRS_ANYGRASP_CLOSE_FRAMES", "28"))
        self.close_start_width = float(os.environ.get("OMNILRS_ANYGRASP_CLOSE_START_WIDTH", "0.140"))
        self.close_target_width = None
        self.close_command_width = None
        self.hold_width_ratio = float(os.environ.get("OMNILRS_ANYGRASP_HOLD_WIDTH_RATIO", str(self.close_width_ratio)))
        self.hold_width_margin = float(os.environ.get("OMNILRS_ANYGRASP_HOLD_WIDTH_MARGIN", str(self.close_width_margin)))
        self.hold_min_width = float(os.environ.get("OMNILRS_ANYGRASP_HOLD_MIN_WIDTH", "0.035"))
        self.hold_tighten_delay_frames = int(os.environ.get("OMNILRS_ANYGRASP_HOLD_TIGHTEN_DELAY_FRAMES", "24"))
        self.hold_tighten_frames = int(os.environ.get("OMNILRS_ANYGRASP_HOLD_TIGHTEN_FRAMES", "70"))
        self.lift_min_frames = int(os.environ.get("OMNILRS_ANYGRASP_LIFT_MIN_FRAMES", "35"))
        self.lift_settle_frames = int(os.environ.get("OMNILRS_ANYGRASP_LIFT_SETTLE_FRAMES", "8"))
        self.lift_max_frames = int(os.environ.get("OMNILRS_ANYGRASP_LIFT_MAX_FRAMES", "85"))
        self.lift_reached_frames = 0
        self.hold_target_width = None
        self.hold_command_width = None
        self.reach_tolerance = float(os.environ.get("OMNILRS_ANYGRASP_REACH_TOL", "0.012"))
        self.pre_max_frames = int(os.environ.get("OMNILRS_ANYGRASP_PRE_MAX_FRAMES", "120"))
        self.descend_max_frames = int(os.environ.get("OMNILRS_ANYGRASP_DESCEND_MAX_FRAMES", "120"))
        self.pre_world_xy_tolerance = float(os.environ.get("OMNILRS_ANYGRASP_PRE_WORLD_XY_TOL", "0.035"))
        self.pre_world_z_tolerance = float(os.environ.get("OMNILRS_ANYGRASP_PRE_WORLD_Z_TOL", "0.060"))
        self.descend_world_xy_tolerance = float(os.environ.get("OMNILRS_ANYGRASP_DESCEND_WORLD_XY_TOL", "0.030"))
        self.close_world_xy_tolerance = float(os.environ.get("OMNILRS_ANYGRASP_CLOSE_WORLD_XY_TOL", "0.018"))
        self.close_world_z_tolerance = float(os.environ.get("OMNILRS_ANYGRASP_CLOSE_WORLD_Z_TOL", "0.050"))
        self.local_grasp_bias_solver = np.array([
            float(os.environ.get("OMNILRS_ANYGRASP_LOCAL_BIAS_X", "0.000")),
            float(os.environ.get("OMNILRS_ANYGRASP_LOCAL_BIAS_Y", "0.000")),
            float(os.environ.get("OMNILRS_ANYGRASP_LOCAL_BIAS_Z", "0.000")),
        ], dtype=np.float64)
        self.local_descend_start_tool_world = None
        self.local_descend_xy_failures = 0
        self.local_descend_z_failures = 0
        self._descend_total = self.local_descend_step * self.local_descend_frames
        self._descend_solver_step_z = -self.local_descend_step
        self._descend_required_down = None
        if self.world_solver_map in ["calibrated", "calib", "matrix", "ik_calibrated"]:
            self._load_ik_calibration()
        my_print(f"🦾 [AnyGrasp] 手腕RGB-D抓取执行器已启用，RPC={self.service_url}")
        if self.direct_world_grasp:
            my_print("🧪 [AnyGrasp] 当前为已知世界坐标直接抓取调试模式；不会使用AnyGrasp输出的6D抓取候选。")
        else:
            my_print("✅ [AnyGrasp] 当前为真实AnyGrasp RGB-D抓取模式；已知石头坐标只用于ROI/防抓地兜底。")
        my_print(
            f"🧭 [AnyGrasp] IK执行策略: local_first={int(self.local_first)}, "
            f"allow_local_fallback={int(self.allow_local_fallback)}, "
            f"allow_orientation_fallback={int(self.allow_orientation_fallback)}, "
            f"use_anygrasp_yaw={int(self.use_anygrasp_yaw)}, "
            f"world_solver_map={self.world_solver_map}, "
            f"min_score={self.min_grasp_score:.3f}, "
            f"allow_depth_offset={int(self.allow_depth_offset)}, "
            f"tool_center_z_offset={self.tool_center_z_offset*1000:+.0f}mm, "
            f"anchor_known_xy={int(self.anchor_known_xy)}, "
            f"close_ratio={self.close_width_ratio:.2f}, "
            f"close_margin={self.close_width_margin*1000:.0f}mm, "
            f"close_frames={self.close_frames}, "
            f"hold_ratio={self.hold_width_ratio:.2f}, "
            f"hold_margin={self.hold_width_margin*1000:.0f}mm, "
            f"hold_tighten_frames={self.hold_tighten_frames}, "
            f"lift_step={self.lift_max_step*1000:.1f}mm, "
            f"lift_max_frames={self.lift_max_frames}, "
            f"pre_world_xy_tol={self.pre_world_xy_tolerance*1000:.0f}mm, "
            f"descend_world_xy_tol={self.descend_world_xy_tolerance*1000:.0f}mm, "
            f"calibration_path={self.calibration_path if self.calibrated_world_to_solver_matrix is not None else 'n/a'}, "
            f"local_bias_solver="
            f"({self.local_grasp_bias_solver[0]:+.3f},{self.local_grasp_bias_solver[1]:+.3f},{self.local_grasp_bias_solver[2]:+.3f})"
        )

    def set_world_alignment_callback(self, fn):
        self.world_align_fn = fn

    def _load_ik_calibration(self):
        try:
            with open(self.calibration_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            world_to_solver = np.array(data.get("tool_world_to_solver_matrix"), dtype=np.float64)
            solver_to_world = np.array(data.get("solver_to_tool_world_matrix"), dtype=np.float64)
            if world_to_solver.shape != (3, 3) or solver_to_world.shape != (3, 3):
                raise ValueError("calibration matrices must be 3x3")
            rank = int(data.get("solver_rank", np.linalg.matrix_rank(solver_to_world)))
            residual = float(data.get("fit_residual_norm_m", float("nan")))
            if rank < 3:
                raise ValueError(f"calibration rank < 3: {rank}")
            self.calibrated_world_to_solver_matrix = world_to_solver
            self.calibrated_solver_to_world_matrix = solver_to_world
            my_print(
                f"📐 [IKCalib] 已加载标定矩阵: {self.calibration_path}, "
                f"rank={rank}, residual={residual*1000:.2f}mm"
            )
        except Exception as e:
            self.calibrated_world_to_solver_matrix = None
            self.calibrated_solver_to_world_matrix = None
            raise RuntimeError(f"failed_to_load_ik_calibration: {self.calibration_path}: {e}")

    def set_known_target_world_callback(self, fn):
        self.known_target_world_fn = fn

    def detect_wrist_target_once(self, target_prompt):
        if self.detector is not None:
            return self.detector.detect_wrist_target_once(target_prompt)
        rgba = self.wrist_cam.get_rgba() if self.wrist_cam is not None else None
        if rgba is None or rgba.size == 0:
            return None, None
        rgb = rgba[:, :, :3].astype(np.uint8)
        h, w = rgb.shape[:2]
        return np.array([w * 0.35, h * 0.35, w * 0.65, h * 0.70], dtype=np.float64), rgb

    def _reset_run_state(self, target_prompt, grasp_width, roi_box=None):
        self.target_prompt = target_prompt or "rock."
        self.grasp_width = float(np.clip(grasp_width, 0.025, 0.10))
        self.roi_box = [float(v) for v in roi_box] if roi_box is not None else None
        self.frame_count = 0
        self.plan_attempts = 0
        self.selected_grasp = None
        self.grasp_world = None
        self.grasp_orientation = None  # world wxyz quaternion
        self.pre_pos = None
        self.grasp_pos = None
        self.lift_pos = None
        self.closed_on_valid_target = False
        self.local_execution = False
        self.local_approach_failures = 0
        self.orientation_fallback_active = False
        self.local_descend_start_tool_world = None
        self.local_descend_xy_failures = 0
        self.local_descend_z_failures = 0
        self._descend_total = self.local_descend_step * self.local_descend_frames
        self._descend_solver_step_z = -self.local_descend_step
        self._descend_required_down = None
        self.close_target_width = None
        self.close_command_width = None
        self.hold_target_width = None
        self.hold_command_width = None
        self.lift_reached_frames = 0
        self.pregrasp_realigns = 0
        self.align_best_xy = float("inf")
        self.align_worse_count = 0
        self._depth_fail_streak = 0
        self._scan_done = False
        self._scan_step = 0

    def _compute_soft_close_width(self):
        raw_width = float(np.clip(self.grasp_width, 0.025, 0.14))
        ratio_target = raw_width * self.close_width_ratio
        margin_target = raw_width - self.close_width_margin
        target = max(ratio_target, margin_target, self.close_min_width)
        target = min(target, self.close_max_width, raw_width)
        return float(np.clip(target, 0.025, 0.14))

    def _compute_lift_hold_width(self):
        raw_width = float(np.clip(self.grasp_width, 0.025, 0.14))
        acquire_width = self.close_target_width
        if acquire_width is None:
            acquire_width = self._compute_soft_close_width()
        ratio_target = raw_width * self.hold_width_ratio
        margin_target = raw_width - self.hold_width_margin
        target = max(ratio_target, margin_target, self.hold_min_width)
        target = min(target, acquire_width, raw_width, self.close_max_width)
        return float(np.clip(target, 0.025, 0.14))

    def _command_lift_hold_width(self):
        if self.close_target_width is None:
            return None
        hold_width = self.hold_target_width if self.hold_target_width is not None else self.close_target_width
        delay_frames = max(0, int(self.hold_tighten_delay_frames))
        tighten_frames = max(1, int(self.hold_tighten_frames))
        if self.frame_count < delay_frames:
            command_width = self.close_target_width
        else:
            tighten_frame = self.frame_count - delay_frames + 1
            alpha = min(1.0, float(tighten_frame) / float(tighten_frames))
            command_width = self.close_target_width + (hold_width - self.close_target_width) * alpha
        self.hold_command_width = float(np.clip(command_width, 0.025, 0.14))
        if hasattr(self.gripper, "hold"):
            self.gripper.hold(self.hold_command_width)
        else:
            self.gripper.close(self.hold_command_width)

        if hold_width < self.close_target_width - 0.0005:
            if self.frame_count == 0:
                my_print(
                    f"🧷 [AnyGrasp] 抬升防滑收紧计划: acquire_width={self.close_target_width:.3f}, "
                    f"hold_width={hold_width:.3f}, delay={delay_frames}, frames={tighten_frames}"
                )
            if self.frame_count >= delay_frames:
                tighten_frame = self.frame_count - delay_frames + 1
                display_frame = min(tighten_frame, tighten_frames)
                if tighten_frame <= tighten_frames and (tighten_frame == 1 or tighten_frame == tighten_frames or tighten_frame % 10 == 0):
                    my_print(
                        f"🧷 [AnyGrasp] 抬升防滑收紧: frame={display_frame}/{tighten_frames}, "
                        f"command_width={self.hold_command_width:.3f}"
                    )
        return self.hold_command_width

    def start_wrist_first(self, target_prompt, grasp_width=0.055, wrist_rgb=None, wrist_box=None):
        self._reset_run_state(target_prompt, grasp_width, roi_box=wrist_box)
        self.state = "OPEN"
        if wrist_box is not None:
            x0, y0, x1, y1 = [float(v) for v in wrist_box]
            my_print(
                f"🦾 [AnyGrasp] 启动手腕RGB-D抓取: wrist_roi={[round(x0,1), round(y0,1), round(x1,1), round(y1,1)]}, "
                f"prompt={self.target_prompt}, width_hint={self.grasp_width:.3f}"
            )
        else:
            my_print(f"🦾 [AnyGrasp] 启动手腕RGB-D抓取: prompt={self.target_prompt}, width_hint={self.grasp_width:.3f}")

    def start(self, mast_pixel, roi_box, target_prompt, grasp_width=0.055, mast_rgb=None):
        self._reset_run_state(target_prompt, grasp_width, roi_box=None)
        self.state = "OPEN"
        my_print(
            f"🦾 [AnyGrasp] 高杆已交接，等待手腕RGB-D规划；mast_pixel=({mast_pixel[0]:.1f},{mast_pixel[1]:.1f}), "
            f"prompt={self.target_prompt}"
        )

    def _check_service(self):
        if self.service_checked:
            return self.service_ready
        self.service_checked = True
        try:
            res = requests.get(self.health_url, timeout=2.0)
            self.service_ready = bool(res.ok and res.json().get("ok"))
        except Exception as e:
            self.service_ready = False
            my_print(f"❌ [AnyGrasp] RPC服务不可用: {e}")
            my_print("   请先启动: conda activate ript_vla && cd ~/OmniLRS1 && export LD_LIBRARY_PATH=\"/home/xunden/isaacsim/exts/isaacsim.ros2.bridge/humble/lib:${LD_LIBRARY_PATH}\" && python anygrasp_rpc_server.py")
        return self.service_ready

    def _encode_rgb_png(self, rgb):
        img = Image.fromarray(rgb.astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _encode_depth_npy(self, depth):
        buf = io.BytesIO()
        np.save(buf, np.asarray(depth, dtype=np.float32))
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _capture_wrist_rgbd(self):
        if self.wrist_cam is None:
            raise RuntimeError("no wrist camera")
        rgba = self.wrist_cam.get_rgba()
        if rgba is None or rgba.size == 0:
            raise RuntimeError("empty wrist rgba")
        rgb = rgba[:, :, :3].astype(np.uint8)

        depth = None
        if hasattr(self.wrist_cam, "get_depth"):
            try:
                depth = self.wrist_cam.get_depth()
            except Exception:
                depth = None
        if depth is None and hasattr(self.wrist_cam, "get_current_frame"):
            try:
                frame = self.wrist_cam.get_current_frame()
                for key in ["distance_to_image_plane", "distance_to_camera", "depth"]:
                    if isinstance(frame, dict) and key in frame:
                        depth = frame[key]
                        break
            except Exception:
                depth = None
        if depth is None:
            raise RuntimeError("empty wrist depth; call wrist_cam.add_distance_to_image_plane_to_frame()")
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim > 2:
            depth = np.squeeze(depth)
        if depth.shape[:2] != rgb.shape[:2]:
            depth_img = Image.fromarray(depth.astype(np.float32))
            depth = np.array(depth_img.resize((rgb.shape[1], rgb.shape[0])), dtype=np.float32)
        return rgb, depth

    def _intrinsics(self, rgb):
        h, w = rgb.shape[:2]
        fx = float(os.environ.get("OMNILRS_WRIST_FX", str(w * 0.5)))
        fy = float(os.environ.get("OMNILRS_WRIST_FY", str(w * 0.5)))
        cx = float(os.environ.get("OMNILRS_WRIST_CX", str(w * 0.5)))
        cy = float(os.environ.get("OMNILRS_WRIST_CY", str(h * 0.5)))
        if hasattr(self.wrist_cam, "get_intrinsics_matrix"):
            try:
                k = np.asarray(self.wrist_cam.get_intrinsics_matrix(), dtype=np.float64)
                if k.shape[0] >= 3 and k.shape[1] >= 3:
                    fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
            except Exception:
                pass
        return fx, fy, cx, cy

    def _project_known_target_to_wrist(self, rgb=None):
        if self.known_target_world_fn is None:
            return None
        target_world = self.known_target_world_fn()
        if target_world is None:
            return None
        target_world = np.array(target_world, dtype=np.float64)
        if rgb is None:
            rgba = self.wrist_cam.get_rgba() if self.wrist_cam is not None else None
            if rgba is None or rgba.size == 0:
                return None
            rgb = rgba[:, :, :3].astype(np.uint8)
        h, w = rgb.shape[:2]
        cam_pos, cam_quat = self.get_camera_pose_fn()
        target_cam = world_to_optical_camera(cam_pos, cam_quat, target_world)
        fx, fy, cx, cy = self._intrinsics(rgb)
        uv = None
        inside = False
        if target_cam[2] > 0.03:
            u = cx + target_cam[0] * fx / target_cam[2]
            v = cy + target_cam[1] * fy / target_cam[2]
            uv = np.array([u, v], dtype=np.float64)
            inside = bool(0 <= u < w and 0 <= v < h)
        return {
            "target_world": target_world,
            "target_cam": target_cam,
            "uv": uv,
            "inside": inside,
            "shape": (h, w),
            "rgb": rgb,
        }

    def _roi_center(self, roi_box):
        x0, y0, x1, y1 = [float(v) for v in roi_box]
        return np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)], dtype=np.float64)

    def _roi_matches_known_target(self, roi_box, rgb=None, log_prefix="AnyGrasp"):
        proj = self._project_known_target_to_wrist(rgb=rgb)
        if proj is None:
            # No camera image → can't validate → accept (original behaviour).
            return True
        if proj["uv"] is None or not proj["inside"]:
            # Rock projects OUTSIDE the wrist image → the wrist can't see it.
            # DINO detections at this point are likely the gripper/arm/ground.
            my_print(
                f"🚫 [{log_prefix}] 拒绝手腕ROI: 已知石头在手腕画面外，"
                f"DINO候选大概率是夹爪/车体误检。"
            )
            return False
        roi_center = self._roi_center(roi_box)
        dist = float(np.linalg.norm(roi_center - proj["uv"]))
        if dist > self.roi_validate_px:
            my_print(
                f"🚫 [{log_prefix}] 拒绝手腕ROI: 与已知石头投影相差 {dist:.1f}px，"
                f"roi_center=({roi_center[0]:.1f},{roi_center[1]:.1f}), "
                f"rock_pixel=({proj['uv'][0]:.1f},{proj['uv'][1]:.1f})"
            )
            return False
        return True

    def _set_roi_from_known_projection(self, proj):
        if proj is None or proj["uv"] is None or not proj["inside"]:
            return False
        h, w = proj["shape"]
        u, v = proj["uv"]
        side = float(np.clip(0.18 * min(w, h), 70.0, 125.0))
        self.roi_box = [
            float(max(0.0, u - side)),
            float(max(0.0, v - side)),
            float(min(w - 1.0, u + side)),
            float(min(h - 1.0, v + side)),
        ]
        try:
            debug_img = Image.fromarray(proj["rgb"].astype(np.uint8))
            draw = ImageDraw.Draw(debug_img)
            draw.rectangle(self.roi_box, outline="cyan", width=6)
            debug_img.save("debug_anygrasp_projected_roi.jpg")
        except Exception:
            pass
        my_print(
            f"🎯 [AnyGrasp] 使用已知石头世界坐标投影ROI: "
            f"pixel=({u:.1f},{v:.1f}), roi={[round(v,1) for v in self.roi_box]}"
        )
        return True

    def _wrist_depth_at_projection(self, proj, win=7):
        if proj is None or proj["uv"] is None or not proj["inside"]:
            return None
        try:
            _, depth = self._capture_wrist_rgbd()
        except Exception:
            return None
        h, w = depth.shape[:2]
        u, v = proj["uv"]
        u = int(np.clip(round(float(u)), 0, w - 1))
        v = int(np.clip(round(float(v)), 0, h - 1))
        u0, u1 = max(0, u - win), min(w, u + win + 1)
        v0, v1 = max(0, v - win), min(h, v + win + 1)
        patch = depth[v0:v1, u0:u1]
        valid = patch[np.isfinite(patch) & (patch > 0.03) & (patch < 2.5)]
        if len(valid) == 0:
            return None
        return float(np.median(valid))

    def _maybe_refresh_roi_from_wrist(self):
        if self.roi_box is not None:
            if self._roi_matches_known_target(self.roi_box):
                return True
            self.roi_box = None

        proj = self._project_known_target_to_wrist()
        if self._set_roi_from_known_projection(proj):
            return True

        box, rgb = self.detect_wrist_target_once(self.target_prompt)
        if box is not None:
            candidate_roi = [float(v) for v in box]
            if not self._roi_matches_known_target(candidate_roi, rgb=rgb, log_prefix="AnyGrasp/DINO"):
                return False
            self.roi_box = candidate_roi
            try:
                debug_img = Image.fromarray(rgb.astype(np.uint8))
                draw = ImageDraw.Draw(debug_img)
                draw.rectangle(self.roi_box, outline="cyan", width=6)
                debug_img.save("debug_anygrasp_wrist_roi.jpg")
            except Exception:
                pass
            my_print(f"🎯 [AnyGrasp] 手腕DINO给出ROI: {[round(v,1) for v in self.roi_box]}")
            return True

        if proj is not None:
            target_cam = proj["target_cam"]
            uv = proj["uv"]
            if uv is None:
                my_print(f"⚠️ [AnyGrasp] 已知石头不在手腕相机前方，暂不做全图抓取: cam_z={target_cam[2]:.3f}")
            else:
                my_print(
                    f"⚠️ [AnyGrasp] 已知石头投影不在手腕图像内，暂不做全图抓取: "
                    f"pixel=({uv[0]:.1f},{uv[1]:.1f}), cam=({target_cam[0]:.3f},{target_cam[1]:.3f},{target_cam[2]:.3f})"
                )
        return False

    def _request_anygrasp(self, rgb, depth):
        fx, fy, cx, cy = self._intrinsics(rgb)
        payload = {
            "rgb_png": self._encode_rgb_png(rgb),
            "depth_npy": self._encode_depth_npy(depth),
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "roi_box": self.roi_box,
            "z_min": self.z_min,
            "z_max": self.z_max,
            "max_points": self.max_points,
            "top_k": 20,
        }
        res = requests.post(self.service_url, json=payload, timeout=self.plan_timeout)
        res.raise_for_status()
        data = res.json()
        if not data.get("ok"):
            raise RuntimeError(data.get("error", "anygrasp_failed"))
        return data

    def _tool_center_world(self):
        if self.get_tool_center_fn is not None:
            pos = self.get_tool_center_fn()
            if pos is not None:
                return np.array(pos, dtype=np.float64)
        ee_pos, _ = self.ee_solver.compute_end_effector_pose()
        return np.array(ee_pos, dtype=np.float64)

    def _world_delta_to_solver_delta(self, delta_world):
        delta_world = np.array(delta_world, dtype=np.float64)
        if self.world_solver_map in ["calibrated", "calib", "matrix", "ik_calibrated"]:
            if self.calibrated_world_to_solver_matrix is None:
                raise RuntimeError("ik_calibration_not_loaded")
            return self.calibrated_world_to_solver_matrix.dot(delta_world)
        if self.world_solver_map in ["neg_xy", "flip_xy", "minus_xy"]:
            return np.array([-delta_world[0], -delta_world[1], delta_world[2]], dtype=np.float64)
        if self.world_solver_map not in ["root", "root_quat", "robot_root", "root_forward", "root_fwd", "robot_root_forward"]:
            return delta_world.copy()
        try:
            _, root_quat = self.robot.get_world_pose()
            root_q = quat_wxyz_normalize(root_quat)
            if self.world_solver_map in ["root_forward", "root_fwd", "robot_root_forward"]:
                return quat_wxyz_apply(root_q, delta_world)
            root_q_inv = np.array([root_q[0], -root_q[1], -root_q[2], -root_q[3]], dtype=np.float64)
            return quat_wxyz_apply(root_q_inv, delta_world)
        except Exception:
            return delta_world

    def _solver_delta_to_world_delta(self, delta_solver):
        delta_solver = np.array(delta_solver, dtype=np.float64)
        if self.world_solver_map in ["calibrated", "calib", "matrix", "ik_calibrated"]:
            if self.calibrated_solver_to_world_matrix is None:
                raise RuntimeError("ik_calibration_not_loaded")
            return self.calibrated_solver_to_world_matrix.dot(delta_solver)
        if self.world_solver_map in ["neg_xy", "flip_xy", "minus_xy"]:
            return np.array([-delta_solver[0], -delta_solver[1], delta_solver[2]], dtype=np.float64)
        if self.world_solver_map not in ["root", "root_quat", "robot_root", "root_forward", "root_fwd", "robot_root_forward"]:
            return delta_solver.copy()
        try:
            _, root_quat = self.robot.get_world_pose()
            root_q = quat_wxyz_normalize(root_quat)
            if self.world_solver_map in ["root_forward", "root_fwd", "robot_root_forward"]:
                root_q_inv = np.array([root_q[0], -root_q[1], -root_q[2], -root_q[3]], dtype=np.float64)
                return quat_wxyz_apply(root_q_inv, delta_solver)
            return quat_wxyz_apply(root_q, delta_solver)
        except Exception:
            return delta_solver

    def _biased_local_grasp_world(self):
        if self.grasp_world is None:
            return None
        bias_world = self._solver_delta_to_world_delta(self.local_grasp_bias_solver)
        return np.array(self.grasp_world, dtype=np.float64) + bias_world

    def _choose_best_grasp(self, grasps):
        usable = []
        raw_debug = []
        rejected_score = 0
        for idx, g in enumerate(grasps):
            score = float(g.get("score", 0.0))
            width = float(g.get("width", self.grasp_width))
            trans = np.array(g.get("translation", [0, 0, 0]), dtype=np.float64)
            if trans.shape[0] >= 3 and np.all(np.isfinite(trans[:3])):
                raw_debug.append((score, width, trans.copy(), idx))
            if score < self.min_grasp_score:
                rejected_score += 1
                continue
            if not np.all(np.isfinite(trans)) or trans[2] < self.z_min or trans[2] > self.z_max:
                continue
            if width < 0.008 or width > 0.115:
                continue
            usable.append((score, g))
        if raw_debug and self.debug_top_grasps > 0:
            raw_debug.sort(key=lambda item: item[0], reverse=True)
            parts = []
            for score, width, trans, idx in raw_debug[:self.debug_top_grasps]:
                parts.append(
                    f"#{idx}:score={score:.3f},width={width*1000:.0f}mm,"
                    f"cam=({trans[0]:+.3f},{trans[1]:+.3f},{trans[2]:.3f})"
                )
            my_print(f"🧪 [AnyGrasp] 原始候选top{min(len(raw_debug), self.debug_top_grasps)}: " + " | ".join(parts))
        if rejected_score > 0:
            my_print(
                f"⚠️ [AnyGrasp] 因 OMNILRS_ANYGRASP_MIN_SCORE={self.min_grasp_score:.3f} "
                f"过滤低分候选: {rejected_score}/{len(grasps)}"
            )
        if not usable:
            return None
        usable.sort(key=lambda item: item[0], reverse=True)
        return usable[0][1]

    def _known_target_world(self):
        if self.known_target_world_fn is None:
            return None
        try:
            target = self.known_target_world_fn()
        except Exception:
            return None
        if target is None:
            return None
        target = np.array(target, dtype=np.float64)
        if not np.all(np.isfinite(target)):
            return None
        return target

    def _set_plan_from_world(self, grasp_world, source, score=None, grasp_cam=None,
                            grasp_orientation_world=None):
        grasp_world = np.array(grasp_world, dtype=np.float64)

        ee_pos, _ = self.ee_solver.compute_end_effector_pose()
        ee_pos = np.array(ee_pos, dtype=np.float64)

        tool_world = self._tool_center_world()
        delta_world = grasp_world - tool_world
        # Solver targets are expressed as run.py-style local XYZ increments.
        # Use the whole AnyGrasp target delta in strict mode; local fallback is
        # opt-in only because hand-written vertical descents can collide with the
        # ground and move the rock.
        delta_solver = self._world_delta_to_solver_delta(delta_world)

        target_ee = ee_pos + delta_solver

        self.grasp_pos = target_ee
        self.pre_pos = target_ee + np.array([0.0, 0.0, self.pre_height], dtype=np.float64)
        self.lift_pos = target_ee + np.array([0.0, 0.0, self.lift_height], dtype=np.float64)
        self.grasp_world = grasp_world
        self.grasp_orientation = grasp_orientation_world  # may be None → fallback
        self.local_execution = False
        self.selected_grasp = {
            "score": float(score) if score is not None else 1.0,
            "width": float(self.grasp_width),
            "translation": None if grasp_cam is None else np.array(grasp_cam, dtype=np.float64).tolist(),
            "source": source,
        }

        cam_msg = ""
        if grasp_cam is not None:
            grasp_cam = np.array(grasp_cam, dtype=np.float64)
            cam_msg = f", cam=({grasp_cam[0]:.3f},{grasp_cam[1]:.3f},{grasp_cam[2]:.3f})"
        score_msg = "" if score is None else f", score={float(score):.3f}"
        orient_msg = ""
        if grasp_orientation_world is not None:
            # Report yaw deviation from the hardcoded base_down_quat as a quick
            # sanity check in logs.
            orient_msg = ", orient=anygrasp"
        else:
            orient_msg = ", orient=base_down(fallback)"
        my_print(
            f"✅ [AnyGrasp] 抓取目标生成: source={source}{score_msg}, width={self.grasp_width:.3f}{cam_msg}{orient_msg}, "
            f"world=({grasp_world[0]:.3f},{grasp_world[1]:.3f},{grasp_world[2]:.3f}), "
            f"ee_target=({self.grasp_pos[0]:.3f},{self.grasp_pos[1]:.3f},{self.grasp_pos[2]:.3f})"
        )
        my_print(
            f"🧮 [AnyGrasp/IK] ee=({ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f}), "
            f"tool_world=({tool_world[0]:.3f},{tool_world[1]:.3f},{tool_world[2]:.3f}), "
            f"delta_world=({delta_world[0]:+.3f},{delta_world[1]:+.3f},{delta_world[2]:+.3f}), "
            f"delta_solver=({delta_solver[0]:+.3f},{delta_solver[1]:+.3f},{delta_solver[2]:+.3f}), "
            f"map={self.world_solver_map}, "
            f"pre=({self.pre_pos[0]:.3f},{self.pre_pos[1]:.3f},{self.pre_pos[2]:.3f})"
        )

    def _visualize_grasp_pose(self, rgb, grasp_cam, R_optical, fx, fy, cx, cy, score=None):
        """Draw the AnyGrasp 6D pose on the wrist RGB image before execution.

        Saves ``debug_anygrasp_pose_preview.jpg`` in the working directory so
        the operator can inspect whether the grasp center, approach axis, and
        closing direction look reasonable *before* the arm moves.
        """
        try:
            h, w = rgb.shape[:2]

            # ---- project grasp center (camera optical coords) to pixel ----
            x, y, z = float(grasp_cam[0]), float(grasp_cam[1]), float(grasp_cam[2])
            if z < 0.02:
                return
            u0 = int(round(cx + x * fx / z))
            v0 = int(round(cy + y * fy / z))

            # ---- axis endpoints: 3 cm along each grasp-frame axis ----
            R = np.asarray(R_optical, dtype=np.float64)
            axis_len = 0.030  # metres
            axis_colors = [
                (255, 0, 0),    # red   – approach  (R[:,0])
                (0, 255, 0),    # green – closing   (R[:,1])
                (0, 0, 255),    # blue  – orthogonal (R[:,2])
            ]
            axis_names = ["approach", "closing", "ortho"]

            img = Image.fromarray(rgb.astype(np.uint8))
            draw = ImageDraw.Draw(img)

            endpoints_px = []
            for i in range(3):
                pt_cam = grasp_cam + R[:, i] * axis_len
                u1 = int(round(cx + pt_cam[0] * fx / pt_cam[2]))
                v1 = int(round(cy + pt_cam[1] * fy / pt_cam[2]))
                endpoints_px.append((u1, v1))
                if 0 <= u1 < w and 0 <= v1 < h:
                    draw.line([(u0, v0), (u1, v1)], fill=axis_colors[i], width=3)

            # ---- gripper width bar (horizontal, along closing axis) ----
            half_w = float(self.grasp_width) * 0.5
            left_cam = grasp_cam + R[:, 1] * half_w
            right_cam = grasp_cam - R[:, 1] * half_w
            ul = int(round(cx + left_cam[0] * fx / left_cam[2]))
            vl = int(round(cy + left_cam[1] * fy / left_cam[2]))
            ur = int(round(cx + right_cam[0] * fx / right_cam[2]))
            vr = int(round(cy + right_cam[1] * fy / right_cam[2]))
            if all(0 <= p < w and 0 <= q < h for p, q in [(ul, vl), (ur, vr)]):
                draw.line([(ul, vl), (ur, vr)], fill=(255, 255, 0), width=5)  # yellow bar

            # ---- centre dot & legend ----
            r_dot = 5
            draw.ellipse([(u0 - r_dot, v0 - r_dot), (u0 + r_dot, v0 + r_dot)],
                         fill=(255, 255, 255), outline=(0, 0, 0))
            for i, name in enumerate(axis_names):
                ex, ey = endpoints_px[i]
                draw.text((ex + 4, ey - 8), name, fill=axis_colors[i])

            display_score = float(score) if score is not None else (
                float(self.selected_grasp.get("score", 0.0)) if self.selected_grasp else 0.0
            )
            draw.text((6, 6),
                      f"score={display_score:.3f}  width={self.grasp_width*1000:.0f}mm  "
                      f"cam_z={z:.3f}m",
                      fill=(0, 255, 0))

            save_path = os.path.join(os.getcwd(), "debug_anygrasp_pose_preview.jpg")
            img.save(save_path)
            my_print(f"📸 [AnyGrasp] 抓取位姿预览已保存: {save_path}  "
                     f"(红=approach, 绿=closing, 蓝=ortho, 黄条=夹爪宽度)")

        except Exception as e:
            my_print(f"⚠️ [AnyGrasp] 位姿可视化失败: {e}")

    def _plan_once(self):
        self._maybe_refresh_roi_from_wrist()
        if self.roi_box is None and not self.allow_full_image:
            raise RuntimeError(
                "no_wrist_roi_for_anygrasp; DINO服务未连接且石头未投影到手腕图像，拒绝对整张图抓地/抓车体"
            )
        rgb, depth = self._capture_wrist_rgbd()
        if self.direct_world_grasp:
            known_world = self._known_target_world()
            if known_world is not None:
                grasp_world = known_world + self.direct_world_offset
                grasp_world[2] += self.z_bias
                self.grasp_width = float(np.clip(self.direct_world_width, 0.025, 0.10))
                try:
                    Image.fromarray(rgb.astype(np.uint8)).save("debug_anygrasp_wrist_rgb.png")
                except Exception:
                    pass
                my_print(
                    f"🎯 [WorldGrasp] 使用已知目标世界坐标直接生成抓取目标: "
                    f"known=({known_world[0]:.3f},{known_world[1]:.3f},{known_world[2]:.3f}), "
                    f"offset=({self.direct_world_offset[0]:+.3f},{self.direct_world_offset[1]:+.3f},{self.direct_world_offset[2]:+.3f})"
                )
                self._set_plan_from_world(grasp_world, source="known_world_direct")
                return

        data = self._request_anygrasp(rgb, depth)
        grasp = self._choose_best_grasp(data.get("grasps", []))
        if grasp is None:
            raise RuntimeError("no_usable_grasp_after_filter")

        cam_pos, cam_quat = self.get_camera_pose_fn()
        grasp_cam = np.array(grasp["translation"], dtype=np.float64)
        grasp_depth = float(grasp.get("depth", 0.0))

        # AnyGrasp translation is already the grasp centre in the wrist camera
        # frame.  Do not push it down by ``depth`` by default; that was turning
        # valid candidates into below-ground targets.
        grasp_world_raw = optical_camera_to_world(cam_pos, cam_quat, grasp_cam)

        requested_depth_frac = float(os.environ.get("OMNILRS_ANYGRASP_DEPTH_FRAC", "0.0"))
        if abs(requested_depth_frac) > 1e-6 and not self.allow_depth_offset:
            my_print(
                f"🛡️ [AnyGrasp] 已忽略 OMNILRS_ANYGRASP_DEPTH_FRAC={requested_depth_frac:.3f}；"
                f"AnyGrasp translation 已是抓取中心，默认不再按 depth 下压。"
            )
            depth_frac = 0.0
        else:
            depth_frac = requested_depth_frac
        depth_offset_world_z = -grasp_depth * depth_frac  # negative = lower in world
        candidate_world = grasp_world_raw.copy()
        candidate_world[2] += depth_offset_world_z
        candidate_world[2] += self.z_bias  # manual trim (env OMNILRS_ANYGRASP_Z_BIAS)

        # For visualization we still show the un-offset grasp in camera frame
        # so the preview matches what AnyGrasp originally predicted.
        grasp_cam_offset = grasp_cam.copy()
        grasp_cam_offset[2] += grasp_depth * depth_frac  # same offset, for vis only

        self.grasp_width = float(np.clip(grasp.get("width", self.grasp_width), 0.025, 0.10))
        score = float(grasp.get("score", 0.0))

        # ── Convert AnyGrasp rotation_matrix from camera optical → world quat ──
        R_optical = np.array(grasp.get("rotation_matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                             dtype=np.float64)

        # ── Visualize the grasp pose BEFORE the arm moves ──
        # (we visualize the *adjusted* grasp so the preview matches execution)
        fx, fy, cx, cy = self._intrinsics(rgb)
        self._visualize_grasp_pose(rgb, grasp_cam_offset, R_optical, fx, fy, cx, cy, score=score)

        known_world = self._known_target_world()
        if known_world is not None:
            target_err = candidate_world - known_world
            target_err_xy = float(np.linalg.norm(target_err[:2]))
            max_target_xy = float(os.environ.get("OMNILRS_ANYGRASP_MAX_TARGET_XY_ERR", "0.050"))
            max_target_z = float(os.environ.get("OMNILRS_ANYGRASP_MAX_TARGET_Z_ERR", "0.080"))
            max_target_z_below = float(os.environ.get("OMNILRS_ANYGRASP_MAX_TARGET_Z_BELOW", "0.005"))
            my_print(
                f"🎯 [AnyGrasp] 候选与已知目标偏差: "
                f"xy={target_err_xy*1000:.1f}mm, z={target_err[2]*1000:+.1f}mm "
                f"(limit_xy={max_target_xy*1000:.0f}mm, "
                f"limit_z={max_target_z*1000:.0f}mm, below_limit={max_target_z_below*1000:.0f}mm)"
            )
            if (
                target_err_xy > max_target_xy
                or abs(float(target_err[2])) > max_target_z
                or float(target_err[2]) < -max_target_z_below
            ):
                raise RuntimeError(
                    f"anygrasp_candidate_far_from_known_target: "
                    f"xy={target_err_xy:.3f}, z={float(target_err[2]):+.3f}"
                )
            if self.anchor_known_xy:
                old_xy = candidate_world[:2].copy()
                candidate_world[:2] = known_world[:2]
                my_print(
                    f"🎯 [AnyGrasp] 已用已知月岩中心锚定抓取XY: "
                    f"candidate_xy=({old_xy[0]:.3f},{old_xy[1]:.3f}) -> "
                    f"known_xy=({candidate_world[0]:.3f},{candidate_world[1]:.3f})"
                )

        # AnyGrasp's translation is a grasp candidate on/near the object.  The
        # IK target in this demo is the two-fingertip control centre, which must
        # sit above that candidate by the gripper geometry offset.  The direct
        # world-grasp test that succeeds uses the same idea via
        # OMNILRS_ANYGRASP_DIRECT_WORLD_Z_OFFSET.
        grasp_world = candidate_world.copy()
        grasp_world[2] += self.tool_center_z_offset

        grasp_orient_full = anygrasp_rotation_to_world_quat(R_optical, cam_quat)
        if grasp_orient_full is not None:
            grasp_orient_world = grasp_quat_to_topdown_yaw(grasp_orient_full,
                                                           self.base_down_quat)
            my_print(
                f"🧭 [AnyGrasp] full quat → top-down projected (approach=world −Z, "
                f"yaw from AnyGrasp): "
                f"qw={grasp_orient_world[0]:.3f}, qx={grasp_orient_world[1]:.3f}, "
                f"qy={grasp_orient_world[2]:.3f}, qz={grasp_orient_world[3]:.3f}"
            )
            if not self.use_anygrasp_yaw:
                grasp_orient_world = None
                my_print("🧭 [AnyGrasp] 已按配置禁用 AnyGrasp yaw，回退 base_down_quat。")
        else:
            grasp_orient_world = None
            my_print("⚠️ [AnyGrasp] rotation_matrix 转换失败，回退到 base_down_quat")

        my_print(
            f"📏 [AnyGrasp] depth={grasp_depth*1000:.0f}mm, "
            f"world_z_offset={depth_offset_world_z*1000:+.0f}mm "
            f"(frac={depth_frac}), z_bias={self.z_bias*1000:+.0f}mm, "
            f"tool_center_z_offset={self.tool_center_z_offset*1000:+.0f}mm, "
            f"candidate_world=({candidate_world[0]:.3f},{candidate_world[1]:.3f},{candidate_world[2]:.3f})"
        )
        self._set_plan_from_world(grasp_world, source="anygrasp", score=score,
                                  grasp_cam=grasp_cam_offset,
                                  grasp_orientation_world=grasp_orient_world)

        try:
            Image.fromarray(rgb.astype(np.uint8)).save("debug_anygrasp_wrist_rgb.png")
        except Exception:
            pass

    def _go_to(self, target_pos, max_step=0.008):
        """Run.py-style incremental IK: move toward *target_pos* in small steps."""
        orient = self.base_down_quat if self.orientation_fallback_active or self.grasp_orientation is None else self.grasp_orientation
        current_pos, _ = self.ee_solver.compute_end_effector_pose()
        current_pos = np.array(current_pos, dtype=np.float64)
        target_pos = np.array(target_pos, dtype=np.float64)

        delta = target_pos - current_pos
        dist = float(np.linalg.norm(delta))
        if dist < 0.001:
            return True

        # Clamp to max_step to mimic run.py incremental control.
        step = delta * (max_step / max(dist, 1e-8))
        step_target = current_pos + step

        action, success = self.ee_solver.compute_inverse_kinematics(
            target_position=step_target,
            target_orientation=orient,
        )
        if (
            (not success)
            and self.allow_orientation_fallback
            and self.grasp_orientation is not None
            and not self.orientation_fallback_active
        ):
            fallback_action, fallback_success = self.ee_solver.compute_inverse_kinematics(
                target_position=step_target,
                target_orientation=self.base_down_quat,
            )
            if fallback_success:
                self.orientation_fallback_active = True
                action, success = fallback_action, True
                my_print("🧭 [AnyGrasp] AnyGrasp yaw 姿态IK失败，已回退为 base_down_quat 继续执行。")
        if success:
            self.robot.apply_action(action)
        return bool(success)

    def _go_delta_solver(self, delta_solver):
        ee_pos, _ = self.ee_solver.compute_end_effector_pose()
        return self._go_to(np.array(ee_pos, dtype=np.float64) + np.array(delta_solver, dtype=np.float64))

    def _solver_target_error(self, target_pos):
        current_pos, _ = self.ee_solver.compute_end_effector_pose()
        return float(np.linalg.norm(np.array(target_pos, dtype=np.float64) - np.array(current_pos, dtype=np.float64)))

    def _world_target_error(self):
        if self.grasp_world is None:
            return None, None, None
        tool_world = self._tool_center_world()
        delta_world = np.array(self.grasp_world, dtype=np.float64) - np.array(tool_world, dtype=np.float64)
        err_xy = float(np.linalg.norm(delta_world[:2]))
        return delta_world, err_xy, tool_world

    def _world_target_error_msg(self):
        delta_world, err_xy, tool_world = self._world_target_error()
        if delta_world is None:
            return ""
        return (
            f", tool_world_err_xy={err_xy*1000:.1f}mm, "
            f"tool_world_err_z={delta_world[2]*1000:+.1f}mm, "
            f"tool_world=({tool_world[0]:.3f},{tool_world[1]:.3f},{tool_world[2]:.3f})"
        )

    def _pre_world_alignment_ok(self):
        delta_world, err_xy, _ = self._world_target_error()
        if delta_world is None:
            return True
        pre_z_err = abs(float(delta_world[2]) + self.pre_height)
        if err_xy <= self.pre_world_xy_tolerance and pre_z_err <= self.pre_world_z_tolerance:
            return True
        my_print(
            f"❌ [AnyGrasp] pre-grasp solver到位但真实tool未悬在目标上方，拒绝下探避免扫石头: "
            f"tool_world_err_xy={err_xy*1000:.1f}mm > {self.pre_world_xy_tolerance*1000:.0f}mm "
            f"或 pre_z_err={pre_z_err*1000:.1f}mm > {self.pre_world_z_tolerance*1000:.0f}mm"
            f"{self._world_target_error_msg()}"
        )
        return False

    def _descend_world_alignment_ok(self, for_close=False):
        delta_world, err_xy, _ = self._world_target_error()
        if delta_world is None:
            return True
        if for_close:
            z_err = abs(float(delta_world[2]))
            if err_xy <= self.close_world_xy_tolerance and z_err <= self.close_world_z_tolerance:
                return True
            my_print(
                f"❌ [AnyGrasp] IK到达但真实tool未到抓取中心，拒绝闭合夹爪: "
                f"tool_world_err_xy={err_xy*1000:.1f}mm > {self.close_world_xy_tolerance*1000:.0f}mm "
                f"或 tool_world_z_abs_err={z_err*1000:.1f}mm > {self.close_world_z_tolerance*1000:.0f}mm"
                f"{self._world_target_error_msg()}"
            )
            return False
        if err_xy <= self.descend_world_xy_tolerance:
            return True
        my_print(
            f"❌ [AnyGrasp] 下探过程中真实tool水平偏差过大，立即停止避免把石头扫入车底: "
            f"tool_world_err_xy={err_xy*1000:.1f}mm > {self.descend_world_xy_tolerance*1000:.0f}mm"
            f"{self._world_target_error_msg()}"
        )
        return False

    def _sector_probe_delta_solver(self):
        """Solver-frame coarse motion. UR control convention in this demo: +X front, -Y right."""
        sector = self.target_sector.replace("-", "_")
        delta = np.zeros(3, dtype=np.float64)
        if "front" in sector or "forward" in sector:
            delta[0] += self.sector_front_step
        if "back" in sector:
            delta[0] -= self.sector_front_step
        if "right" in sector:
            delta[1] -= self.sector_right_step
        if "left" in sector:
            delta[1] += self.sector_right_step
        if np.linalg.norm(delta[:2]) < 1e-6:
            delta[0] += self.sector_front_step
        return delta

    def _wrist_visibility_align_step(self):
        proj = self._project_known_target_to_wrist()

        # ── Debug info on first frame ──
        if self.frame_count == 1:
            if proj is not None:
                cam_pos, _ = self.get_camera_pose_fn()
                inside_str = "图内" if proj["inside"] else "图外"
                uv_str = f"({proj['uv'][0]:.0f},{proj['uv'][1]:.0f})" if proj["uv"] is not None else "None"
                my_print(f"🔍 [AnyGrasp] 手腕对准起点: 石头投影={uv_str}({inside_str}), "
                         f"相机=({cam_pos[0]:.3f},{cam_pos[1]:.3f},{cam_pos[2]:.3f})")
                try:
                    wrist_rgb = proj["rgb"].astype(np.uint8)
                    dbg = Image.fromarray(wrist_rgb)
                    d = ImageDraw.Draw(dbg)
                    h_img, w_img = proj["shape"]
                    if proj["uv"] is not None:
                        u, v = proj["uv"]
                        r = 12
                        d.ellipse([(u - r, v - r), (u + r, v + r)], outline="red", width=4)
                        d.line([(u - 20, v), (u + 20, v)], fill="red", width=3)
                        d.line([(u, v - 20), (u, v + 20)], fill="red", width=3)
                    cx, cy = w_img // 2, h_img // 2
                    d.line([(cx - 30, cy), (cx + 30, cy)], fill="green", width=1)
                    d.line([(cx, cy - 30), (cx, cy + 30)], fill="green", width=1)
                    d.text((6, 6), f"proj={uv_str} {inside_str}", fill=(0, 255, 0))
                    dbg.save("debug_wrist_align_start.jpg")
                    my_print("📸 [AnyGrasp] 手腕对准起点图已保存: debug_wrist_align_start.jpg")
                except Exception:
                    pass
            else:
                my_print("🔍 [AnyGrasp] 手腕对准起点: 高杆目标投影失败，启动手腕DINO自主扫描")

        # ── Path A: rock is well inside and centred → use projection directly ──
        if proj is not None and proj["inside"]:
            h, w = proj["shape"]
            u, v = proj["uv"]
            margin = max(100, w * 0.15)
            centered = margin < u < (w - margin) and margin < v < (h - margin)
            if centered:
                self._set_roi_from_known_projection(proj)
                my_print(f"🎯 [AnyGrasp] 石头投影居中(pixel={u:.0f},{v:.0f})，直接设ROI，跳过DINO扫描。")
                return True
            elif self.frame_count % 8 == 1:
                my_print(f"🔍 [AnyGrasp] 石头在画面内但偏边缘(pixel={u:.0f},{v:.0f})，继续扫描使其居中...")

        # ── Initialise scan state once ──
        if not hasattr(self, '_scan_done'):
            self._scan_done = False
            self._scan_step = 0

        if self._scan_done:
            return False

        # ── Path B: try wrist DINO every few frames ──
        if self.frame_count % 4 == 1:
            if self._maybe_refresh_roi_from_wrist():
                my_print(f"🎯 [AnyGrasp] 手腕DINO扫描找到石头: roi={[round(v,1) for v in (self.roi_box or [])]}")
                self._scan_done = True
                return True

        # ── Guided scanning: use projection to move rock toward image centre ──
        scan_step_size = float(os.environ.get("OMNILRS_ANYGRASP_SCAN_STEP", "0.025"))
        scan_max = int(os.environ.get("OMNILRS_ANYGRASP_SCAN_MAX_STEPS", "20"))

        if self._scan_step >= scan_max:
            self._scan_done = True
            my_print(f"❌ [AnyGrasp] 扫描{scan_max}步完成，石头仍在画面外。")
            return False

        move_solver = np.zeros(3, dtype=np.float64)

        if proj is not None and proj["uv"] is not None:
            # Rock has a valid projection — move to bring it toward image centre.
            h, w = proj["shape"]
            u, v = proj["uv"]
            edge_margin = 120  # pixels from edge to trigger correction
            if v > h - edge_margin:
                move_solver[2] = -scan_step_size  # rock near bottom → lower wrist
            elif v < edge_margin:
                move_solver[2] = scan_step_size   # rock near top → raise wrist
            if u > w - edge_margin:
                move_solver[1] = -scan_step_size  # rock near right → move left
            elif u < edge_margin:
                move_solver[1] = scan_step_size   # rock near left → move right
            # Always add a small forward nudge for coverage
            move_solver[0] = scan_step_size * 0.3
        else:
            # No projection at all — blind forward/lateral sweep
            pattern_idx = self._scan_step % 4
            if pattern_idx == 0 or pattern_idx == 2:
                move_solver[0] = scan_step_size
            elif pattern_idx == 1:
                move_solver[1] = scan_step_size
            else:
                move_solver[1] = -scan_step_size

        self._scan_step += 1
        self._go_delta_solver(move_solver)

        if self.frame_count % 4 == 1:
            uv_str = "None"
            if proj is not None and proj["uv"] is not None:
                uv_str = f"({proj['uv'][0]:.0f},{proj['uv'][1]:.0f})"
            my_print(f"🔍 [AnyGrasp] 扫描: step={self._scan_step}/{scan_max}, "
                     f"rock_pixel={uv_str}, move=({move_solver[0]:+.3f},{move_solver[1]:+.3f},{move_solver[2]:+.3f})")
        return False

    def _start_local_grasp_fallback(self, reason):
        if self.selected_grasp is None or self.grasp_world is None:
            return False
        self.local_execution = True
        self.orientation_fallback_active = True
        self.local_approach_failures = 0
        self.local_descend_start_tool_world = None
        self.local_descend_xy_failures = 0
        self.local_descend_z_failures = 0
        self.state = "LOCAL_APPROACH"
        self.frame_count = 0
        tool_pos = self._tool_center_world()
        local_target = self._biased_local_grasp_world()
        err_xy = float(np.linalg.norm((np.array(local_target, dtype=np.float64) - tool_pos)[:2]))
        my_print(
            f"🧩 [AnyGrasp] {reason}；切换为局部执行: "
            f"当前tool到AnyGrasp抓取点XY误差={err_xy*1000:.1f}mm，"
            f"bias_solver=({self.local_grasp_bias_solver[0]:+.3f},{self.local_grasp_bias_solver[1]:+.3f},{self.local_grasp_bias_solver[2]:+.3f})，"
            f"局部执行固定使用 base_down_quat，先小步逼近，达标后再下探闭合。"
        )
        return True

    def _local_approach_step(self):
        local_target = self._biased_local_grasp_world()
        if local_target is None:
            return False
        tool_pos = self._tool_center_world()
        delta_world = np.array(local_target, dtype=np.float64) - tool_pos
        dist_xy = float(np.linalg.norm(delta_world[:2]))
        if dist_xy < self.local_approach_stop_xy:
            return True
        move_world = np.array([delta_world[0], delta_world[1], 0.0], dtype=np.float64)
        move_norm = float(np.linalg.norm(move_world[:2]))
        if move_norm < 1e-5:
            return True
        move_world[:2] = move_world[:2] / move_norm * min(self.local_approach_step, move_norm)
        move_solver = self._world_delta_to_solver_delta(move_world)
        move_solver[2] = 0.0
        if self.frame_count <= 3:
            my_print(
                f"↘️ [AnyGrasp] 局部XY指令: "
                f"delta_world_xy=({delta_world[0]:+.4f},{delta_world[1]:+.4f}), "
                f"move_solver=({move_solver[0]:+.4f},{move_solver[1]:+.4f},{move_solver[2]:+.4f}), "
                f"map={self.world_solver_map}"
            )
        ok = self._go_delta_solver(move_solver)
        if not ok:
            ok = self._go_delta_solver(move_solver * 0.35)
        if self.frame_count <= 3:
            tool_after = self._tool_center_world()
            actual_world = np.array(tool_after, dtype=np.float64) - np.array(tool_pos, dtype=np.float64)
            after_delta = np.array(local_target, dtype=np.float64) - np.array(tool_after, dtype=np.float64)
            my_print(
                f"↘️ [AnyGrasp] 局部XY实际: "
                f"tool_move_world=({actual_world[0]:+.4f},{actual_world[1]:+.4f},{actual_world[2]:+.4f}), "
                f"err_xy_after={np.linalg.norm(after_delta[:2])*1000:.1f}mm, ok={int(bool(ok))}"
            )
        if not ok:
            self.local_approach_failures += 1
        return ok or self.local_approach_failures < 4

    def _realign_after_ik_failure(self, reason):
        if self.pregrasp_realigns >= self.pregrasp_realign_limit:
            return False
        self.pregrasp_realigns += 1
        self.selected_grasp = None
        self.grasp_orientation = None
        self.orientation_fallback_active = False
        self.pre_pos = None
        self.grasp_pos = None
        self.lift_pos = None
        self.roi_box = None
        self.state = "WRIST_ALIGN"
        self.frame_count = self.sector_probe_frames + 1
        self.align_best_xy = float("inf")
        self.align_worse_count = 0
        self._depth_fail_streak = 0
        my_print(
            f"🔁 [AnyGrasp] {reason}；不判定抓取失败，继续手腕前探后重规划 "
            f"({self.pregrasp_realigns}/{self.pregrasp_realign_limit})。"
        )
        return True

    def step(self):
        if self.state in ["IDLE", "DONE", "FAILED"]:
            return self.state

        if self.state == "OPEN":
            self.gripper.open()
            if (not self.direct_world_grasp) and (not self._check_service()):
                self.state = "FAILED"
                return self.state
            self.state = "ANYGRASP_PLAN" if self.direct_world_grasp else "WRIST_ALIGN"
            self.frame_count = 0
            return self.state

        if self.state == "WRIST_ALIGN":
            self.frame_count += 1
            ready = self._wrist_visibility_align_step()
            if ready:
                self.state = "ANYGRASP_PLAN"
                self.frame_count = 0
            elif self.frame_count > self.align_frames:
                my_print("❌ [AnyGrasp] 手腕前探后仍没有获得可信石头视野，拒绝调用AnyGrasp抓地/抓车体。")
                self.state = "FAILED"
            return self.state

        if self.state == "ANYGRASP_PLAN":
            try:
                self._plan_once()
                if self.local_first and self.allow_local_fallback:
                    if self._start_local_grasp_fallback("local_first已启用，跳过绝对pre-grasp IK，按run.py风格小步逼近"):
                        return self.state
                    my_print("⚠️ [AnyGrasp] local_first启用但缺少局部目标，回退绝对pre-grasp。")
                self.state = "PRE_GRASP"
                self.frame_count = 0
            except Exception as e:
                self.plan_attempts += 1
                my_print(f"⚠️ [AnyGrasp] 规划失败 attempt={self.plan_attempts}: {e}")
                if self.plan_attempts >= 3:
                    self.state = "FAILED"
            return self.state

        if self.state == "PRE_GRASP":
            ok = self._go_to(self.pre_pos)
            self.frame_count += 1
            err = self._solver_target_error(self.pre_pos)
            if self.frame_count == 1:
                my_print(
                    f"⬆️ [AnyGrasp] 前置位: pre=({self.pre_pos[0]:.3f},{self.pre_pos[1]:.3f},{self.pre_pos[2]:.3f}), "
                    f"reach_tol={self.reach_tolerance*1000:.0f}mm"
                )
            elif self.frame_count % 20 == 1:
                my_print(
                    f"⬆️ [AnyGrasp] 前置位接近中: err={err*1000:.1f}mm, "
                    f"ok={int(bool(ok))}{self._world_target_error_msg()}"
                )
            if not ok and self.frame_count > 10:
                if self.allow_local_fallback and self._start_local_grasp_fallback("pre-grasp IK不可达"):
                    return self.state
                else:
                    my_print("❌ [AnyGrasp] pre-grasp IK不可达。")
                    self.state = "FAILED"
            elif err <= self.reach_tolerance:
                if not self._pre_world_alignment_ok():
                    self.state = "FAILED"
                    return self.state
                self.state = "DESCEND"
                self.frame_count = 0
            elif self.frame_count > self.pre_max_frames:
                my_print(f"❌ [AnyGrasp] pre-grasp 未到位，拒绝继续下探: err={err*1000:.1f}mm")
                self.state = "FAILED"
            return self.state

        if self.state == "LOCAL_APPROACH":
            self.frame_count += 1
            if self.frame_count == 1:
                my_print(
                    f"↘️ [AnyGrasp] 局部XY小步逼近: step={self.local_approach_step:.4f}m, "
                    f"stop_xy={self.local_approach_stop_xy*1000:.0f}mm, "
                    f"max_frames={self.local_approach_frames}"
                )
            keep_trying = self._local_approach_step()
            if self.frame_count % 8 == 1 and self.grasp_world is not None:
                tool_pos = self._tool_center_world()
                local_target = self._biased_local_grasp_world()
                err_xy = float(np.linalg.norm((np.array(local_target, dtype=np.float64) - tool_pos)[:2]))
                my_print(f"↘️ [AnyGrasp] 局部逼近中: tool_err_xy={err_xy*1000:.1f}mm, ik_failures={self.local_approach_failures}")
            if self.frame_count >= self.local_approach_frames or not keep_trying:
                tool_pos = self._tool_center_world()
                local_target = self._biased_local_grasp_world()
                err_xy = float(np.linalg.norm((np.array(local_target, dtype=np.float64) - tool_pos)[:2]))
                max_descend_xy = float(os.environ.get("OMNILRS_ANYGRASP_LOCAL_MAX_DESCEND_XY_ERR", "0.025"))
                if err_xy > max_descend_xy:
                    my_print(
                        f"❌ [AnyGrasp] 局部逼近不足，不进入下探闭合: "
                        f"tool_err_xy={err_xy*1000:.1f}mm > {max_descend_xy*1000:.0f}mm"
                    )
                    if not self._realign_after_ik_failure("局部XY逼近不足"):
                        self.state = "FAILED"
                    return self.state
                self.state = "LOCAL_DESCEND"
                self.frame_count = 0
            return self.state

        if self.state == "LOCAL_DESCEND":
            self.frame_count += 1

            if self.frame_count == 1:
                self.local_descend_start_tool_world = self._tool_center_world()
                self.local_descend_xy_failures = 0
                self.local_descend_z_failures = 0

                tool_pos = self._tool_center_world()
                local_target = self._biased_local_grasp_world()
                err_vec = np.array(local_target, dtype=np.float64)[:2] - tool_pos[:2]
                err_xy_start = float(np.linalg.norm(err_vec))
                self._descend_z_steps = max(1, self.local_descend_frames)
                base_descend_total = self.local_descend_step * self.local_descend_frames
                required_down = max(
                    0.0,
                    float(tool_pos[2] - np.array(local_target, dtype=np.float64)[2] - self.local_descend_clearance)
                )
                descend_total = min(max(base_descend_total, required_down), self.local_descend_max_total)
                self._descend_required_down = required_down
                self._descend_total = descend_total
                self._descend_solver_step_z = -descend_total / self._descend_z_steps

                my_print(
                    f"⬇️ [AnyGrasp] 绝对抓取位不可达，执行 run.py 风格 solver-Z 局部下探: "
                    f"frames={self.local_descend_frames}, z_steps={self._descend_z_steps}, "
                    f"err_xy_start={err_xy_start*1000:.0f}mm, "
                    f"target_z={np.array(local_target, dtype=np.float64)[2]:.3f}, "
                    f"required_down={required_down:.3f}m, "
                    f"cmd_down_total={descend_total:.3f}m, "
                    f"solver_step_z={self._descend_solver_step_z:+.4f}m"
                )

            if self.frame_count <= self.local_descend_frames:
                tool_before = self._tool_center_world()
                local_target = self._biased_local_grasp_world()
                descend_err_xy = 0.0
                correction_solver = np.zeros(3, dtype=np.float64)
                if local_target is not None:
                    err_vec_xy = np.array(local_target, dtype=np.float64)[:2] - np.array(tool_before, dtype=np.float64)[:2]
                    descend_err_xy = float(np.linalg.norm(err_vec_xy))
                    if descend_err_xy > self.local_descend_abort_xy:
                        my_print(
                            f"❌ [AnyGrasp] 局部下探前XY漂移过大，停止下探避免扫石头: "
                            f"err_xy={descend_err_xy*1000:.1f}mm > {self.local_descend_abort_xy*1000:.0f}mm"
                        )
                        if not self._realign_after_ik_failure("局部下探XY漂移过大"):
                            self.state = "FAILED"
                        return self.state
                    if descend_err_xy > self.local_descend_xy_deadband:
                        correction_world = np.array([err_vec_xy[0], err_vec_xy[1], 0.0], dtype=np.float64)
                        correction_norm = float(np.linalg.norm(correction_world[:2]))
                        correction_world[:2] = (
                            correction_world[:2] / max(correction_norm, 1e-8)
                            * min(self.local_descend_xy_correction_step, correction_norm)
                        )
                        correction_solver = self._world_delta_to_solver_delta(correction_world)
                        correction_solver[2] = 0.0

                move_solver = correction_solver + np.array([0.0, 0.0, self._descend_solver_step_z], dtype=np.float64)
                ok = self._go_delta_solver(move_solver)
                if not ok:
                    ok = self._go_delta_solver(np.array([0.0, 0.0, -0.0010], dtype=np.float64))
                if not ok:
                    self.local_descend_z_failures += 1
                if self.local_descend_start_tool_world is not None:
                    actual_world = self._tool_center_world() - self.local_descend_start_tool_world
                    actual_world_xy = float(np.linalg.norm(actual_world[:2]))
                    max_xy_drift = float(os.environ.get("OMNILRS_ANYGRASP_LOCAL_MAX_WORLD_XY_DRIFT", "0.030"))
                    if actual_world_xy > max_xy_drift:
                        my_print(
                            f"❌ [AnyGrasp] 局部世界Z下探发生过大水平漂移: "
                            f"world_xy={actual_world_xy*1000:.1f}mm > {max_xy_drift*1000:.0f}mm；"
                            f"中止本次闭合，避免空夹。"
                        )
                        self.state = "FAILED"
                        return self.state
                if self.frame_count % 12 == 1 or self.frame_count == self.local_descend_frames:
                    actual_msg = "actual=unknown"
                    if self.local_descend_start_tool_world is not None:
                        actual_world = self._tool_center_world() - self.local_descend_start_tool_world
                        actual_solver = self._world_delta_to_solver_delta(actual_world)
                        actual_world_xy = float(np.linalg.norm(actual_world[:2]))
                        actual_world_down = -float(actual_world[2])
                        actual_msg = (
                            f"actual_world_xy={actual_world_xy*1000:.1f}mm, "
                            f"actual_world_down={actual_world_down*1000:.1f}mm, "
                            f"err_xy_now={descend_err_xy*1000:.1f}mm, "
                            f"xy_corr=({correction_solver[0]*1000:.1f},{correction_solver[1]*1000:.1f})mm, "
                            f"solver_forward={actual_solver[0]*1000:.1f}mm, "
                            f"solver_right={-actual_solver[1]*1000:.1f}mm"
                        )
                    my_print(
                        f"⬇️ [AnyGrasp] 局部下探进度: frame={self.frame_count}/{self.local_descend_frames}, "
                        f"{actual_msg}, ik_fail_xy={self.local_descend_xy_failures}, "
                        f"ik_fail_z={self.local_descend_z_failures}"
                    )
            else:
                final_msg = ""
                actual_world_down = None
                if self.local_descend_start_tool_world is not None:
                    actual_world = self._tool_center_world() - self.local_descend_start_tool_world
                    actual_solver = self._world_delta_to_solver_delta(actual_world)
                    actual_world_xy = float(np.linalg.norm(actual_world[:2]))
                    actual_world_down = -float(actual_world[2])
                    final_msg = (
                        f" 实际累计: world_xy={actual_world_xy*1000:.1f}mm, "
                        f"world_down={actual_world_down*1000:.1f}mm, "
                        f"solver_forward={actual_solver[0]*1000:.1f}mm, "
                        f"solver_right={-actual_solver[1]*1000:.1f}mm, "
                        f"ik_fail_xy={self.local_descend_xy_failures}, ik_fail_z={self.local_descend_z_failures}."
                    )
                configured_min_down = float(os.environ.get("OMNILRS_ANYGRASP_LOCAL_MIN_DESCEND_M", "0.025"))
                required_down = self._descend_required_down if self._descend_required_down is not None else 0.0
                min_down = max(configured_min_down, min(required_down * 0.75, self.local_descend_max_total * 0.90))
                if actual_world_down is not None and actual_world_down < min_down:
                    my_print(
                        f"❌ [AnyGrasp] 局部下探实际位移不足，不闭合夹爪。"
                        f"world_down={actual_world_down*1000:.1f}mm < {min_down*1000:.0f}mm.{final_msg}"
                    )
                    if not self._realign_after_ik_failure("局部下探不足"):
                        self.state = "FAILED"
                    return self.state
                my_print(f"🤏 [AnyGrasp] 局部下探完成，闭合夹爪。{final_msg}")
                self.closed_on_valid_target = True
                self.state = "CLOSE"
                self.frame_count = 0
            return self.state

        if self.state == "DESCEND":
            if self.frame_count == 0 and not self._descend_world_alignment_ok(for_close=False):
                self.state = "FAILED"
                return self.state
            ok = self._go_to(self.grasp_pos)
            self.frame_count += 1
            err = self._solver_target_error(self.grasp_pos)
            if self.frame_count == 1:
                my_print(
                    f"⬇️ [AnyGrasp] 下探到 AnyGrasp 抓取位: "
                    f"grasp=({self.grasp_pos[0]:.3f},{self.grasp_pos[1]:.3f},{self.grasp_pos[2]:.3f}), "
                    f"reach_tol={self.reach_tolerance*1000:.0f}mm"
                )
            elif self.frame_count % 20 == 1:
                my_print(
                    f"⬇️ [AnyGrasp] 抓取位接近中: err={err*1000:.1f}mm, "
                    f"ok={int(bool(ok))}{self._world_target_error_msg()}"
                )
            if not self._descend_world_alignment_ok(for_close=False):
                self.state = "FAILED"
                return self.state
            if not ok and self.frame_count > 10:
                if self.allow_local_fallback and self._start_local_grasp_fallback("grasp IK不可达"):
                    return self.state
                else:
                    my_print("❌ [AnyGrasp] grasp IK不可达。")
                    self.state = "FAILED"
            elif err <= self.reach_tolerance:
                if not self._descend_world_alignment_ok(for_close=True):
                    self.state = "FAILED"
                    return self.state
                my_print(f"🤏 [AnyGrasp] 闭合前真实tool误差确认{self._world_target_error_msg()}")
                self.closed_on_valid_target = True
                self.state = "CLOSE"
                self.frame_count = 0
            elif self.frame_count > self.descend_max_frames:
                my_print(
                    f"❌ [AnyGrasp] 抓取位未到位，不闭合夹爪: "
                    f"err={err*1000:.1f}mm{self._world_target_error_msg()}"
                )
                self.state = "FAILED"
            return self.state

        if self.state == "CLOSE":
            if self.frame_count == 0:
                self.close_target_width = self._compute_soft_close_width()
                self.hold_target_width = self._compute_lift_hold_width()
                self.close_command_width = float(np.clip(self.close_start_width, self.close_target_width, 0.14))
                my_print(
                    f"🤏 [AnyGrasp] 软闭合夹爪: raw_width={self.grasp_width:.3f}, "
                    f"acquire_width={self.close_target_width:.3f}, "
                    f"hold_width={self.hold_target_width:.3f}, "
                    f"ratio={self.close_width_ratio:.2f}, margin={self.close_width_margin*1000:.0f}mm, "
                    f"frames={max(1, self.close_frames)}"
                )

            close_frames = max(1, int(self.close_frames))
            start_width = float(np.clip(self.close_start_width, self.close_target_width, 0.14))
            alpha = min(1.0, float(self.frame_count + 1) / float(close_frames))
            command_width = start_width + (self.close_target_width - start_width) * alpha
            self.close_command_width = float(command_width)
            self.gripper.close(command_width)
            self.frame_count += 1
            if self.frame_count % 8 == 0 or self.frame_count == close_frames:
                my_print(
                    f"🤏 [AnyGrasp] 软闭合进度: frame={self.frame_count}/{close_frames}, "
                    f"command_width={self.close_command_width:.3f}"
                )
            if self.frame_count >= close_frames:
                self.gripper.close(self.close_target_width)
                self.state = "LIFT"
                self.frame_count = 0
            return self.state

        if self.state == "LIFT":
            self._command_lift_hold_width()
            if self.local_execution:
                ok = self._go_delta_solver(np.array([0.0, 0.0, self.local_lift_step], dtype=np.float64))
                self.frame_count += 1
                if self.frame_count == 1:
                    my_print(
                        f"⬆️ [AnyGrasp] 局部闭合后按 run.py 风格 solver-Z 上抬: step_z={self.local_lift_step:.4f}m, "
                        f"frames={self.local_lift_frames}"
                    )
                if not ok:
                    self._go_delta_solver(np.array([0.0, 0.0, 0.0015], dtype=np.float64))
                if self.frame_count > self.local_lift_frames:
                    self.state = "DONE"
                return self.state

            ok = self._go_to(self.lift_pos, max_step=self.lift_max_step)
            self.frame_count += 1
            if ok:
                self.lift_reached_frames += 1
            else:
                self.lift_reached_frames = 0
            if self.frame_count == 1:
                my_print(
                    f"⬆️ [AnyGrasp] 闭合后沿世界Z垂直上抬: lift=({self.lift_pos[0]:.3f},{self.lift_pos[1]:.3f},{self.lift_pos[2]:.3f})"
                )
            if not ok and self.frame_count > 10:
                my_print("⚠️ [AnyGrasp] lift IK不可达，结束进入验收。")
                self.state = "DONE"
            elif (
                ok
                and self.frame_count >= self.lift_min_frames
                and self.lift_reached_frames >= self.lift_settle_frames
            ):
                my_print(
                    f"✅ [AnyGrasp] 抬升目标已稳定: frame={self.frame_count}, "
                    f"settle={self.lift_reached_frames}/{self.lift_settle_frames}"
                )
                self.state = "DONE"
            elif self.frame_count > self.lift_max_frames:
                self.state = "DONE"
            return self.state

        return self.state
# ==========================================
# 🚀 3. 主程序入口与仿真循环
# ==========================================
@hydra.main(config_name="config", config_path="cfg")
def run(cfg: DictConfig):
    cfg_container = OmegaConf.to_container(cfg, resolve=True)
    cfg_inst = instantiateConfigs(cfg_container)
    
    my_print("\n⏳ Isaac Sim 物理引擎启动中...\n")
    SM, simulation_app = startSim(cfg_inst)

    logging.getLogger("omni.physx.plugin").setLevel(logging.ERROR)
    carb.settings.get_settings().set("/log/level", "error")

    import omni.usd
    from pxr import UsdPhysics, UsdGeom, PhysxSchema, Gf, Sdf
    import omni.timeline
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.prims import RigidPrim
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.extensions import get_extension_path_from_name
    from omni.isaac.sensor import Camera
    try:
        from isaacsim.core.utils.xforms import get_world_pose
    except Exception:
        from omni.isaac.core.utils.xforms import get_world_pose
    try:
        from isaacsim.core.utils.types import ArticulationAction
    except Exception:
        ArticulationAction = None

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

    print("\n================ 🔍 底盘车轮节点排查 ==================")
    wheel_prims = []
    
    for prim in stage.Traverse():
        # 第一关：名字里只要有 "wheel" 的，统统抓过来审问！
        if "wheel" in prim.GetName().lower():
            
            # 获取它在底层 API 里最真实的类型名字
            real_type_name = prim.GetTypeName()
            
            # 💡 [核心修改] 模糊匹配！只要名字里包含 "PhysicsRevolute" 就行，管它后面带不带 Joint！
            if "PhysicsRevolute" in real_type_name:
                wheel_prims.append(prim)
                
                # 把真相打印出来，让你我都能看到它到底叫啥！
                my_print(f"👀 成功抓取车轮: {prim.GetName()} | 真实类型: '{real_type_name}'")
                
                # ==========================================
                # 注入 1000 亿的绝对液压驻车制动！
                # ==========================================
                drive = UsdPhysics.DriveAPI.Get(prim, "angular") or UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive.GetTargetVelocityAttr().Set(0.0)
                drive.GetMaxForceAttr().Set(1e8)
                drive.GetDampingAttr().Set(1e6)
                drive.GetStiffnessAttr().Set(0.0)  # 刚度归零
                
    # 🚨 最终宣判
    my_print(f"✅ 最终成功注入液压驻车制动的车轮数量: {len(wheel_prims)}")
    print("=====================================================\n")

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
    my_print("深度相机已经挂载")

    def discover_stage_cameras():
        camera_paths = []
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Camera) or prim.GetTypeName() == "Camera":
                camera_paths.append(prim.GetPath().pathString)
        my_print("📷 [Camera Dump] 当前 stage 内 Camera prim:")
        for path in camera_paths:
            my_print(f"   - {path}")
        return camera_paths

    def resolve_camera_path(path, camera_paths, label):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            return None
        if prim.IsA(UsdGeom.Camera) or prim.GetTypeName() == "Camera":
            my_print(f"✅ [WristCam] {label} 指向 Camera prim: {path}")
            return path
        prefix = path.rstrip("/") + "/"
        for cam_path in camera_paths:
            if cam_path.startswith(prefix):
                my_print(f"✅ [WristCam] {label} 是安装座，使用子 Camera: {cam_path}")
                return cam_path
        return None

    def set_local_translate(prim, position):
        xform = UsdGeom.Xformable(prim)
        translate_op = None
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break
        if translate_op is None:
            translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))

    def pick_or_create_wrist_camera(camera_paths):
        override = os.environ.get("OMNILRS_ARM_CAMERA_PATH", "").strip()
        if override:
            resolved = resolve_camera_path(override, camera_paths, "OMNILRS_ARM_CAMERA_PATH")
            if resolved:
                return resolved

        known_mounts = [
            "/Robots/husky/jackal/ur3e/wrist_3_link/camera_mount_wrist",
            "/Robots/husky/jackal/ur10e/wrist_3_link/camera_mount_wrist",
            "/Robots/husky/jackal/ur3e/tool0",
            "/Robots/husky/jackal/ur10e/tool0",
            "/Robots/husky/jackal/ur3e/wrist_3_link",
            "/Robots/husky/jackal/ur10e/wrist_3_link",
        ]
        for mount in known_mounts:
            resolved = resolve_camera_path(mount, camera_paths, "已知手腕安装点")
            if resolved:
                return resolved

        exclude = ["mast_base", "rsd455", "pseudo_depth", "omnivision", "camera_mount_head", "vlp16"]
        prefer = ["wrist", "tool", "hand", "gripper", "robotiq", "finger", "ee", "end_effector"]
        for path in camera_paths:
            low = path.lower()
            if any(k in low for k in prefer) and not any(k in low for k in exclude):
                my_print(f"✅ [WristCam] 自动选择手腕/夹爪相机: {path}")
                return path

        mount_path = None
        for prim in stage.Traverse():
            path = prim.GetPath().pathString
            low = path.lower()
            if "/robots/husky/jackal" not in low:
                continue
            if low.endswith("/tool0"):
                mount_path = path
                break
            if mount_path is None and low.endswith("/wrist_3_link"):
                mount_path = path
        if mount_path is None:
            my_print("⚠️ [WristCam] 找不到 wrist_3_link/tool0，无法创建临时手腕相机。")
            return None

        cam_path = mount_path.rstrip("/") + "/codex_wrist_camera"
        cam_prim = stage.DefinePrim(cam_path, "Camera")
        set_local_translate(cam_prim, [0.04, 0.0, 0.0])
        cam_schema = UsdGeom.Camera(cam_prim)
        try:
            cam_schema.CreateFocalLengthAttr().Set(18.0)
            cam_schema.CreateFocusDistanceAttr().Set(0.35)
        except Exception:
            pass
        my_print(f"✅ [WristCam] stage 中没有现成手腕相机，已创建临时 Camera: {cam_path}")
        return cam_path

    stage_camera_paths = discover_stage_cameras()
    wrist_cam_prim_path = pick_or_create_wrist_camera(stage_camera_paths)
    wrist_cam = None
    if wrist_cam_prim_path:
        wrist_cam = Camera(prim_path=wrist_cam_prim_path, resolution=(640, 480))
        wrist_cam.initialize()
        wrist_cam.set_resolution((640, 480))
        try:
            wrist_cam.add_distance_to_image_plane_to_frame()
            my_print("✅ [WristCam] 已开启手腕相机 distance_to_image_plane 深度输出。")
        except Exception as e:
            my_print(f"⚠️ [WristCam] 手腕相机深度输出开启失败，AnyGrasp将无法使用RGB-D: {e}")
        my_print(f"✅ [WristCam] 已绑定手腕视觉相机: {wrist_cam_prim_path}")
    else:
        my_print("❌ [WristCam] 没有手腕相机，抓取阶段会失败。")
    
    # ========================================================
    # 🎯 5. 不穿模月岩加载
    # 调试模式按用户标定坐标初始化岩石；默认不再二次移动，避免目标从标定点瞬移。
    # ========================================================
    GRASP_DEBUG_MODE = os.environ.get("OMNILRS_GRASP_DEBUG", "1").lower() not in ["0", "false", "no", "off"]
    DEBUG_ROCK_FORWARD = float(os.environ.get("OMNILRS_DEBUG_ROCK_FORWARD", "1.05"))
    DEBUG_ROCK_LATERAL = float(os.environ.get("OMNILRS_DEBUG_ROCK_LATERAL", "0.00"))
    DEBUG_ROCK_Z = float(os.environ.get("OMNILRS_DEBUG_ROCK_Z", "0.13089"))
    DEBUG_ROCK_FRAME = os.environ.get("OMNILRS_DEBUG_ROCK_FRAME", "base").strip().lower()
    DEBUG_ROCK_SETTLE_FRAMES = int(os.environ.get("OMNILRS_DEBUG_ROCK_SETTLE_FRAMES", "45"))
    DEBUG_RELOCATE_ROCK = os.environ.get("OMNILRS_DEBUG_RELOCATE_ROCK", "0").lower() in ["1", "true", "yes", "on"]
    DEBUG_BASE_YAW_DEG = float(os.environ.get("OMNILRS_DEBUG_BASE_YAW_DEG", "-70.0"))
    DEBUG_PRE_YAW_SETTLE_FRAMES = int(os.environ.get("OMNILRS_DEBUG_PRE_YAW_SETTLE_FRAMES", "60"))
    DEBUG_BASE_YAW_SETTLE_FRAMES = int(os.environ.get("OMNILRS_DEBUG_BASE_YAW_SETTLE_FRAMES", "35"))
    DEBUG_WRIST_FIRST = os.environ.get("OMNILRS_DEBUG_WRIST_FIRST", "1").lower() not in ["0", "false", "no", "off"]
    IK_CALIBRATE_MODE = os.environ.get("OMNILRS_IK_CALIBRATE", "0").lower() in ["1", "true", "yes", "on"]
    INIT_ROCK_POS = np.array([3.3014, 4.2414, 0.13089], dtype=np.float64)
    rock_prim_path = "/World/SRB_Apollo_Rock"
    apollo_rock_path = os.path.expanduser("~/Luna-VLA/ript-vla/space_robotics_bench/assets/srb_assets/object/rock/apollo_sample1.usdz")
    
    test_rock = None
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

        test_rock = RigidPrim(prim_path=rock_prim_path, name="srb_rock", position=INIT_ROCK_POS, scale=np.array([1.0, 1.0, 1.0]), mass=0.5)
        test_rock.initialize()
        PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath(rock_prim_path)).CreateEnableCCDAttr().Set(True)
        if GRASP_DEBUG_MODE:
            my_print(
                f"🧪 [GraspDebug] 月岩按标定坐标初始化: INIT_ROCK_POS={INIT_ROCK_POS.tolist()}；"
                "默认保持该标定点不再二次移动。"
            )

    
    
    # ========================================================
    # 🟢 双模初始化与终极状态机主循环
    # ========================================================
    vlm_brain = VLMAgent()
    dino_tracker = DinoTracker(depth_cam=depth_cam)
    gripper = RobotiqGripperDriver(my_robot, action_cls=ArticulationAction)
    wrist_probe_executor = WristVisionSweepGraspExecutor(my_robot, ee_solver, gripper, wrist_cam, dino_tracker, vlm_brain)
    grasp_executor = wrist_probe_executor
    USE_ANYGRASP = os.environ.get("OMNILRS_USE_ANYGRASP", "1").lower() not in ["0", "false", "no", "off"]
    
    base_instruction = "你的正前方有一块小岩石。请找到这块岩石并用机械臂抓取。"
    
    # 🔴 核心状态机变量
    ROBOT_STATE = "IK_CALIBRATE" if IK_CALIBRATE_MODE else ("GRASP_DEBUG_INIT" if GRASP_DEBUG_MODE else "SEARCHING")
    locked_roi_box = None      # 锁定后的局部跟踪框 [xmin, ymin, xmax, ymax]
    latest_rgb_img = None
    final_cam_pos = None
    locked_target_prompt = "the small rock."
    locked_base_pos = None
    locked_base_ori = None
    locked_wheel_pos = None
    grasp_retry_count = 0
    hold_grasp_after_lift = os.environ.get("OMNILRS_GRASP_HOLD_AFTER_LIFT", "1").lower() not in ["0", "false", "no", "off"]
    hold_arm_after_lift = os.environ.get("OMNILRS_GRASP_HOLD_ARM", "0").lower() in ["1", "true", "yes", "on"]
    retry_grasp_on_fail = os.environ.get("OMNILRS_GRASP_RETRY_ON_FAIL", "0").lower() in ["1", "true", "yes", "on"]
    hold_gripper_refresh_frames = int(os.environ.get("OMNILRS_GRASP_HOLD_GRIPPER_REFRESH", "0"))
    hold_arm_refresh_frames = max(1, int(os.environ.get("OMNILRS_GRASP_HOLD_ARM_REFRESH", "5")))
    hold_arm_zero_vel_frames = max(0, int(os.environ.get("OMNILRS_GRASP_HOLD_ARM_ZERO_VEL_FRAMES", "12")))
    hold_arm_zero_vel_warmup = max(0, int(os.environ.get("OMNILRS_GRASP_HOLD_ARM_ZERO_VEL_WARMUP", "60")))
    hold_arm_joint_positions = None
    hold_gripper_width = None
    hold_frame_count = 0
    anygrasp_target_world = None
    anygrasp_target_source = "unset"
    base_lock_joint_path = "/World/GraspBaseLockJoint"
    grasp_start_joint_positions = None
    restore_target_joint_positions = None
    restore_next_state = None
    restore_frame_count = 0
    debug_init_attempts = 0
    debug_pre_yaw_settle_frames = DEBUG_PRE_YAW_SETTLE_FRAMES
    debug_base_rotated = False
    debug_base_yaw_settle_frames = 0
    debug_base_crept_forward = False
    DEBUG_BASE_CREEP_FORWARD_M = float(os.environ.get("OMNILRS_DEBUG_BASE_CREEP_FORWARD_M", "0.0"))
    debug_rock_relocated = False
    debug_rock_settle_frames = 0

    startup_frames, step_counter = 0, 0
    VISION_INTERVAL = 15 
    WORLD_ALIGN_DEBUG = os.environ.get("OMNILRS_DEBUG_WORLD_ALIGN", "1").lower() not in ["0", "false", "no", "off"]
    WORLD_ALIGN_OFFSET = np.array([
        float(os.environ.get("OMNILRS_DEBUG_WORLD_ALIGN_X_OFFSET", "0.0")),
        float(os.environ.get("OMNILRS_DEBUG_WORLD_ALIGN_Y_OFFSET", "0.0")),
        0.0,
    ], dtype=np.float64)
    



    # ==========================================
    # 🚀 [物理黑客] 注入宇宙级防滑摩擦材质
    # ==========================================
    from pxr import UsdPhysics, UsdShade, Tf

    # 1. 定义全局物理材质定义域
    material_path = "/World/SuperFrictionMaterial"
    if not stage.GetPrimAtPath(material_path).IsValid():
        UsdShade.Material.Define(stage, material_path)
        phys_mat = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(material_path))
        
        # 设定极端物理摩擦系数以对抗月壤陡坡的重力下滑分力
        phys_mat.CreateStaticFrictionAttr().Set(1000.0)
        phys_mat.CreateDynamicFrictionAttr().Set(1000.0)
        phys_mat.CreateRestitutionAttr().Set(0.0) 

    # 2. 获取 Articulation 根节点并执行物理目的（Physics Purpose）材质绑定
    robot_prim = stage.GetPrimAtPath("/Robots/husky/jackal")
    if robot_prim.IsValid():
        binding_api = UsdShade.MaterialBindingAPI.Apply(robot_prim)
        super_mat = UsdShade.Material(stage.GetPrimAtPath(material_path))
        
        # 将材质绑定明确限定在物理管线（physics）中，不干涉视觉 Shading
        binding_api.Bind(super_mat, materialPurpose="physics")
        print("🎯 [物理系统配置] 极限摩擦材质成功绑定至 Articulation 根节点，目的管线：physics")
    else:
        print("❌ [配置异常] 无法定位小车根节点，请核对路径定义。")

    def find_robot_base_link_path():
        preferred = "/Robots/husky/jackal/base_link"
        if stage.GetPrimAtPath(preferred).IsValid():
            return preferred
        for prim in stage.Traverse():
            path = prim.GetPath().pathString
            if path.startswith("/Robots/husky/jackal") and prim.GetName() == "base_link":
                return path
        return None

    def release_grasp_base_lock():
        if stage.GetPrimAtPath(base_lock_joint_path).IsValid():
            stage.RemovePrim(base_lock_joint_path)
            my_print("🔓 [BaseLock] 已释放抓取底盘固定关节，底盘可重新移动。")

    def create_grasp_base_lock():
        release_grasp_base_lock()
        base_link_path = find_robot_base_link_path()
        if base_link_path is None:
            my_print("⚠️ [BaseLock] 找不到 base_link，无法创建物理固定关节。")
            return False
        try:
            base_pos, base_ori = get_world_pose(base_link_path)
            base_pos = np.array(base_pos, dtype=np.float64)
            base_ori = quat_wxyz_normalize(base_ori)
            joint = UsdPhysics.FixedJoint.Define(stage, base_lock_joint_path)
            joint.CreateBody1Rel().SetTargets([Sdf.Path(base_link_path)])
            joint.CreateLocalPos0Attr().Set(Gf.Vec3f(float(base_pos[0]), float(base_pos[1]), float(base_pos[2])))
            joint.CreateLocalRot0Attr().Set(Gf.Quatf(float(base_ori[0]), Gf.Vec3f(float(base_ori[1]), float(base_ori[2]), float(base_ori[3]))))
            joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
            joint.CreateBreakForceAttr().Set(1e20)
            joint.CreateBreakTorqueAttr().Set(1e20)
            my_robot.set_linear_velocity(np.zeros(3))
            my_robot.set_angular_velocity(np.zeros(3))
            my_print(f"🔒 [BaseLock] 已用物理FixedJoint锁定底盘: {base_link_path}")
            return True
        except Exception as e:
            my_print(f"⚠️ [BaseLock] 创建底盘固定关节失败: {e}")
            return False

    def freeze_base_and_wheels(hard_pose_lock=False):
        if hard_pose_lock:
            my_print("⚠️ [BaseLock] 已禁用每帧 set_world_pose；抓取阶段使用 FixedJoint 物理锁底盘。")
        for prim in wheel_prims:
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if drive:
                drive.GetTargetVelocityAttr().Set(0.0)
                drive.GetMaxForceAttr().Set(1e8)
                drive.GetDampingAttr().Set(1e6)
                drive.GetStiffnessAttr().Set(0.0)

    def hold_done_state():
        nonlocal hold_frame_count
        freeze_base_and_wheels(hard_pose_lock=False)
        if (
            hold_gripper_width is not None
            and hold_gripper_refresh_frames > 0
            and hold_frame_count % hold_gripper_refresh_frames == 0
        ):
            if hasattr(gripper, "hold"):
                gripper.hold(hold_gripper_width)
            else:
                gripper.close(hold_gripper_width)
        if hold_arm_joint_positions is not None and len(joint_indices) == 6:
            try:
                if hold_frame_count % hold_arm_refresh_frames == 0:
                    if ArticulationAction is not None:
                        action = ArticulationAction(joint_positions=hold_arm_joint_positions, joint_indices=joint_indices)
                        my_robot.apply_action(action)
                    else:
                        my_robot.set_joint_positions(hold_arm_joint_positions, joint_indices=joint_indices)
                if (
                    hold_arm_zero_vel_frames > 0
                    and hold_frame_count < hold_arm_zero_vel_warmup
                    and hold_frame_count % hold_arm_zero_vel_frames == 0
                ):
                    my_robot.set_joint_velocities(np.zeros(len(joint_indices)), joint_indices=joint_indices)
            except Exception as e:
                if hold_frame_count % 120 == 0:
                    my_print(f"⚠️ [Grasp Hold] 机械臂保持命令失败: {e}")
        hold_frame_count += 1
        if hold_frame_count % 240 == 1 and hold_gripper_width is not None:
            my_print(
                f"🧷 [Grasp Hold] 保持抓取中: gripper_width={hold_gripper_width:.3f}, "
                f"arm_hold={'gentle' if hold_arm_joint_positions is not None else 'off'}, "
                f"gripper_refresh={hold_gripper_refresh_frames if hold_gripper_refresh_frames > 0 else 'off'}"
            )

    def discover_gripper_control_paths():
        fingertip_paths = []
        tool_paths = []
        for prim in stage.Traverse():
            path = prim.GetPath().pathString
            low_path = path.lower()
            if "/robots/husky/jackal" not in low_path:
                continue
            name = prim.GetName().lower()
            if ("fingertip" in name or "fingertip" in low_path) and "material" not in low_path:
                fingertip_paths.append(path)
            elif low_path.endswith("/tool0"):
                tool_paths.append(path)
        if len(fingertip_paths) >= 2:
            selected = sorted(fingertip_paths)[:2]
            my_print(f"✅ [WorldAlign] 使用双指尖世界坐标做末端校正: {selected}")
            return selected, "fingertip_center"
        if tool_paths:
            selected = sorted(tool_paths)[:1]
            my_print(f"⚠️ [WorldAlign] 未找到双指尖，使用 tool0 世界坐标做末端校正: {selected}")
            return selected, "tool0"
        my_print("⚠️ [WorldAlign] 未找到指尖/tool0，关闭世界坐标辅助校正，避免 wrist_3_link 偏置导致抓偏。")
        return [], "disabled"

    gripper_control_paths, gripper_control_source = discover_gripper_control_paths()

    def get_current_gripper_control_center():
        points = []
        for path in gripper_control_paths:
            try:
                pos, _ = get_world_pose(path)
                points.append(np.array(pos, dtype=np.float64))
            except Exception:
                pass
        if points:
            return np.mean(np.stack(points, axis=0), axis=0)
        try:
            ee_pos, _ = ee_solver.compute_end_effector_pose()
            return np.array(ee_pos, dtype=np.float64)
        except Exception:
            return None

    wrist3_link_paths = [
        "/Robots/husky/jackal/ur3e/wrist_3_link",
        "/Robots/husky/jackal/ur10e/wrist_3_link",
    ]

    def get_wrist3_link_world_position():
        for path in wrist3_link_paths:
            try:
                if stage.GetPrimAtPath(path).IsValid():
                    pos, _ = get_world_pose(path)
                    return np.array(pos, dtype=np.float64), path
            except Exception:
                pass
        try:
            ee_pos, _ = ee_solver.compute_end_effector_pose()
            return np.array(ee_pos, dtype=np.float64), "ee_solver_pose"
        except Exception:
            return None, "unavailable"

    def get_wrist_camera_world_pose():
        if not wrist_cam_prim_path:
            raise RuntimeError("no wrist camera prim path")
        cam_pos, cam_quat = get_world_pose(wrist_cam_prim_path)
        return np.array(cam_pos, dtype=np.float64), quat_wxyz_normalize(cam_quat)

    if USE_ANYGRASP:
        grasp_executor = AnyGraspWristGraspExecutor(
            my_robot,
            ee_solver,
            gripper,
            wrist_cam,
            get_camera_pose_fn=get_wrist_camera_world_pose,
            get_tool_center_fn=get_current_gripper_control_center,
            detector=wrist_probe_executor,
        )
        my_print("✅ [AnyGrasp] demo7 默认使用 AnyGrasp RGB-D 抓取；设置 OMNILRS_USE_ANYGRASP=0 可回退旧 WristVision。")
    else:
        my_print("⚠️ [AnyGrasp] 已按 OMNILRS_USE_ANYGRASP=0 回退旧 WristVision 视觉伺服抓取。")

    def get_debug_world_alignment():
        if not (GRASP_DEBUG_MODE and WORLD_ALIGN_DEBUG):
            return None

        rock_pos = None
        if test_rock is not None:
            try:
                rock_pos, _ = test_rock.get_world_pose()
            except Exception:
                rock_pos = None
        if rock_pos is None:
            try:
                rock_pos, _ = get_world_pose(rock_prim_path)
            except Exception:
                return None
        rock_pos = np.array(rock_pos, dtype=np.float64) + WORLD_ALIGN_OFFSET

        points = []
        for path in gripper_control_paths:
            try:
                pos, _ = get_world_pose(path)
                points.append(np.array(pos, dtype=np.float64))
            except Exception:
                pass
        if not points:
            return None
        tool_pos = np.mean(np.stack(points, axis=0), axis=0)
        source = gripper_control_source

        delta = np.array([rock_pos[0] - tool_pos[0], rock_pos[1] - tool_pos[1], 0.0], dtype=np.float64)
        return {
            "delta": delta,
            "err_norm": float(np.linalg.norm(delta[:2])),
            "rock": rock_pos,
            "tool": tool_pos,
            "source": source,
        }

    if GRASP_DEBUG_MODE and WORLD_ALIGN_DEBUG and gripper_control_paths:
        grasp_executor.set_world_alignment_callback(get_debug_world_alignment)
        my_print("🧭 [WorldAlign] 调试模式启用夹爪/石头世界坐标末端辅助对齐。")

    def recovery_state_after_grasp_failure():
        return "GRASP_DEBUG_INIT" if GRASP_DEBUG_MODE else "TRACKING"

    def capture_grasp_start_pose():
        nonlocal grasp_start_joint_positions
        current = my_robot.get_joint_positions()
        if current is not None:
            grasp_start_joint_positions = np.array(current, dtype=np.float64)
            my_print("💾 [Grasp Recovery] 已记录抓取前机械臂关节姿态，失败时会先回位再重试。")

    def begin_arm_restore_for_retry(next_state, reason):
        nonlocal ROBOT_STATE, restore_target_joint_positions, restore_next_state, restore_frame_count
        if grasp_start_joint_positions is None or len(joint_indices) != 6:
            my_print(f"⚠️ [Grasp Recovery] 无抓取前姿态记录，无法平滑回位；直接切换到 {next_state}。")
            grasp_executor.state = "IDLE"
            release_grasp_base_lock()
            ROBOT_STATE = next_state
            return
        restore_target_joint_positions = np.array(grasp_start_joint_positions, dtype=np.float64)
        restore_next_state = next_state
        restore_frame_count = 0
        grasp_executor.state = "IDLE"
        my_print(f"↩️ [Grasp Recovery] {reason}，先打开夹爪并恢复机械臂到抓取前姿态。")
        ROBOT_STATE = "RESTORE_ARM"

    def latch_grasp_hold(reason, keep_gripper=True):
        nonlocal ROBOT_STATE, locked_wheel_pos, hold_arm_joint_positions, hold_gripper_width, hold_frame_count
        locked_wheel_pos = my_robot.get_joint_positions()
        hold_frame_count = 0
        if hold_arm_after_lift and locked_wheel_pos is not None and len(joint_indices) == 6:
            current = np.array(locked_wheel_pos, dtype=np.float64)
            hold_arm_joint_positions = current[joint_indices].copy()
            try:
                if ArticulationAction is not None:
                    action = ArticulationAction(joint_positions=hold_arm_joint_positions, joint_indices=joint_indices)
                    my_robot.apply_action(action)
                my_robot.set_joint_velocities(np.zeros(len(joint_indices)), joint_indices=joint_indices)
            except Exception as e:
                my_print(f"⚠️ [Grasp Hold] 初始机械臂稳定命令失败: {e}")
        else:
            hold_arm_joint_positions = None
        target_width = None
        if keep_gripper:
            target_width = getattr(grasp_executor, "hold_target_width", None)
            if target_width is None:
                target_width = getattr(grasp_executor, "close_target_width", None)
        hold_gripper_width = None if target_width is None else float(np.clip(target_width, 0.025, 0.14))
        if hold_gripper_width is not None:
            if hasattr(gripper, "hold"):
                gripper.hold(hold_gripper_width)
            else:
                gripper.close(hold_gripper_width)
        hold_width_msg = f"{hold_gripper_width:.3f}" if hold_gripper_width is not None else "unchanged"
        my_print(
            f"🧷 [Grasp Hold] {reason}；进入抓取保持状态，不再打开夹爪/回位重试。"
            f"hold_width={hold_width_msg}, "
            f"arm_hold={'gentle' if hold_arm_joint_positions is not None else 'off'}, "
            f"gripper_refresh={hold_gripper_refresh_frames if hold_gripper_refresh_frames > 0 else 'off'}"
        )
        ROBOT_STATE = "DONE"

    def step_restore_arm_state():
        nonlocal ROBOT_STATE, restore_target_joint_positions, restore_next_state, restore_frame_count
        nonlocal debug_init_attempts
        freeze_base_and_wheels(hard_pose_lock=False)
        gripper.open()

        current = my_robot.get_joint_positions()
        if current is None or restore_target_joint_positions is None:
            release_grasp_base_lock()
            ROBOT_STATE = restore_next_state or recovery_state_after_grasp_failure()
            return

        current = np.array(current, dtype=np.float64)
        target = np.array(restore_target_joint_positions, dtype=np.float64)
        next_positions = current.copy()
        arm_error = target[joint_indices] - current[joint_indices]
        max_joint_step = 0.040
        next_positions[joint_indices] = current[joint_indices] + np.clip(arm_error, -max_joint_step, max_joint_step)

        if ArticulationAction is not None:
            action = ArticulationAction(joint_positions=next_positions[joint_indices], joint_indices=joint_indices)
            my_robot.apply_action(action)
        else:
            my_robot.set_joint_positions(next_positions[joint_indices], joint_indices=joint_indices)

        restore_frame_count += 1
        err_norm = float(np.linalg.norm(arm_error))
        if restore_frame_count % 12 == 1:
            my_print(f"↩️ [Grasp Recovery] 机械臂回位中: joint_err={err_norm:.3f}, frame={restore_frame_count}")

        if err_norm < 0.045 or restore_frame_count > 95:
            my_print("✅ [Grasp Recovery] 机械臂已回到抓取前姿态，准备重新尝试。")
            release_grasp_base_lock()
            debug_init_attempts = 0
            ROBOT_STATE = restore_next_state or recovery_state_after_grasp_failure()

    def is_reasonable_mast_debug_box(rgb, box):
        h, w = rgb.shape[:2]
        x0, y0, x1, y1 = [float(v) for v in box]
        bw, bh = max(0.0, x1 - x0), max(0.0, y1 - y0)
        area_ratio = (bw * bh) / max(1.0, float(w * h))
        if area_ratio < 0.00008:
            return False
        if area_ratio > 0.16 or bw / max(1.0, w) > 0.45 or bh / max(1.0, h) > 0.45:
            return False

        ix0, iy0 = int(max(0, x0)), int(max(0, y0))
        ix1, iy1 = int(min(w, x1)), int(min(h, y1))
        crop = rgb[iy0:iy1, ix0:ix1].astype(np.float32)
        if crop.size == 0:
            return False
        r, g, b = crop[..., 0], crop[..., 1], crop[..., 2]
        yellow_ratio = float(np.mean((r > 150) & (g > 130) & (b < 95)))
        white_robot_ratio = float(np.mean((r > 190) & (g > 190) & (b > 180)))
        if yellow_ratio > 0.08 or white_robot_ratio > 0.48:
            return False
        return True

    def choose_grasp_debug_mast_box(rgb, boxes):
        h, w = rgb.shape[:2]
        expected_u = w * 0.52
        expected_v = h * 0.78
        best_box, best_score = None, -1e18
        for box in boxes:
            if not is_reasonable_mast_debug_box(rgb, box):
                continue
            x0, y0, x1, y1 = [float(v) for v in box]
            bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
            area = bw * bh
            cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            center_penalty = 1.1 * abs(cx - expected_u) + 0.75 * abs(cy - expected_v)
            size_bonus = min(area, 16000.0) * 0.012
            bottom_bonus = 60.0 * (cy / max(1.0, h))
            score = size_bonus + bottom_bonus - center_penalty
            if score > best_score:
                best_score = score
                best_box = [x0, y0, x1, y1]
        return best_box

    def save_debug_mast_box(rgb, box, path="debug_grasp_debug_mast.jpg"):
        try:
            debug_img = Image.fromarray(rgb.astype(np.uint8))
            draw = ImageDraw.Draw(debug_img)
            x0, y0, x1, y1 = [float(v) for v in box]
            draw.rectangle([x0, y0, x1, y1], outline="red", width=6)
            debug_img.save(path)
        except Exception as e:
            my_print(f"⚠️ [GraspDebug] 高杆调试图保存失败: {e}")

    def save_debug_wrist_box(rgb, box, path="debug_wrist_start_lock.jpg"):
        try:
            debug_img = Image.fromarray(rgb.astype(np.uint8))
            draw = ImageDraw.Draw(debug_img)
            x0, y0, x1, y1 = [float(v) for v in box]
            draw.rectangle([x0, y0, x1, y1], outline="lime", width=6)
            debug_img.save(path)
        except Exception as e:
            my_print(f"⚠️ [GraspDebug] 手腕调试图保存失败: {e}")

    def rotate_debug_base_after_startup():
        nonlocal ROBOT_STATE, debug_base_rotated, debug_base_yaw_settle_frames, debug_init_attempts

        if debug_base_rotated:
            return True

        debug_init_attempts = 0
        if abs(DEBUG_BASE_YAW_DEG) < 1e-6:
            debug_base_rotated = True
            debug_base_yaw_settle_frames = 0
            my_print("🧪 [GraspDebug] 调试底盘右转角为0，跳过转向，直接准备放置月岩。")
            return True

        try:
            freeze_base_and_wheels(hard_pose_lock=False)
            base_pos, base_ori = my_robot.get_world_pose()
            base_pos = np.array(base_pos, dtype=np.float64)
            base_ori = quat_wxyz_normalize(base_ori)
            yaw_delta = quat_wxyz_from_yaw(math.radians(DEBUG_BASE_YAW_DEG))
            target_ori = quat_wxyz_normalize(quat_wxyz_multiply(yaw_delta, base_ori))
            my_robot.set_linear_velocity(np.zeros(3))
            my_robot.set_angular_velocity(np.zeros(3))
            my_robot.set_world_pose(position=base_pos, orientation=target_ori)
            my_robot.set_linear_velocity(np.zeros(3))
            my_robot.set_angular_velocity(np.zeros(3))
            debug_base_rotated = True
            debug_base_yaw_settle_frames = max(0, DEBUG_BASE_YAW_SETTLE_FRAMES)
            my_print(
                f"🧪 [GraspDebug] demo2初始化与稳定等待完成后，原地调整车身朝向: yaw_delta={DEBUG_BASE_YAW_DEG:+.1f}deg "
                f"(默认负值表示右转)，settle_frames={debug_base_yaw_settle_frames}。"
            )
            return True
        except Exception as e:
            my_print(f"❌ [GraspDebug] 调试底盘转向失败: {e}")
            ROBOT_STATE = "DONE"
            return False

    def normalize_xy(vec, fallback=None):
        vec = np.array(vec, dtype=np.float64)
        vec[2] = 0.0
        norm = float(np.linalg.norm(vec[:2]))
        if norm < 1e-6:
            return np.array(fallback if fallback is not None else [1.0, 0.0, 0.0], dtype=np.float64)
        return vec / norm

    def debug_rock_basis_after_startup():
        base_pos, base_ori = my_robot.get_world_pose()
        base_pos = np.array(base_pos, dtype=np.float64)
        base_ori = quat_wxyz_normalize(base_ori)

        axis_name = os.environ.get("OMNILRS_DEBUG_ROCK_AXIS", "+x").strip().lower()
        axis_map = {
            "+x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
            "x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
            "-x": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
            "+y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
            "y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
            "-y": np.array([0.0, -1.0, 0.0], dtype=np.float64),
        }
        base_forward = normalize_xy(quat_wxyz_apply(base_ori, axis_map.get(axis_name, axis_map["+x"])))
        base_right = normalize_xy(np.array([base_forward[1], -base_forward[0], 0.0], dtype=np.float64))
        if DEBUG_ROCK_FRAME in ["base", "robot", "base_link"]:
            candidate_msgs = []
            for name in ["+x", "-x", "+y", "-y"]:
                direction = normalize_xy(quat_wxyz_apply(base_ori, axis_map[name]))
                candidate = base_pos + direction * DEBUG_ROCK_FORWARD
                candidate[2] = DEBUG_ROCK_Z
                candidate_msgs.append(f"{name}=({candidate[0]:.3f},{candidate[1]:.3f},{candidate[2]:.3f})")
            my_print(
                f"🧭 [GraspDebug] base_link 四轴候选车前位置: {' | '.join(candidate_msgs)}；"
                f"当前使用 OMNILRS_DEBUG_ROCK_AXIS={axis_name}"
            )

        if DEBUG_ROCK_FRAME in ["mast", "camera", "cam"]:
            try:
                cam_pos, cam_quat = get_world_pose(real_mast_cam_prim_path)
                cam_pos = np.array(cam_pos, dtype=np.float64)
                cam_quat = quat_wxyz_normalize(cam_quat)
                cam_forward = optical_camera_to_world(cam_pos, cam_quat, np.array([0.0, 0.0, 1.0], dtype=np.float64)) - cam_pos
                cam_right = optical_camera_to_world(cam_pos, cam_quat, np.array([1.0, 0.0, 0.0], dtype=np.float64)) - cam_pos
                forward = normalize_xy(cam_forward, fallback=base_forward)
                right = normalize_xy(cam_right, fallback=base_right)
                if abs(float(np.dot(forward[:2], right[:2]))) > 0.35:
                    right = normalize_xy(np.array([forward[1], -forward[0], 0.0], dtype=np.float64))
                return base_pos, forward, right, "mast_camera_projected"
            except Exception as e:
                my_print(f"⚠️ [GraspDebug] 无法使用高杆相机朝向计算车前方向，退回 base 轴: {e}")

        return base_pos, base_forward, base_right, f"base_axis_{axis_name}"

    def read_rock_world_pos():
        if test_rock is not None:
            try:
                pos, _ = test_rock.get_world_pose()
                return np.array(pos, dtype=np.float64)
            except Exception:
                pass
        try:
            pos, _ = get_world_pose(rock_prim_path)
            return np.array(pos, dtype=np.float64)
        except Exception:
            return None

    def read_anygrasp_target_world():
        # Always use the true rock pose from the stage (physics-settled).
        # Mast DINO depth estimates are unreliable for 3D projection.
        rock = read_rock_world_pos()
        if rock is not None:
            return rock
        if anygrasp_target_world is not None:
            return np.array(anygrasp_target_world, dtype=np.float64)
        return None

    if USE_ANYGRASP and hasattr(grasp_executor, "set_known_target_world_callback"):
        grasp_executor.set_known_target_world_callback(read_anygrasp_target_world)
        my_print("🎯 [AnyGrasp] 调试模式已接入月岩世界坐标投影ROI兜底，避免DINO离线时抓地/抓车体。")

    def project_world_to_mast_pixel(world_pos):
        try:
            cam_pos, cam_quat = get_world_pose(real_mast_cam_prim_path)
            target_cam = world_to_optical_camera(cam_pos, quat_wxyz_normalize(cam_quat), world_pos)
            if target_cam[2] <= 0.03:
                return None, target_cam
            u = dino_tracker.cx + target_cam[0] * dino_tracker.fx / target_cam[2]
            v = dino_tracker.cy + target_cam[1] * dino_tracker.fy / target_cam[2]
            return np.array([u, v], dtype=np.float64), target_cam
        except Exception:
            return None, None

    def mast_box_to_world(rgb_img, box):
        x_c, y_c, z_c = dino_tracker._get_3d_coordinates(rgb_img, box)
        if z_c <= 0.03:
            return None
        cam_pos, cam_quat = get_world_pose(real_mast_cam_prim_path)
        return optical_camera_to_world(
            np.array(cam_pos, dtype=np.float64),
            quat_wxyz_normalize(cam_quat),
            np.array([x_c, y_c, z_c], dtype=np.float64),
        )

    def projected_mast_box_from_world(rgb_img, target_world, side_px=34.0):
        uv, _ = project_world_to_mast_pixel(target_world)
        if uv is None:
            return None
        h, w = rgb_img.shape[:2]
        u, v = float(uv[0]), float(uv[1])
        if not (-side_px <= u <= w + side_px and -side_px <= v <= h + side_px):
            return None
        return [
            float(max(0.0, u - side_px)),
            float(max(0.0, v - side_px)),
            float(min(w - 1.0, u + side_px)),
            float(min(h - 1.0, v + side_px)),
        ]

    def select_mast_guided_target(rgb_img, boxes):
        debug_rock = read_rock_world_pos() if GRASP_DEBUG_MODE else None
        max_debug_err = float(os.environ.get("OMNILRS_MAST_DEBUG_MAX_ERR", "0.35"))
        h, w = rgb_img.shape[:2]
        sector = os.environ.get("OMNILRS_TARGET_SECTOR", "front").strip().lower().replace("-", "_")
        default_u_ratio = 0.64 if "right" in sector else (0.36 if "left" in sector else 0.52)
        default_v_ratio = 0.74 if ("front" in sector or "forward" in sector) else 0.62
        expected_u = float(os.environ.get("OMNILRS_MAST_TARGET_U_RATIO", str(default_u_ratio))) * w
        expected_v = float(os.environ.get("OMNILRS_MAST_TARGET_V_RATIO", str(default_v_ratio))) * h
        best = None
        best_score = -1e18
        for box in boxes:
            if not is_reasonable_mast_debug_box(rgb_img, box):
                continue
            world = mast_box_to_world(rgb_img, box)
            if world is None:
                continue
            if debug_rock is not None:
                err = float(np.linalg.norm((world - debug_rock)[:2]))
                if err > max_debug_err:
                    my_print(
                        f"🚫 [MastGuide] 拒绝高杆候选: 与调试月岩相差 {err*1000:.1f}mm > "
                        f"{max_debug_err*1000:.0f}mm, box={[round(float(v),1) for v in box]}"
                    )
                    continue
            x0, y0, x1, y1 = [float(v) for v in box]
            bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
            cx_box, cy_box = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            area = bw * bh
            score = min(area, 12000.0) * 0.010 + 0.20 * cy_box
            score -= 0.85 * abs(cx_box - expected_u) + 0.55 * abs(cy_box - expected_v)
            if debug_rock is not None:
                score -= 900.0 * err
            if score > best_score:
                best_score = score
                best = (box, world, score)

        if best is not None:
            box, world, score = best
            debug_msg = ""
            if debug_rock is not None:
                debug_msg = f", err_to_debug_rock={np.linalg.norm((world-debug_rock)[:2])*1000:.1f}mm"
            my_print(
                f"🎯 [MastGuide] 高杆选中候选: box={[round(float(v),1) for v in box]}, "
                f"world=({world[0]:.3f},{world[1]:.3f},{world[2]:.3f}), score={score:.1f}{debug_msg}"
            )
            return box, world, "mast_dino_depth"

        if debug_rock is not None:
            fallback_box = projected_mast_box_from_world(rgb_img, debug_rock)
            if fallback_box is not None:
                my_print(
                    f"🎯 [MastGuide] DINO未给出可信近处石头，使用调试月岩高杆投影: "
                    f"box={[round(v,1) for v in fallback_box]}, world=({debug_rock[0]:.3f},{debug_rock[1]:.3f},{debug_rock[2]:.3f})"
                )
                return fallback_box, np.array(debug_rock, dtype=np.float64), "mast_debug_projection"
        return None, None, "none"

    def force_move_rock_world(target_pos):
        target_pos = np.array(target_pos, dtype=np.float64)
        moved = False
        if test_rock is not None:
            try:
                test_rock.set_world_pose(position=target_pos)
                moved = True
            except TypeError:
                try:
                    test_rock.set_world_pose(target_pos)
                    moved = True
                except Exception:
                    moved = False
            except Exception:
                moved = False

            for method_name in ["set_linear_velocity", "set_angular_velocity"]:
                try:
                    getattr(test_rock, method_name)(np.zeros(3))
                except Exception:
                    pass

        rock_prim = stage.GetPrimAtPath(rock_prim_path)
        if rock_prim.IsValid():
            try:
                set_local_translate(rock_prim, target_pos)
                moved = True
            except Exception as e:
                my_print(f"❌ [GraspDebug] 无法写入月岩 root xform: {e}")
        return moved

    def relocate_debug_rock_after_startup():
        nonlocal ROBOT_STATE, debug_rock_relocated, debug_rock_settle_frames, debug_init_attempts

        if debug_rock_relocated:
            return True

        before_pos = read_rock_world_pos()
        base_pos, forward_dir, right_dir, frame_name = debug_rock_basis_after_startup()
        debug_target = base_pos + forward_dir * DEBUG_ROCK_FORWARD + right_dir * DEBUG_ROCK_LATERAL
        debug_target[2] = DEBUG_ROCK_Z

        moved = force_move_rock_world(debug_target)
        after_pos = read_rock_world_pos()
        readback_err = float(np.linalg.norm(after_pos - debug_target)) if after_pos is not None else float("inf")

        if moved:
            debug_rock_relocated = True
            debug_rock_settle_frames = max(0, DEBUG_ROCK_SETTLE_FRAMES)
            debug_init_attempts = 0
            dino_tracker.last_u, dino_tracker.last_v = None, None
            before_msg = "unknown" if before_pos is None else f"({before_pos[0]:.3f},{before_pos[1]:.3f},{before_pos[2]:.3f})"
            after_msg = "unknown" if after_pos is None else f"({after_pos[0]:.3f},{after_pos[1]:.3f},{after_pos[2]:.3f})"
            my_print(
                f"🧪 [GraspDebug] 车辆完成 demo2/demo5 同款初始化后，强制移动月岩到车前: "
                f"frame={frame_name}, forward={DEBUG_ROCK_FORWARD:.2f}, lateral={DEBUG_ROCK_LATERAL:.2f}, "
                f"z={DEBUG_ROCK_Z:.2f}, settle_frames={debug_rock_settle_frames}"
            )
            my_print(
                f"🧪 [GraspDebug] rock before={before_msg}, target=({debug_target[0]:.3f},{debug_target[1]:.3f},{debug_target[2]:.3f}), "
                f"after={after_msg}, readback_err={readback_err:.4f}m"
            )
            if readback_err > 0.03:
                my_print("⚠️ [GraspDebug] 月岩移动读回误差较大，可能是 USD 引用根节点/刚体 root 不一致。")
            return True

        my_print("❌ [GraspDebug] 月岩 prim 不可用，无法进入抓取调试。")
        ROBOT_STATE = "DONE"
        return False

    def step_grasp_debug_init():
        nonlocal ROBOT_STATE, locked_roi_box, latest_rgb_img, locked_target_prompt
        nonlocal locked_base_pos, locked_base_ori, locked_wheel_pos, debug_init_attempts
        nonlocal anygrasp_target_world, anygrasp_target_source

        freeze_base_and_wheels(hard_pose_lock=False)
        debug_prompts = [
            "the nearby small rock between the gripper fingers.",
            "the small rock in the wrist camera.",
            "the small gray rock.",
            "the small rock in front of the robot.",
        ]

        if DEBUG_WRIST_FIRST and not USE_ANYGRASP:
            wrist_errors = []
            for prompt in debug_prompts:
                try:
                    wrist_box, wrist_rgb = grasp_executor.detect_wrist_target_once(prompt)
                    if wrist_box is None or wrist_rgb is None:
                        continue
                    locked_target_prompt = prompt
                    locked_roi_box = [float(v) for v in wrist_box]
                    save_debug_wrist_box(wrist_rgb, locked_roi_box)

                    rgba_data = mast_cam.get_rgba()
                    if rgba_data is not None and rgba_data.size > 0:
                        latest_rgb_img = rgba_data[:, :, :3].astype(np.uint8)

                    locked_base_pos, locked_base_ori = my_robot.get_world_pose()
                    locked_wheel_pos = my_robot.get_joint_positions()
                    create_grasp_base_lock()
                    dino_tracker.last_u, dino_tracker.last_v = None, None
                    capture_grasp_start_pose()
                    grasp_executor.start_wrist_first(
                        locked_target_prompt,
                        grasp_width=0.055,
                        wrist_rgb=wrist_rgb,
                        wrist_box=locked_roi_box,
                    )
                    x0, y0, x1, y1 = locked_roi_box
                    target_pixel = np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)], dtype=np.float64)
                    my_print("\n" + "█"*55)
                    my_print("🧪 [GraspDebug] 手腕相机已直接看到近处目标，高杆ROI不再参与选目标。")
                    my_print(
                        f"🧪 [GraspDebug] wrist_roi={[round(v, 1) for v in locked_roi_box]}, "
                        f"target_pixel=({target_pixel[0]:.1f},{target_pixel[1]:.1f}), prompt={locked_target_prompt}"
                    )
                    my_print("█"*55 + "\n")
                    ROBOT_STATE = "ARM_LOCKING"
                    return
                except Exception as e:
                    wrist_errors.append(str(e))

            debug_init_attempts += 1
            if debug_init_attempts % 4 == 1:
                err_msg = f": {wrist_errors[-1]}" if wrist_errors else ""
                my_print(f"⚠️ [GraspDebug] 手腕优先暂未锁定近处石头，才允许退回高杆粗ROI{err_msg}")

        rgba_data = mast_cam.get_rgba()
        if rgba_data is None or rgba_data.size == 0:
            debug_init_attempts += 1
            if debug_init_attempts % 4 == 1:
                my_print("⚠️ [GraspDebug] 高杆相机暂时没有图像，保持刹车等待。")
            return

        rgb_img = rgba_data[:, :, :3].astype(np.uint8)
        latest_rgb_img = rgb_img
        img_h, img_w = rgb_img.shape[:2]
        debug_init_attempts += 1

        selected_box = None
        selected_prompt = debug_prompts[0]
        selected_world = None
        selected_source = "none"
        dino_errors = []

        for prompt in debug_prompts:
            try:
                payload = {
                    "image_base64": dino_tracker._encode_image(rgb_img),
                    "text_prompt": prompt,
                }
                res = requests.post(dino_tracker.server_url, json=payload, timeout=5).json()
                boxes = res.get("boxes", []) if res.get("found") else []
                if boxes:
                    candidate, candidate_world, candidate_source = select_mast_guided_target(rgb_img, boxes)
                    if candidate is not None:
                        selected_box = candidate
                        selected_world = candidate_world
                        selected_source = candidate_source
                        selected_prompt = prompt
                        break
            except Exception as e:
                dino_errors.append(str(e))

        if selected_box is None:
            projected_box, projected_world, projected_source = select_mast_guided_target(rgb_img, [])
            if projected_box is not None:
                selected_box = projected_box
                selected_world = projected_world
                selected_source = projected_source
                selected_prompt = debug_prompts[0]

        if selected_box is None:
            if debug_init_attempts < 4:
                if dino_errors:
                    my_print(f"⚠️ [GraspDebug] DINO暂未锁定车前岩石: {dino_errors[-1]}")
                else:
                    my_print("⚠️ [GraspDebug] DINO暂未锁定车前岩石，继续等待下一帧。")
                return
            selected_box = [
                img_w * float(os.environ.get("OMNILRS_MAST_FALLBACK_X0", "0.56")),
                img_h * float(os.environ.get("OMNILRS_MAST_FALLBACK_Y0", "0.56")),
                img_w * float(os.environ.get("OMNILRS_MAST_FALLBACK_X1", "0.78")),
                img_h * float(os.environ.get("OMNILRS_MAST_FALLBACK_Y1", "0.88")),
            ]
            selected_prompt = debug_prompts[0]
            my_print(
                "⚠️ [GraspDebug] DINO连续未锁定，使用目标扇区先验小ROI兜底；"
                "请查看 debug_grasp_debug_mast.jpg 确认红框是否框住石头。"
            )
            selected_world = read_rock_world_pos() if GRASP_DEBUG_MODE else None
            selected_source = "fallback_mast_roi"

        locked_target_prompt = selected_prompt
        locked_roi_box = [float(v) for v in selected_box]
        save_debug_mast_box(rgb_img, locked_roi_box)
        if selected_world is not None:
            anygrasp_target_world = np.array(selected_world, dtype=np.float64)
            anygrasp_target_source = selected_source
            my_print(
                f"🎯 [MastGuide] 已把高杆目标交给手腕前探: source={anygrasp_target_source}, "
                f"target_world=({anygrasp_target_world[0]:.3f},{anygrasp_target_world[1]:.3f},{anygrasp_target_world[2]:.3f})"
            )
        else:
            anygrasp_target_world = None
            anygrasp_target_source = "unset"
            my_print("⚠️ [MastGuide] 没有可靠高杆目标世界点，AnyGrasp只能使用调试rock pose或拒绝全图抓取。")

        locked_base_pos, locked_base_ori = my_robot.get_world_pose()
        locked_wheel_pos = my_robot.get_joint_positions()
        create_grasp_base_lock()

        x0, y0, x1, y1 = locked_roi_box
        target_pixel = np.array([0.5 * (x0 + x1), y0 + 0.45 * (y1 - y0)], dtype=np.float64)
        dino_tracker.last_u, dino_tracker.last_v = None, None
        capture_grasp_start_pose()
        grasp_executor.start(
            target_pixel,
            locked_roi_box,
            locked_target_prompt,
            grasp_width=0.055,
            mast_rgb=latest_rgb_img,
        )
        my_print("\n" + "█"*55)
        my_print("🧪 [GraspDebug] 已跳过导航，直接启动手腕视觉抓取调试。")
        my_print(
            f"🧪 [GraspDebug] mast_roi={[round(v, 1) for v in locked_roi_box]}, "
            f"target_pixel=({target_pixel[0]:.1f},{target_pixel[1]:.1f}), prompt={locked_target_prompt}"
        )
        my_print("█"*55 + "\n")
        ROBOT_STATE = "ARM_LOCKING"

    def step_active_grasp_state():
        nonlocal ROBOT_STATE, locked_wheel_pos, grasp_retry_count
        freeze_base_and_wheels(hard_pose_lock=False)
        grasp_state = grasp_executor.step()
        if grasp_state in ["OPEN", "WRIST_SEARCH", "WRIST_SERVO", "WRIST_ALIGN", "ANYGRASP_PLAN", "PRE_GRASP"]:
            ROBOT_STATE = "ARM_LOCKING"
        elif grasp_state in ["NEAR_DESCEND", "VISUAL_APPROACH", "LOCAL_APPROACH", "LOCAL_DESCEND", "DESCEND", "CLOSE", "LIFT"]:
            ROBOT_STATE = "GRASPING"
        elif grasp_state == "DONE":
            if hold_grasp_after_lift:
                my_print("✅ [Grasp] 抓取规划、夹取、抬升动作完成。")
                latch_grasp_hold("抓取已抬升完成")
            else:
                my_print("✅ [Grasp] 抓取规划、夹取、抬升动作完成，进入 VLM 验收。")
                locked_wheel_pos = my_robot.get_joint_positions()
                ROBOT_STATE = "VERIFY_GRASP"
        elif grasp_state == "FAILED":
            grasp_retry_count += 1
            locked_wheel_pos = my_robot.get_joint_positions()
            my_print(f"❌ [Grasp] 抓取规划/IK/执行失败，retry={grasp_retry_count}")
            if not retry_grasp_on_fail:
                latch_grasp_hold("抓取未成功但已按配置停止重试", keep_gripper=True)
            elif grasp_retry_count <= 2:
                if GRASP_DEBUG_MODE:
                    next_state = recovery_state_after_grasp_failure()
                    begin_arm_restore_for_retry(next_state, "调试模式抓取失败")
                else:
                    begin_arm_restore_for_retry(recovery_state_after_grasp_failure(), "抓取失败")
            else:
                my_print("🛑 [Grasp Recovery] 多次失败，进入保持状态，保留日志供调参。")
                ROBOT_STATE = "DONE"

    def step_verify_grasp_state():
        nonlocal ROBOT_STATE, locked_wheel_pos, grasp_retry_count
        freeze_base_and_wheels()
        rgba = mast_cam.get_rgba()
        if rgba is None or rgba.size == 0:
            my_print("⚠️ [VLM 验收] 高杆相机无图像，按执行完成处理。")
            ROBOT_STATE = "DONE"
            return
        rgb = rgba[:, :, :3].astype(np.uint8)
        if not getattr(grasp_executor, "closed_on_valid_target", False):
            my_print("❌ [VLM 验收] 抓取器没有在通过复核的手腕目标上闭合，拒绝宣布成功。")
            grasp_retry_count += 1
            if grasp_retry_count <= 2:
                begin_arm_restore_for_retry(recovery_state_after_grasp_failure(), "验收发现未在有效目标上闭合")
            else:
                ROBOT_STATE = "DONE"
            return
        verify_rgb = rgb
        if wrist_cam is not None:
            wrist_rgba = wrist_cam.get_rgba()
            if wrist_rgba is not None and wrist_rgba.size > 0:
                verify_rgb = wrist_rgba[:, :, :3].astype(np.uint8)
                try:
                    Image.fromarray(verify_rgb).save("debug_wrist_verify.jpg")
                except Exception:
                    pass
                my_print("🔎 [VLM 验收] 使用手腕相机近景图进行抓取验收。")
        ok, failure_mode = vlm_brain.verify_grasp_success(verify_rgb, locked_target_prompt)
        locked_wheel_pos = my_robot.get_joint_positions()
        if ok:
            my_print("🏁 [Mission] 月岩样本采集闭环完成。")
            ROBOT_STATE = "DONE"
        else:
            grasp_retry_count += 1
            if grasp_retry_count <= 2:
                if GRASP_DEBUG_MODE:
                    begin_arm_restore_for_retry(recovery_state_after_grasp_failure(), f"验收失败({failure_mode})")
                else:
                    begin_arm_restore_for_retry(recovery_state_after_grasp_failure(), f"验收失败({failure_mode})")
            else:
                my_print(f"🛑 [Grasp Recovery] 验收失败({failure_mode})且重试耗尽，进入保持状态。")
                ROBOT_STATE = "DONE"

    ik_calib_state = {
        "phase": "INIT",
        "frame": 0,
        "axis_index": 0,
        "results": [],
        "move_frames": int(os.environ.get("OMNILRS_IK_CALIB_MOVE_FRAMES", "45")),
        "return_frames": int(os.environ.get("OMNILRS_IK_CALIB_RETURN_FRAMES", "45")),
        "step_m": float(os.environ.get("OMNILRS_IK_CALIB_STEP", "0.015")),
        "min_tool_z": float(os.environ.get("OMNILRS_IK_CALIB_MIN_TOOL_Z", "0.35")),
        "orientation_mode": os.environ.get("OMNILRS_IK_CALIB_ORIENTATION", "base_down").strip().lower(),
        "use_debug_yaw": os.environ.get("OMNILRS_IK_CALIB_USE_DEBUG_YAW", "1").lower() in ["1", "true", "yes", "on"],
        "debug_pose_ready": False,
        "base_locked": False,
        "target_quat": None,
        "axis_start": None,
        "target_pos": None,
        "ik_failures": 0,
    }
    ik_calib_axes = [
        ("+X", np.array([1.0, 0.0, 0.0], dtype=np.float64)),
        ("+Y", np.array([0.0, 1.0, 0.0], dtype=np.float64)),
        ("+Z", np.array([0.0, 0.0, 1.0], dtype=np.float64)),
    ]

    def fmt_m(vec):
        vec = np.array(vec, dtype=np.float64)
        return f"({vec[0]:+.4f},{vec[1]:+.4f},{vec[2]:+.4f})m"

    def fmt_mm(vec):
        vec = np.array(vec, dtype=np.float64) * 1000.0
        return f"({vec[0]:+.1f},{vec[1]:+.1f},{vec[2]:+.1f})mm"

    def capture_ik_calib_pose():
        ee_pos, ee_quat = ee_solver.compute_end_effector_pose()
        tool_pos = get_current_gripper_control_center()
        wrist_pos, wrist_source = get_wrist3_link_world_position()
        base_pos, base_quat = my_robot.get_world_pose()
        return {
            "ee_pos": np.array(ee_pos, dtype=np.float64),
            "ee_quat": quat_wxyz_normalize(ee_quat),
            "tool_pos": None if tool_pos is None else np.array(tool_pos, dtype=np.float64),
            "wrist_pos": None if wrist_pos is None else np.array(wrist_pos, dtype=np.float64),
            "wrist_source": wrist_source,
            "base_pos": np.array(base_pos, dtype=np.float64),
            "base_quat": quat_wxyz_normalize(base_quat),
        }

    def apply_ik_calib_target(target_pos):
        try:
            action, success = ee_solver.compute_inverse_kinematics(
                target_position=np.array(target_pos, dtype=np.float64),
                target_orientation=ik_calib_state["target_quat"],
            )
            if success:
                my_robot.apply_action(action)
            return bool(success)
        except Exception as e:
            ik_calib_state["last_error"] = str(e)
            return False

    def save_ik_calibration_result(payload):
        out_path = os.path.join(os.getcwd(), "ik_calibration_latest.json")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            my_print(f"💾 [IKCalib] 标定结果已保存: {out_path}")
        except Exception as e:
            my_print(f"⚠️ [IKCalib] 标定结果保存失败: {e}")

    def finish_ik_calibration():
        valid = [r for r in ik_calib_state["results"] if r.get("success")]
        payload = {
            "created_at": datetime.datetime.now().isoformat(),
            "step_m": ik_calib_state["step_m"],
            "move_frames": ik_calib_state["move_frames"],
            "return_frames": ik_calib_state["return_frames"],
            "orientation_mode": ik_calib_state["orientation_mode"],
            "gripper_control_source": gripper_control_source,
            "gripper_control_paths": gripper_control_paths,
            "results": [],
        }
        for r in ik_calib_state["results"]:
            payload["results"].append({
                "axis": r["axis"],
                "success": bool(r.get("success")),
                "ik_failures": int(r.get("ik_failures", 0)),
                "commanded_solver_delta": np.array(r["commanded_solver_delta"], dtype=np.float64).tolist(),
                "actual_solver_delta": np.array(r["actual_solver_delta"], dtype=np.float64).tolist(),
                "tool_world_delta": np.array(r["tool_world_delta"], dtype=np.float64).tolist(),
                "wrist_world_delta": np.array(r["wrist_world_delta"], dtype=np.float64).tolist(),
                "return_tool_residual": np.array(r.get("return_tool_residual", [0, 0, 0]), dtype=np.float64).tolist(),
            })

        if len(valid) >= 3:
            solver_cols = np.stack([np.array(r["actual_solver_delta"], dtype=np.float64) for r in valid[:3]], axis=1)
            tool_cols = np.stack([np.array(r["tool_world_delta"], dtype=np.float64) for r in valid[:3]], axis=1)
            wrist_cols = np.stack([np.array(r["wrist_world_delta"], dtype=np.float64) for r in valid[:3]], axis=1)
            solver_rank = int(np.linalg.matrix_rank(solver_cols))
            solver_to_tool_world = tool_cols @ np.linalg.pinv(solver_cols)
            solver_to_wrist_world = wrist_cols @ np.linalg.pinv(solver_cols)
            tool_world_to_solver = np.linalg.pinv(solver_to_tool_world)
            residual = tool_cols - solver_to_tool_world @ solver_cols
            payload["solver_actual_columns"] = solver_cols.tolist()
            payload["tool_world_columns"] = tool_cols.tolist()
            payload["wrist_world_columns"] = wrist_cols.tolist()
            payload["solver_to_tool_world_matrix"] = solver_to_tool_world.tolist()
            payload["solver_to_wrist_world_matrix"] = solver_to_wrist_world.tolist()
            payload["tool_world_to_solver_matrix"] = tool_world_to_solver.tolist()
            payload["fit_residual_norm_m"] = float(np.linalg.norm(residual))
            payload["solver_rank"] = solver_rank

            my_print("📐 [IKCalib] solver_delta -> tool_world_delta 线性矩阵 M，含义: world_delta ≈ M @ solver_delta")
            for row in solver_to_tool_world:
                my_print(f"📐 [IKCalib] M row: [{row[0]:+.5f}, {row[1]:+.5f}, {row[2]:+.5f}]")
            my_print("📐 [IKCalib] tool_world_delta -> solver_delta 反解矩阵 Pinv(M)，后续 AnyGrasp 应优先用这个。")
            for row in tool_world_to_solver:
                my_print(f"📐 [IKCalib] Pinv row: [{row[0]:+.5f}, {row[1]:+.5f}, {row[2]:+.5f}]")
            my_print(
                f"📐 [IKCalib] solver_rank={solver_rank}, fit_residual={np.linalg.norm(residual)*1000:.2f}mm"
            )
        else:
            my_print(f"❌ [IKCalib] 有效轴数量不足: {len(valid)}/3，不能生成3D标定矩阵。")

        save_ik_calibration_result(payload)
        my_print("✅ [IKCalib] 标定流程结束。不会进入 AnyGrasp 抓取。")

    def step_ik_calibration_state():
        nonlocal ROBOT_STATE, debug_pre_yaw_settle_frames, debug_base_yaw_settle_frames
        freeze_base_and_wheels(hard_pose_lock=False)
        gripper.open()

        try:
            if ik_calib_state["use_debug_yaw"] and not ik_calib_state["debug_pose_ready"]:
                if debug_pre_yaw_settle_frames > 0:
                    if debug_pre_yaw_settle_frames == DEBUG_PRE_YAW_SETTLE_FRAMES:
                        my_print(
                            f"🧪 [IKCalib] 标定前复现抓取调试姿态：先稳定 "
                            f"{DEBUG_PRE_YAW_SETTLE_FRAMES} 帧，再执行 yaw_delta={DEBUG_BASE_YAW_DEG:+.1f}deg。"
                        )
                    debug_pre_yaw_settle_frames -= 1
                    return
                if not debug_base_rotated:
                    rotate_debug_base_after_startup()
                    return
                if debug_base_yaw_settle_frames > 0:
                    debug_base_yaw_settle_frames -= 1
                    if debug_base_yaw_settle_frames == 0:
                        base_pos, _ = my_robot.get_world_pose()
                        my_print(
                            f"🧪 [IKCalib] 标定前车身姿态稳定完成: "
                            f"base_pos=({base_pos[0]:.3f},{base_pos[1]:.3f},{base_pos[2]:.3f})。"
                        )
                    return
                ik_calib_state["debug_pose_ready"] = True

            if ik_calib_state["phase"] == "INIT":
                if not ik_calib_state["base_locked"]:
                    create_grasp_base_lock()
                    ik_calib_state["base_locked"] = True
                pose = capture_ik_calib_pose()
                tool_pos = pose["tool_pos"]
                if tool_pos is None:
                    my_print("❌ [IKCalib] 无法读取真实夹爪中心，停止标定。")
                    ROBOT_STATE = "DONE"
                    return
                if tool_pos[2] < ik_calib_state["min_tool_z"]:
                    my_print(
                        f"❌ [IKCalib] 当前夹爪中心过低，拒绝标定: "
                        f"tool_z={tool_pos[2]:.3f}m < {ik_calib_state['min_tool_z']:.3f}m"
                    )
                    ROBOT_STATE = "DONE"
                    return
                if ik_calib_state["orientation_mode"] in ["current", "keep", "same"]:
                    ik_calib_state["target_quat"] = pose["ee_quat"]
                else:
                    ik_calib_state["target_quat"] = np.array([0.0, 0.7071, 0.7071, 0.0], dtype=np.float64)

                my_print("\n" + "█" * 62)
                my_print("🧪 [IKCalib] 启动 IK 三轴标定模式：不调用 AnyGrasp，不闭合夹爪，不抓石头。")
                my_print(
                    f"🧪 [IKCalib] step={ik_calib_state['step_m']*1000:.1f}mm, "
                    f"move_frames={ik_calib_state['move_frames']}, "
                    f"return_frames={ik_calib_state['return_frames']}, "
                    f"orientation={ik_calib_state['orientation_mode']}"
                )
                my_print(
                    f"🧪 [IKCalib] start ee_solver={fmt_m(pose['ee_pos'])}, "
                    f"tool_world={fmt_m(tool_pos)}, wrist_world={fmt_m(pose['wrist_pos'])}, "
                    f"wrist_source={pose['wrist_source']}"
                )
                my_print(
                    f"🧪 [IKCalib] base_world=({pose['base_pos'][0]:.3f},{pose['base_pos'][1]:.3f},{pose['base_pos'][2]:.3f}), "
                    f"tool_source={gripper_control_source}, tool_paths={gripper_control_paths}"
                )
                my_print("█" * 62 + "\n")
                ik_calib_state["phase"] = "START_AXIS"
                ik_calib_state["frame"] = 0
                return

            if ik_calib_state["phase"] == "START_AXIS":
                if ik_calib_state["axis_index"] >= len(ik_calib_axes):
                    finish_ik_calibration()
                    ROBOT_STATE = "DONE"
                    return
                axis_name, axis_vec = ik_calib_axes[ik_calib_state["axis_index"]]
                pose = capture_ik_calib_pose()
                command = axis_vec * ik_calib_state["step_m"]
                ik_calib_state["axis_start"] = pose
                ik_calib_state["axis_name"] = axis_name
                ik_calib_state["axis_command"] = command
                ik_calib_state["target_pos"] = pose["ee_pos"] + command
                ik_calib_state["ik_failures"] = 0
                ik_calib_state["frame"] = 0
                ik_calib_state["phase"] = "MOVE_AXIS"
                my_print(
                    f"➡️ [IKCalib] 测试 solver {axis_name}: "
                    f"start_ee={fmt_m(pose['ee_pos'])}, start_tool={fmt_m(pose['tool_pos'])}, "
                    f"cmd_solver_delta={fmt_mm(command)}"
                )
                return

            if ik_calib_state["phase"] == "MOVE_AXIS":
                ok = apply_ik_calib_target(ik_calib_state["target_pos"])
                if not ok:
                    ik_calib_state["ik_failures"] += 1
                ik_calib_state["frame"] += 1
                if ik_calib_state["frame"] % 15 == 1:
                    pose = capture_ik_calib_pose()
                    start = ik_calib_state["axis_start"]
                    my_print(
                        f"➡️ [IKCalib] solver {ik_calib_state['axis_name']} 移动中: "
                        f"frame={ik_calib_state['frame']}/{ik_calib_state['move_frames']}, "
                        f"actual_solver={fmt_mm(pose['ee_pos'] - start['ee_pos'])}, "
                        f"tool_world={fmt_mm(pose['tool_pos'] - start['tool_pos'])}, "
                        f"ik_failures={ik_calib_state['ik_failures']}"
                    )
                if ik_calib_state["frame"] >= ik_calib_state["move_frames"]:
                    ik_calib_state["phase"] = "MEASURE_AXIS"
                return

            if ik_calib_state["phase"] == "MEASURE_AXIS":
                start = ik_calib_state["axis_start"]
                after = capture_ik_calib_pose()
                axis_name = ik_calib_state["axis_name"]
                result = {
                    "axis": axis_name,
                    "success": ik_calib_state["ik_failures"] < max(3, ik_calib_state["move_frames"] // 2),
                    "ik_failures": ik_calib_state["ik_failures"],
                    "commanded_solver_delta": ik_calib_state["axis_command"].copy(),
                    "actual_solver_delta": after["ee_pos"] - start["ee_pos"],
                    "tool_world_delta": after["tool_pos"] - start["tool_pos"],
                    "wrist_world_delta": after["wrist_pos"] - start["wrist_pos"],
                }
                ik_calib_state["results"].append(result)
                my_print(
                    f"✅ [IKCalib] solver {axis_name} 测量: "
                    f"cmd_solver={fmt_mm(result['commanded_solver_delta'])}, "
                    f"actual_solver={fmt_mm(result['actual_solver_delta'])}, "
                    f"tool_world_delta={fmt_mm(result['tool_world_delta'])}, "
                    f"wrist_world_delta={fmt_mm(result['wrist_world_delta'])}, "
                    f"ik_failures={result['ik_failures']}"
                )
                ik_calib_state["frame"] = 0
                ik_calib_state["phase"] = "RETURN_AXIS"
                return

            if ik_calib_state["phase"] == "RETURN_AXIS":
                start = ik_calib_state["axis_start"]
                ok = apply_ik_calib_target(start["ee_pos"])
                if not ok:
                    ik_calib_state["ik_failures"] += 1
                ik_calib_state["frame"] += 1
                if ik_calib_state["frame"] >= ik_calib_state["return_frames"]:
                    returned = capture_ik_calib_pose()
                    residual = returned["tool_pos"] - start["tool_pos"]
                    ik_calib_state["results"][-1]["return_tool_residual"] = residual
                    my_print(
                        f"↩️ [IKCalib] solver {ik_calib_state['axis_name']} 已回起点: "
                        f"return_tool_residual={fmt_mm(residual)}, "
                        f"return_solver_residual={fmt_mm(returned['ee_pos'] - start['ee_pos'])}"
                    )
                    ik_calib_state["axis_index"] += 1
                    ik_calib_state["phase"] = "START_AXIS"
                    ik_calib_state["frame"] = 0
                return

        except Exception as e:
            my_print(f"❌ [IKCalib] 标定异常: {e}")
            ROBOT_STATE = "DONE"

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
            if ROBOT_STATE == "IK_CALIBRATE":
                step_ik_calibration_state()
                step_counter += 1
                continue

            if GRASP_DEBUG_MODE and ROBOT_STATE == "GRASP_DEBUG_INIT":
                if debug_pre_yaw_settle_frames > 0:
                    if debug_pre_yaw_settle_frames == DEBUG_PRE_YAW_SETTLE_FRAMES:
                        my_print(
                            f"🧪 [GraspDebug] demo2前90帧初始化已完成，先原地稳定 "
                            f"{DEBUG_PRE_YAW_SETTLE_FRAMES} 帧，再执行调试右转。"
                        )
                    debug_pre_yaw_settle_frames -= 1
                    freeze_base_and_wheels(hard_pose_lock=False)
                    if debug_pre_yaw_settle_frames == 0:
                        base_pos, _ = my_robot.get_world_pose()
                        my_print(
                            f"🧪 [GraspDebug] demo2初始化后稳定等待完成: "
                            f"base_pos=({base_pos[0]:.3f},{base_pos[1]:.3f},{base_pos[2]:.3f})，开始右转。"
                        )
                elif not debug_base_rotated:
                    rotate_debug_base_after_startup()
                    freeze_base_and_wheels(hard_pose_lock=False)
                elif debug_base_yaw_settle_frames > 0:
                    debug_base_yaw_settle_frames -= 1
                    freeze_base_and_wheels(hard_pose_lock=False)
                    if debug_base_yaw_settle_frames == 0:
                        base_pos, base_ori = my_robot.get_world_pose()
                        my_print(
                            f"🧪 [GraspDebug] 车身转向稳定完成: "
                            f"base_pos=({base_pos[0]:.3f},{base_pos[1]:.3f},{base_pos[2]:.3f})，"
                            "目标月岩保持在标定坐标，下一步读取高杆相机 ROI。"
                        )
                elif not debug_base_crept_forward and abs(DEBUG_BASE_CREEP_FORWARD_M) > 0.001:
                    # ── 底盘前移：手臂水平伸展太长会限制垂直工作空间 ──
                    base_pos, base_ori = my_robot.get_world_pose()
                    base_pos = np.array(base_pos, dtype=np.float64)
                    base_ori = quat_wxyz_normalize(base_ori)
                    forward_dir = quat_wxyz_apply(base_ori, np.array([1.0, 0.0, 0.0], dtype=np.float64))
                    forward_dir[2] = 0.0
                    forward_dir = forward_dir / max(np.linalg.norm(forward_dir), 1e-12)
                    new_base = base_pos + forward_dir * DEBUG_BASE_CREEP_FORWARD_M
                    my_robot.set_world_pose(position=new_base, orientation=base_ori)
                    my_robot.set_linear_velocity(np.zeros(3))
                    my_robot.set_angular_velocity(np.zeros(3))
                    debug_base_crept_forward = True
                    my_print(
                        f"🚗 [GraspDebug] 底盘前移 {DEBUG_BASE_CREEP_FORWARD_M:+.2f}m: "
                        f"({base_pos[0]:.3f},{base_pos[1]:.3f}) → ({new_base[0]:.3f},{new_base[1]:.3f})"
                    )
                elif not debug_rock_relocated:
                    if DEBUG_RELOCATE_ROCK:
                        relocate_debug_rock_after_startup()
                    else:
                        debug_rock_relocated = True
                        debug_rock_settle_frames = max(0, DEBUG_ROCK_SETTLE_FRAMES)
                        debug_init_attempts = 0
                        dino_tracker.last_u, dino_tracker.last_v = None, None
                        current_rock_pos = read_rock_world_pos()
                        if current_rock_pos is not None:
                            my_print(
                                f"🧪 [GraspDebug] 保持月岩标定坐标，不执行二次移动: "
                                f"rock=({current_rock_pos[0]:.3f},{current_rock_pos[1]:.3f},{current_rock_pos[2]:.3f}), "
                                f"settle_frames={debug_rock_settle_frames}"
                            )
                        else:
                            my_print("🧪 [GraspDebug] 保持月岩标定坐标，不执行二次移动；但无法读回 rock pose。")
                    freeze_base_and_wheels(hard_pose_lock=False)
                elif debug_rock_settle_frames > 0:
                    debug_rock_settle_frames -= 1
                    freeze_base_and_wheels(hard_pose_lock=False)
                    if debug_rock_settle_frames == 0:
                        settled_pos = read_rock_world_pos()
                        if settled_pos is not None:
                            my_print(
                                f"🧪 [GraspDebug] 月岩标定位置沉降/稳定完成: "
                                f"settled rock pos=({settled_pos[0]:.3f},{settled_pos[1]:.3f},{settled_pos[2]:.3f})，"
                                "开始读取高杆相机 ROI。"
                            )
                        else:
                            my_print("🧪 [GraspDebug] 月岩标定位置稳定完成，但无法读回坐标，开始读取高杆相机 ROI。")
                elif step_counter % VISION_INTERVAL == 0:
                    step_grasp_debug_init()
                else:
                    freeze_base_and_wheels(hard_pose_lock=False)
                step_counter += 1
                continue

            if ROBOT_STATE == "RESTORE_ARM":
                step_restore_arm_state()
                step_counter += 1
                continue

            if ROBOT_STATE in ["ARM_LOCKING", "GRASPING"]:
                step_active_grasp_state()
                step_counter += 1
                continue

            if ROBOT_STATE == "VERIFY_GRASP":
                step_verify_grasp_state()
                step_counter += 1
                continue

            if ROBOT_STATE == "DONE":
                hold_done_state()
                step_counter += 1
                continue

            if step_counter % VISION_INTERVAL == 0: 
                rgba_data = mast_cam.get_rgba()
                
                if rgba_data is not None and rgba_data.size > 0:
                    rgb_img = rgba_data[:, :, :3]
                    latest_rgb_img = rgb_img
                    img_h, img_w = rgb_img.shape[:2]
                    
                    # ====================================================
                    # 🟡 状态 1: 搜索与 EQA 终极锁定 (停车思考模式)
                    # ====================================================
                    if ROBOT_STATE == "SEARCHING":
                        release_grasp_base_lock()
                        locked_base_pos = None
                        locked_base_ori = None
                        locked_wheel_pos = None
                        locked_joint_positions = None  # 用来记忆刹车瞬间的关节角度
                        # 🚨 停车思考原则：大模型看图时，底盘必须绝对静止！不许乱动！
                        if len(wheel_prims) > 0:
                            for prim in wheel_prims:
                                UsdPhysics.DriveAPI.Get(prim, "angular").GetTargetVelocityAttr().Set(0.0)

                        my_print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] 🔍 [状态: SEARCHING] 原地停车，启动 VLM+DINO 级联搜索...")
                        task_dict = vlm_brain.extract_target_prompt(rgb_img, base_instruction)
                        
                        if task_dict:
                            target_prompt = task_dict.get('target_prompt', 'rock.')
                            locked_target_prompt = target_prompt
                            my_print(f"🎯 VLM 锁定目标特征: {target_prompt}，正在呼叫 DINO 执行全局 Set-of-Mark...")
                            
                            # 强制进行一次全局搜索并让 VLM 裁判拍板
                            res_global = requests.post(dino_tracker.server_url, json={"image_base64": dino_tracker._encode_image(rgb_img), "text_prompt": target_prompt}, timeout=5).json()
                            
                            if res_global.get("found"):
                                # DINO 可能返回了多个框，我们用记忆系统挑出最好的，或者直接让 VLM 裁决第一个
                                best_box = dino_tracker._find_best_box_by_memory(res_global["boxes"])
                                
                                my_print("⚖️ 提交 VLM 裁判庭进行最终目标确认 (EQA 拍板)...")
                                if vlm_brain.verify_dino_box(rgb_img, best_box, target_prompt):
                                    my_print("✅ [EQA 确认通过] 目标已绝对锁死！VLM 进入休眠，移交底盘控制权！")
                                    locked_roi_box = best_box
                                    ROBOT_STATE = "TRACKING" # 🔴 状态切换！
                                else:
                                    my_print("❌ [EQA 驳回] 裁判认为不是目标！继续搜索...")
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
                        margin_x = int((locked_roi_box[2] - locked_roi_box[0]) * 0.8)
                        margin_y = int((locked_roi_box[3] - locked_roi_box[1]) * 0.8)
                        
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
                                    my_print(f"全图Debug保存失败: {e}")
                                locked_roi_box = g_box 
                                
                                # 获取极其精准的 3D 深度 (来自物理深度相机)
                                x3d, y3d, z3d = dino_tracker._get_3d_coordinates(rgb_img, g_box)
                                my_print(f"🔒 [静默追踪] X={x3d:.3f}m, Y={y3d:.3f}m, 深度Z={z3d:.3f}m")

                                # ==========================================
                                # 🚨
                                # ==========================================
                                if z3d < 0.40:
                                    # 如果深度小于 0.4 米，大概率是框到了车头或者石头已经钻进盲区！
                                    my_print(f"⚠️ [警告] 深度极度异常(Z={z3d:.3f}m)，目标可能进入盲区或发生幻觉！强制制动！")
                                    for prim in wheel_prims:
                                        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                                        drive.GetTargetVelocityAttr().Set(0.0)
                                        drive.GetDampingAttr().Set(100000.0)
                                    final_cam_pos = np.array([x3d, y3d, z3d])
                                    ROBOT_STATE = "ARRIVED"
                                    continue # 直接跳入下一次循环，不再执行下面的代码
                                
                                # ==========================================
                                # 🛑 核心分支 A：到达黄金抓取区，准备交接
                                # ==========================================
                                TARGET_Z_MIN = 0.65  
                                TARGET_Z_MAX = 0.85  
                                
                                # 计算目标在画面中的像素偏差
                                u_center = (g_box[0] + g_box[2]) / 2.0
                                current_yaw_error = u_center - (img_w / 2.0)
                                
                                # 首先判断距离是否进入黄金区间
                                if TARGET_Z_MIN < z3d < TARGET_Z_MAX:
                                    # 其次判断是否严格居中 (像素偏差小于 40)
                                    if abs(current_yaw_error) < 40:
                                        my_print(f"\n🛑 [到达工作空间] 深度Z={z3d:.3f}m, 完美居中！底盘紧急制动！")
                                        # 底盘电机归零，强制刹车锁死并加阻尼防溜车
                                        for prim in wheel_prims:
                                            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                                            drive.GetTargetVelocityAttr().Set(0.0)
                                            drive.GetMaxForceAttr().Set(1e8)
                                            # 抓取阶段底盘由 FixedJoint 锁定，轮子只做温和制动，避免物理求解爆振。
                                            drive.GetDampingAttr().Set(1e6)
                                            drive.GetStiffnessAttr().Set(0.0)
                                        
                                        # 保存最后时刻的相机相对坐标，交接状态机
                                        final_cam_pos = np.array([x3d, y3d, z3d])
                                        ROBOT_STATE = "ARRIVED"
                                    else:
                                        # 距离达标但未对准：强制原地搓轮子微调，绝不前进
                                        my_print(f"⚠️ [近距微调] 距离达标但未居中(偏差{current_yaw_error:.1f}px)，原地微调中...")
                                        v_yaw = current_yaw_error * 0.15
                                        if current_yaw_error > 0: 
                                            v_yaw = max(18.0, min(v_yaw, 25.0))
                                        else: 
                                            v_yaw = min(-18.0, max(v_yaw, -25.0))
                                        
                                        # 🚨 [抗溜车补丁] 给一个向前的微小推力，抵消重力下滑！
                                        v_crawl = -6.0  
                                        
                                        for prim in wheel_prims:
                                            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                                            drive.GetDampingAttr().Set(100.0)
                                            # 左轮和右轮都叠加上这个向前的推力
                                            if "left" in prim.GetName().lower(): 
                                                drive.GetTargetVelocityAttr().Set(v_crawl + v_yaw)
                                            else: 
                                                drive.GetTargetVelocityAttr().Set(v_crawl - v_yaw)
                                    
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
                                my_print("⚠️ 局部视野丢失目标！退回全局搜索模式重新确认。")
                                ROBOT_STATE = "SEARCHING"
                                dino_tracker.last_u, dino_tracker.last_v = None, None
                        except Exception as e:
                            my_print(f"Tracking Error: {e}")
                    # ====================================================
                    # 🎯 状态 3: 到达交接 (高杆粗引导，手腕视觉接管)
                    # ====================================================
                    elif ROBOT_STATE == "ARRIVED":
                        try:
                            my_print("\n" + "█"*55)
                            my_print("📡 [系统交接] 底盘任务结束；不再把高杆像素反投影成抓取坐标。")
                            my_print("🦾 底盘进入休眠，启动高杆粗引导 + 手腕相机视觉扫描抓取。")
                            my_print("█"*55 + "\n")

                            locked_base_pos, locked_base_ori = my_robot.get_world_pose()
                            locked_wheel_pos = my_robot.get_joint_positions()
                            create_grasp_base_lock()

                            if latest_rgb_img is None or locked_roi_box is None:
                                my_print("❌ [WristVision] 缺少高杆RGB/ROI交接数据，无法启动粗引导。")
                                release_grasp_base_lock()
                                ROBOT_STATE = "DONE"
                                continue

                            if USE_ANYGRASP:
                                x0, y0, x1, y1 = [float(v) for v in locked_roi_box]
                                target_pixel = np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)], dtype=np.float64)
                                target_world = mast_box_to_world(latest_rgb_img, locked_roi_box)
                                if target_world is not None:
                                    anygrasp_target_world = np.array(target_world, dtype=np.float64)
                                    anygrasp_target_source = "mast_arrived_depth"
                                    my_print(
                                        f"🎯 [MastGuide] 到达交接目标世界点: "
                                        f"target_world=({anygrasp_target_world[0]:.3f},{anygrasp_target_world[1]:.3f},{anygrasp_target_world[2]:.3f})"
                                    )
                                else:
                                    anygrasp_target_world = read_rock_world_pos() if GRASP_DEBUG_MODE else None
                                    anygrasp_target_source = "arrived_debug_fallback" if anygrasp_target_world is not None else "unset"
                                    my_print("⚠️ [MastGuide] 到达交接ROI没有可用深度，退回调试rock pose或等待手腕拒绝全图。")
                                my_print("🦾 [AnyGrasp 接管] 高杆ROI只作为粗提示；手腕RGB-D/点云模型决定6D抓取位姿。")
                                capture_grasp_start_pose()
                                grasp_executor.start(
                                    target_pixel,
                                    locked_roi_box,
                                    locked_target_prompt,
                                    grasp_width=0.055,
                                    mast_rgb=latest_rgb_img,
                                )
                                ROBOT_STATE = "ARM_LOCKING"
                                continue

                            plan = vlm_brain.propose_grasp_plan(latest_rgb_img, locked_roi_box, locked_target_prompt)
                            if not plan.get("target_confirmed", True):
                                grasp_retry_count += 1
                                my_print(f"❌ [WristVision] VLM认为高杆ROI不是可抓取岩石，retry={grasp_retry_count}")
                                release_grasp_base_lock()
                                ROBOT_STATE = recovery_state_after_grasp_failure() if grasp_retry_count <= 2 else "DONE"
                                continue

                            x0, y0, x1, y1 = [float(v) for v in locked_roi_box]
                            target_pixel = plan.get("grasp_pixel", None)
                            if not (isinstance(target_pixel, list) and len(target_pixel) == 2):
                                target_pixel = np.array([
                                    0.5 * (x0 + x1),
                                    0.5 * (y0 + y1),
                                ], dtype=np.float64)
                            else:
                                target_pixel = np.array(target_pixel, dtype=np.float64)
                            if not (x0 <= target_pixel[0] <= x1 and y0 <= target_pixel[1] <= y1):
                                target_pixel = np.array([0.5 * (x0 + x1), y0 + 0.45 * (y1 - y0)], dtype=np.float64)
                                my_print(f"⚠️ [WristVision] VLM像素落在ROI外，只作为粗方向，拉回ROI中心: pixel={target_pixel.tolist()}")

                            grasp_width = float(np.clip(plan.get("gripper_width_m", 0.055), 0.025, 0.10))
                            my_print("🦾 [WristVision 接管] 高杆只给前/左/右粗方向；手腕相机看到岩石后再闭环抓。")
                            capture_grasp_start_pose()
                            grasp_executor.start(target_pixel, locked_roi_box, locked_target_prompt, grasp_width=grasp_width, mast_rgb=latest_rgb_img)
                            ROBOT_STATE = "ARM_LOCKING"
                                
                        except Exception as e:
                            my_print(f"❌ 交接失败: {e}")
                            release_grasp_base_lock()
                            ROBOT_STATE = "DONE"

                    # ====================================================
                    # 🎯 状态 4: 任务完成挂起 (保留在视觉周期内的兜底分支)
                    # ====================================================
                    elif ROBOT_STATE == "DONE":
                        hold_done_state()
                    

            step_counter += 1

    simulation_app.close()

if __name__ == "__main__":
    run()
