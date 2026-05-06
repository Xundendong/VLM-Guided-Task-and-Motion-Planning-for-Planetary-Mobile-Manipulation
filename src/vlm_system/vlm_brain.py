import base64
import json
import re
from PIL import Image
import io
import os
import numpy as np
import requests

class VLMAgent:
    def __init__(self, api_key=None, model_name="local-qwen2.5-vl-3b"):
        self.model_name = model_name
        self.server_url = "http://127.0.0.1:8000/vlm_decide" 
        print(f"✅ [大脑代理] VLM 中枢初始化完毕 (对接本地模型: {self.model_name})")

    def _encode_image(self, rgb_array):
        if rgb_array.dtype != np.uint8:
            rgb_array = rgb_array.astype(np.uint8)
        img = Image.fromarray(rgb_array)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def get_action(self, rgb_image, instruction, save_dir=None, step=0):
        """核心推理接口：基于‘前景突出物’的精准避障感知"""
        if save_dir is not None:
            img_path = os.path.join(save_dir, f"vision_frame_{step:04d}.jpg")
            img_to_save = Image.fromarray(rgb_image.astype(np.uint8))
            img_to_save.save(img_path)
            print(f"📸 [视觉留档] 小车当前视野已存入: {img_path}")

        img_str = self._encode_image(rgb_image)
        
        # 🚀 物理认知终极进化版：只关注“近处”、“突出”的岩石！无视远方！
        prompt = f"""
        你是一台月球漫游车。这是你车头相机的实时画面。
        任务目标：{instruction}
        
        【老司机驾驶法则 - 目标过滤】：
        1. 真正的威胁 (必须避让)：只有当画面【最下方贴近底边的区域】（代表就在你车轮跟前）出现了【明显突出地面的大块岩石】时，才需要原地转弯（turn_vel设为2.0或-2.0，forward_vel设为0.0）。
        2. 无视远方与坑洼 (放心直行)：远处的石头、浅浅的陨石坑、平缓的坡度、或者画面两侧不挡路的碎石，【一律不要管】！你的底盘很高，可以直接开过去。请果断给出 forward_vel (1.0 到 3.0)，turn_vel 为 0.0。
        
        【输出规范】：
        只返回 JSON 对象：
        {{
            "observation": "明确说明石头是在'近处阻挡'还是在'远处安全/只是陨石坑'",
            "forward_vel": 0.0到3.0,
            "turn_vel": -2.0到2.0
        }}
        """
        
        try:
            payload = {"image_base64": img_str, "instruction": prompt}
            response = requests.post(self.server_url, json=payload, timeout=30)
            if response.status_code == 200:
                result_text = response.json().get("result", "")
                return self._parse_json_action(result_text)
            return 0.0, 0.0
        except Exception as e:
            print(f"❌ [网络异常]: {e}")
            return 0.0, 0.0

    def _parse_json_action(self, text):
        """物理级限速接管系统"""
        print(f"🧐 [大脑原始回复明文]:\n{text}")
        try:
            clean_text = text.replace("```json", "").replace("```", "").strip()
            json_match = re.search(r'\{.*?\}', clean_text, re.DOTALL)
            if json_match:
                action_dict = json.loads(json_match.group())
                
                print(f"👁️ [大脑视觉报告]: {action_dict.get('observation', '未报告')}")
                
                raw_v_x = float(action_dict.get("forward_vel", 0.0))
                raw_v_yaw = float(action_dict.get("turn_vel", 0.0))
                
                # 🛡️ 底盘接管逻辑：严格根据模型的数值输出执行
                if abs(raw_v_yaw) > 0.1 or raw_v_x < 0.5:
                    v_yaw = raw_v_yaw * 25.0 if raw_v_yaw != 0 else 30.0 
                    v_x = 0.0                 
                    print("⚠️ [底盘接管] 发现近处致命威胁！原地掉头避障！")
                else:
                    v_yaw = 0.0
                    v_x = raw_v_x * 15.0      
                    print("🚀 [底盘接管] 远方障碍无视，稳健推进。")
                return v_x, v_yaw
            return 0.0, 0.0
        except Exception:
            return 0.0, 0.0