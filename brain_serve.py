# brain_serve.py
import os
# ==========================================
# 🛑 全方位无死角清除系统残留代理环境变量
# ==========================================
proxy_vars = [
    "http_proxy", "https_proxy", "all_proxy", 
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"
]
for var in proxy_vars:
    os.environ.pop(var, None)
os.environ["NO_PROXY"] = "*"

from fastapi import FastAPI
from pydantic import BaseModel
# ==========================================
# 🔴 拯救 DINO：强行指定 Hugging Face 国内镜像站！
# ==========================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import base64
import io
from PIL import Image
from modelscope import snapshot_download
from transformers import (
    Qwen2_5_VLForConditionalGeneration, 
    AutoProcessor, 
    BitsAndBytesConfig,
    AutoModelForZeroShotObjectDetection # 加载 DINO
)
from qwen_vl_utils import process_vision_info

app = FastAPI()

# ========================================================
# 🧠 模块 1：VLM 大脑 (Qwen2.5-VL-3B 4-bit 极限显存版)
# ========================================================
print("⏳ [大脑初始化] 正在拉取 Qwen2.5-VL-7B，准备 4-bit 量化...")
# ✅ 换回 3B 模型
model_dir = snapshot_download('qwen/Qwen2.5-VL-7B-Instruct')

# ✅ 4-bit 量化配置 (让 7B 模型体积缩减到极致)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",             
    bnb_4bit_compute_dtype=torch.bfloat16, 
    bnb_4bit_use_double_quant=True         
)

# 加载 VLM 模型
vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_dir, 
    device_map="auto",
    quantization_config=quantization_config
)
vlm_processor = AutoProcessor.from_pretrained(model_dir)
print("✅ Qwen2.5-VL-7B 大脑上线！显存毫无压力。")

# ========================================================
# 👁️ 模块 2：DINO 鹰眼 (Grounding DINO Tiny)
# ========================================================
print("⏳ [鹰眼初始化] 正在拉取 Grounding DINO...")
dino_model_id = "IDEA-Research/grounding-dino-tiny"
dino_processor = AutoProcessor.from_pretrained(dino_model_id)
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_id).to("cuda")
print("✅ DINO 鹰眼上线！像素级狙击准备就绪。")


# ========================================================
# 🌐 接口定义
# ========================================================
class VLMRequest(BaseModel):
    image_base64: str
    instruction: str

class DinoRequest(BaseModel):
    image_base64: str
    text_prompt: str

# 🎯 大脑路由：处理逻辑推理与全局决策
@app.post("/vlm_decide")
async def vlm_decide(req: VLMRequest):
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    # 限制分辨率，进一步节省计算量
                    {"type": "image", "image": f"data:image/jpeg;base64,{req.image_base64}", "resized_height": 224, "resized_width": 224},
                    {"type": "text", "text": req.instruction}
                ],
            }
        ]
        
        text = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = vlm_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(vlm_model.device)

        generated_ids = vlm_model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = vlm_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        print(f"🧠 [3B 大脑决策]: {output_text}")
        return {"status": "success", "result": output_text}
        
    except Exception as e:
        print(f"❌ [大脑推理错误]: {e}")
        return {"status": "error", "result": ""}

from PIL import ImageDraw

# 🎯 鹰眼路由：增加 Debug 绘图输出功能
@app.post("/dino_detect")
async def dino_detect(request: DinoRequest):
    try:
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        width, height = image.size
        
        inputs = dino_processor(images=image, text=request.text_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = dino_model(**inputs)
            
        # 兼容层处理
        try:
            results = dino_processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, target_sizes=[image.size[::-1]]
            )[0]
        except TypeError:
            results = dino_processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, target_sizes=[image.size[::-1]]
            )[0]
        
        scores = results["scores"]
        boxes = results["boxes"]
        valid_indices = torch.where(scores > 0.25)[0]
        
        if len(valid_indices) > 0:
            best_valid_idx = valid_indices[torch.argmax(scores[valid_indices])].item()
            box = boxes[best_valid_idx].tolist() # [xmin, ymin, xmax, ymax]
            score = scores[best_valid_idx].item()
            
            # ==========================================
            # 🎨 DEBUG 绘图：输出 DINO 看到的画面
            # ==========================================
            draw = ImageDraw.Draw(image)
            draw.rectangle(box, outline="red", width=5)
            draw.text((box[0], box[1]), f"{request.text_prompt}: {score:.2f}", fill="red")
            image.save("debug_dino.jpg") # 每次检测都会覆盖这张图
            # ==========================================
            
            return {
                "found": True,
                "box": box,
                "score": score,
                "image_width": width,
                "image_height": height
            }
        else:
            return {"found": False}
    except Exception as e:
        print(f"❌ [错误]: {e}")
        return {"found": False}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)