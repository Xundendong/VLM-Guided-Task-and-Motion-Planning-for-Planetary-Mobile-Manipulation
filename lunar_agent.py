__author__ = "Antoine Richard & Customized by You (VLM Strict Edition)"
__license__ = "BSD-3-Clause"

import numpy as np
from omegaconf import DictConfig, OmegaConf, ListConfig
from src.configurations import configFactory
from src.environments_wrappers import startSim
import logging
import hydra
import carb
import os
import sys
import datetime

# ==========================================
# 📝 学术级“黑匣子”日志接管模块 (独立文件夹 + 终端双写)
# ==========================================
run_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.path.join(os.getcwd(), "logs", f"mission_{run_time}")
os.makedirs(LOG_DIR, exist_ok=True)
log_file_path = os.path.join(LOG_DIR, "mission_record.log")

class DualLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
        self.log.write(f"\n\n{'='*60}\n🚀 自动驾驶任务启动: {run_time}\n📂 本次任务记录存档于: {LOG_DIR}\n{'='*60}\n")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = DualLogger(log_file_path)

# 提前屏蔽 Python 层画图库警告
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

# ==========================================
# 纯手工打造：四元数乘法 [w, x, y, z]
# ==========================================
def custom_quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

@hydra.main(config_name="config", config_path="cfg")
def run(cfg: DictConfig):
    cfg = omegaconfToDict(cfg)
    cfg = instantiateConfigs(cfg)
    
    print("\n⏳ 环境启动中，请稍候...\n")
    SM, simulation_app = startSim(cfg)

    logging.getLogger("omni.physx.plugin").setLevel(logging.ERROR)
    carb.settings.get_settings().set("/log/level", "error")
    carb.settings.get_settings().set("/log/outputStreamLevel", "error")

    import omni.usd
    from pxr import UsdPhysics
    import omni.timeline
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.utils.extensions import get_extension_path_from_name
    from isaacsim.core.utils.types import ArticulationAction
    
    # 引入 VLM 模块
    from src.vlm_system.vlm_brain import VLMAgent
    from src.vlm_system.perception import CameraSystem

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

    print("\n" + "⚔️"*20)
    print("【VLM 自动驾驶版】系统初始化中...")
    
    # 切除自带控制器
    graphs_removed = 0
    paths_to_remove = []
    for prim in stage.Traverse():
        type_name = prim.GetTypeName()
        if type_name in ["ActionGraph", "OmniGraphNode"]:
            paths_to_remove.append(prim.GetPath())
            
    for p in paths_to_remove:
        stage.RemovePrim(p)
        graphs_removed += 1
    print(f"✅ 成功拔管！已强行切除 {graphs_removed} 个官方后台控制节点！")

    # 锁定车轮属性
    wheel_prims = []
    for prim in stage.Traverse():
        if prim.GetTypeName() == "PhysicsRevoluteJoint" and "wheel" in prim.GetName().lower():
            wheel_prims.append(prim)
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if not drive: drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive.GetStiffnessAttr().Set(0.0)
            drive.GetDampingAttr().Set(10000000.0)
            drive.GetTargetVelocityAttr().Set(0.0)
            
    print(f"✅ 成功锁定 {len(wheel_prims)} 个动力轮的 USD 底层属性！")

    print("🤖 正在初始化 UR3e 逆运动学(IK)大脑...")
    robot_path = "/Robots/husky/jackal"
    my_robot = Articulation(robot_path)
    my_robot.initialize()
    
    # 获取关节索引 (完全保留你的原版参数)
    safe_home_angles = np.array([0.0, -1.0, 1.0, -1.0, -1.57, 0.0])
    arm_joint_keywords = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
    joint_indices = []
    for kw in arm_joint_keywords:
        for idx, name in enumerate(my_robot.dof_names):
            if kw in name.lower():
                joint_indices.append(idx)
                break

    # 配置 Lula IK
    config_dir = os.path.join(mg_ext_path, "motion_policy_configs")
    urdf_path, yaml_path = "", ""
    for root, dirs, files in os.walk(config_dir):
        if "ur3e" in root.lower() or "universal_robots" in root.lower():
            for f in files:
                if f.endswith(".urdf") and "ur3e" in f.lower():
                    urdf_path = os.path.join(root, f)
                if f.endswith(".yaml") and ("descriptor" in f.lower() or "description" in f.lower()) and "ur3e" in root.lower():
                    yaml_path = os.path.join(root, f)
                    
    # 增加一个路径安全校验
    if not urdf_path or not yaml_path:
        print(f"\n❌ [系统致命错误] 找不到 Lula 配置文件。")
        sys.exit()

    ik_solver = LulaKinematicsSolver(robot_description_path=yaml_path, urdf_path=urdf_path)
    ee_solver = ArticulationKinematicsSolver(my_robot, ik_solver, "wrist_3_link")
    print("✅ 机械臂 IK 初始化完毕！")

    # ==========================================
    # 👁️ 初始化 VLM 感知系统和大模型
    # ==========================================
    cameras = CameraSystem(stage)
    cameras.init_cameras()
    brain = VLMAgent(model_name="local-qwen2.5-vl-3b")
    instruction = "避开前方的岩石和陨石坑，寻找平坦的月壤区域前进。"

    print("⚔️"*20 + "\n")
    
    # 物理引擎热身计数器
    startup_frames = 0
    step_counter = 0
    VLM_THINKING_INTERVAL = 120
    
    # ==========================================
    # 接管主循环
    # ==========================================
    while simulation_app.is_running():
        simulation_app.update()
        if not timeline.is_playing():
            timeline.play()

        # =========================================================
        # 🛡️ 物理引擎热身与开机自检 (原汁原味的三段式动画 0~90 帧)
        # =========================================================
        if timeline.is_playing() and startup_frames <= 90:
            
            # 【第一阶段：0~20帧】硬抗落地冲击，速度清零
            if startup_frames < 20:
                if len(joint_indices) == 6:
                    my_robot.set_joint_positions(safe_home_angles, joint_indices=joint_indices)
                    my_robot.set_joint_velocities(np.zeros(6), joint_indices=joint_indices)
                    
            # 【第二阶段：20~50帧】向前探出一步进行系统自检
            elif startup_frames == 20:
                print("✅ 落地稳定！开始执行机械臂开机自检...")
                
            elif 20 < startup_frames <= 50:
                current_pos, _ = ee_solver.compute_end_effector_pose()
                base_down_quat = np.array([0, 0.7071, 0.7071, 0])
                target_pos = current_pos + np.array([1.0, 0, 0]) * 0.005 
                action, _ = ee_solver.compute_inverse_kinematics(target_pos, base_down_quat)
                my_robot.apply_action(action)
                
            # 【第三阶段：50~90帧】向后缩回一步，恢复原位
            elif 50 < startup_frames < 90:
                current_pos, _ = ee_solver.compute_end_effector_pose()
                base_down_quat = np.array([0, 0.7071, 0.7071, 0])
                target_pos = current_pos + np.array([-1.0, 0, 0]) * 0.005
                action, _ = ee_solver.compute_inverse_kinematics(target_pos, base_down_quat)
                my_robot.apply_action(action)
                
            # 【自检完成】VLM 准备接管
            elif startup_frames == 90:
                print("🚀 机械臂自检完毕！VLM 大脑正式接管底盘控制权！")

            startup_frames += 1
            continue
        # =========================================================

        # =========================================================
        # 🧠 VLM 大脑自动驾驶逻辑 (90 帧之后)
        # =========================================================
        if timeline.is_playing():
            if step_counter % VLM_THINKING_INTERVAL == 0:
                rgb_img = cameras.get_head_rgb()
                
                if rgb_img is not None:
                    print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] 🧠 大脑思考中 (帧数: {startup_frames + step_counter})...")
                    
                    # 思考时底盘刹车
                    for prim in wheel_prims:
                        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                        drive.GetTargetVelocityAttr().Set(0.0)
                    
                    # 呼叫大脑，并传递 log 文件夹路径用于保存此时此刻看到的图片
                    v_x, v_yaw = brain.get_action(rgb_img, instruction, save_dir=LOG_DIR, step=startup_frames + step_counter)
                    print(f"🎯 决策下发 -> 前进: {v_x:.2f}, 转向: {v_yaw:.2f}")
                    
                    # 恢复动力
                    if len(wheel_prims) > 0:
                        for prim in wheel_prims:
                            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                            if "left" in prim.GetName().lower(): 
                                drive.GetTargetVelocityAttr().Set(v_x + v_yaw)
                            else: 
                                drive.GetTargetVelocityAttr().Set(v_x - v_yaw)
                else:
                    print("⚠️ 感知数据为空！")
                    
            step_counter += 1

    simulation_app.close()

if __name__ == "__main__":
    run()