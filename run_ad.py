__author__ = "Antoine Richard"
__license__ = "BSD-3-Clause"

import pygame
import numpy as np
from omegaconf import DictConfig, OmegaConf, ListConfig
from src.configurations import configFactory
from src.environments_wrappers import startSim
import logging
import hydra
import carb

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

@hydra.main(config_name="config", config_path="cfg")
def run(cfg: DictConfig):
    cfg = omegaconfToDict(cfg)
    cfg = instantiateConfigs(cfg)
    
    print("\n⏳ 环境启动中，请稍候...\n")
    SM, simulation_app = startSim(cfg)

    logging.getLogger("omni.physx.plugin").setLevel(logging.ERROR)
    carb.settings.get_settings().set_int("/log/level", 2)

    pygame.init()
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("Jackal & UR3 Teleop")

    # ==========================================
    # 动态版本自适应：消除弃用警告
    # ==========================================
    import omni.usd
    from pxr import UsdPhysics
    import omni.timeline
    from omni.isaac.core.articulations import Articulation
    import os
    from omni.isaac.core.utils.extensions import get_extension_path_from_name

    # 1. 获取运动学包的物理安装路径 (兼容最新版 Isaac Sim)
    ext_name = "isaacsim.robot_motion.motion_generation"
    mg_ext_path = get_extension_path_from_name(ext_name)
    if mg_ext_path is None: # 如果是老版本，则退回旧名字
        ext_name = "omni.isaac.motion_generation"
        mg_ext_path = get_extension_path_from_name(ext_name)

    # 2. 动态导入求解器
    if "isaacsim" in ext_name:
        from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
    else:
        from omni.isaac.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
    # ==========================================
    
    stage = omni.usd.get_context().get_stage()
    timeline = omni.timeline.get_timeline_interface()

    print("\n" + "⚔️"*20)
    print("【完美越野 + 手柄遥操作版】开始实施物理级降维打击...")
    
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

    # ==========================================
    # 挂载手柄
    # ==========================================
    pygame.joystick.init()
    gamepad = None
    if pygame.joystick.get_count() > 0:
        gamepad = pygame.joystick.Joystick(0)
        gamepad.init()
        print(f"🎮 成功检测并挂载手柄: {gamepad.get_name()}")
    else:
        print("⚠️ 未检测到手柄！机械臂将处于锁定状态。")

    # ==========================================
    # 初始化机械臂 IK 大脑 (终极暴力寻址版)
    # ==========================================
    print("🤖 正在初始化 UR3e 逆运动学(IK)大脑...")
    robot_path = "/Robots/husky/jackal"
    my_robot = Articulation(robot_path)
    my_robot.initialize()
    print("🤖 正在初始化 UR3e 逆运动学(IK)大脑...")
    robot_path = "/Robots/husky/jackal"
    my_robot = Articulation(robot_path)
    from omni.isaac.core.utils.types import ArticulationAction
    my_robot.initialize()
    
    # ==========================================
    # 🎯 诊断日志写入文件版 + 安全避撞姿态
    # ==========================================
    log_file = "robot_debug_log.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=== 🤖 机器人关节扫描诊断报告 ===\n")
        f.write(f"当前躯体包含的所有电机(共 {len(my_robot.dof_names)} 个)：\n")
        f.write(str(my_robot.dof_names) + "\n\n")
        
        safe_home_angles = np.array([0.0, -1.0, 1.0, -1.0, -1.57, 0.0])
        arm_joint_keywords = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
        
        joint_indices = []
        f.write("--- 🔍 开始模糊匹配 UR3e 关节 ---\n")
        for kw in arm_joint_keywords:
            found = False
            for idx, name in enumerate(my_robot.dof_names):
                if kw in name.lower():
                    joint_indices.append(idx)
                    f.write(f"✅ 找到匹配项: 关键词 '{kw}' -> 实际命名 '{name}' (ID: {idx})\n")
                    found = True
                    break
            if not found:
                f.write(f"❌ 警告: 未找到包含关键词 '{kw}' 的关节！\n")
                
        f.write("\n--- 🚀 执行姿态注入 ---\n")
        if len(joint_indices) == 6:
            try:
                # 1. 物理瞬移，瞬间把姿态掰过去
                my_robot.set_joint_positions(safe_home_angles, joint_indices=joint_indices)
                
                # 2. 构造官方标准动作指令，锁定电机目标
                home_action = ArticulationAction(joint_positions=safe_home_angles, joint_indices=joint_indices)
                my_robot.apply_action(home_action)
                
                f.write("✅ 机械臂【安全避撞姿态】注入并锁定成功！\n")
            except Exception as e:
                f.write(f"⚠️ 姿态设置出错: {e}\n")
        else:
            f.write(f"❌ 致命错误：只找到了 {len(joint_indices)} 个手臂关节，姿态注入失败！\n")

    print(f"\n📂📂📂 诊断完毕！请立刻去 OmniLRS1 目录下查看 {log_file} 文件！📂📂📂\n")
    # ==========================================
    

    # 暴力寻址：直接去本地硬盘里扫盘，把 URDF 和 YAML 翻出来，彻底绕过官方 API
    config_dir = os.path.join(mg_ext_path, "motion_policy_configs")
    urdf_path = ""
    yaml_path = ""

    for root, dirs, files in os.walk(config_dir):
        if "ur3e" in root.lower() or "universal_robots" in root.lower():
            for f in files:
                if f.endswith(".urdf") and "ur3e" in f.lower():
                    urdf_path = os.path.join(root, f)
                if f.endswith(".yaml") and ("descriptor" in f.lower() or "description" in f.lower()) and "ur3e" in root.lower():
                    yaml_path = os.path.join(root, f)

    if not urdf_path or not yaml_path:
        raise FileNotFoundError(f"❌ 扫盘失败！无法在 {config_dir} 找到 UR3e 的配置文件！")

    # 将绝对物理路径直接喂给 Lula，彻底解决 bad file
    ik_solver = LulaKinematicsSolver(
        robot_description_path=yaml_path,
        urdf_path=urdf_path
    )
    
    ee_solver = ArticulationKinematicsSolver(my_robot, ik_solver, "wrist_3_link")
    print("✅ 机械臂 IK 初始化完毕！")

    print("👉 请点击黑窗口，左手 WSAD 飙车，右手摇杆挥舞机械臂！")
    print("⚔️"*20 + "\n")

    # ==========================================
    # 接管主循环
    # ==========================================
    while simulation_app.is_running():
        simulation_app.update()
        if not timeline.is_playing():
            timeline.play()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                simulation_app.close()
                return

        # --- 1. 底盘键盘控制 ---
        keys = pygame.key.get_pressed()
        forward_vel = 0.0
        turn_vel = 0.0

        if keys[pygame.K_w]: forward_vel = 100.0
        if keys[pygame.K_s]: forward_vel = -100.0
        if keys[pygame.K_a]: turn_vel = -45.0
        if keys[pygame.K_d]: turn_vel = 45.0

        if len(wheel_prims) > 0:
            for prim in wheel_prims:
                drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                if "left" in prim.GetName().lower():
                    drive.GetTargetVelocityAttr().Set(forward_vel + turn_vel)
                else:
                    drive.GetTargetVelocityAttr().Set(forward_vel - turn_vel)

        # --- 2. 手柄 IK 控制 (垂直锁定优化版) ---
        if gamepad:
            pygame.event.pump()
            
            # A. 读取十字键 (Hat) 控制 XY
            hat_x, hat_y = 0, 0
            if gamepad.get_numhats() > 0:
                hat_x, hat_y = gamepad.get_hat(0)
                
            # B. 读取右摇杆轴 3 控制 Z 升降
            axis_z = 0.0
            if gamepad.get_numaxes() > 3:
                axis_z = gamepad.get_axis(3)

            deadzone = 0.15
            dx = float(hat_x)
            dy = float(hat_y)
            dz = axis_z if abs(axis_z) > deadzone else 0.0

            if dx != 0 or dy != 0 or dz != 0:
                # 只获取当前位置
                current_pos, _ = ee_solver.compute_end_effector_pose()
                speed = 0.005
                
                # 计算目标位置
                target_pos = current_pos + np.array([dy, -dx, -dz]) * speed
                
                # =======================================================
                # 🎯 核心逻辑：定义“垂直向下”的四元数 (W, X, Y, Z)
                # 这会让夹爪在移动过程中，无论车怎么跑，始终死死指着地心。
                # =======================================================
                vertical_down_quat = np.array([0, 0.7071, 0.7071, 0]) 

                # 打印状态，方便你观察高度
                print(f"🕹️ 锁定垂直移动 | 目标高度 Z: {target_pos[2]:.3f}")

                # 求解 IK：同时传入位置和姿态约束
                action, success = ee_solver.compute_inverse_kinematics(
                    target_position=target_pos,
                    target_orientation=vertical_down_quat
                )
                
                if success:
                    my_robot.apply_action(action)
                else:
                    # 如果因为姿态锁定导致“够不着”（奇异点），可以尝试只给位置
                    # print("⚠️ 姿态锁定下无法到达，尝试自适应调整...")
                    pass
        pygame.display.flip()

    pygame.quit()
    simulation_app.close()

if __name__ == "__main__":
    run()
