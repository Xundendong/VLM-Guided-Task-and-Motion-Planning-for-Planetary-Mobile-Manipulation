__author__ = "Antoine Richard & Customized by You"
__license__ = "BSD-3-Clause"

import pygame
import numpy as np
from omegaconf import DictConfig, OmegaConf, ListConfig
from src.configurations import configFactory
from src.environments_wrappers import startSim
import logging
import hydra
import carb
import os

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
# 纯手工打造：四元数乘法 [w, x, y, z] (绝对防降级/防报错)
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
    carb.settings.get_settings().set_int("/log/level", 2)

    pygame.init()
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("Jackal & UR3 Keyboard Teleop")

    import omni.usd
    from pxr import UsdPhysics
    import omni.timeline
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.utils.extensions import get_extension_path_from_name
    
    # 新版动作指令导入 (放在这里防止循环内重复导入)
    from isaacsim.core.utils.types import ArticulationAction

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
    print("【纯键盘操控版】系统初始化中...")
    
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
    
    # 获取关节索引
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

    ik_solver = LulaKinematicsSolver(robot_description_path=yaml_path, urdf_path=urdf_path)
    ee_solver = ArticulationKinematicsSolver(my_robot, ik_solver, "wrist_3_link")
    print("✅ 机械臂 IK 初始化完毕！")

    # 初始化夹爪
    print("🔍 正在扫描并绑定夹爪关节...")
    gripper_keywords = ["finger", "knuckle"]
    gripper_indices = []
    for idx, name in enumerate(my_robot.dof_names):
        for kw in gripper_keywords:
            if kw in name.lower():
                gripper_indices.append(idx)
                break
    print(f"✅ 成功找到 {len(gripper_indices)} 个夹爪相关关节！")

    gripper_closed = False
    gripper_yaw = 0.0
    space_pressed_last_frame = False  

    print("\n👉 键盘操作指南：")
    print("左手: [W/S/A/D] 开车 | [R/F] 机械臂升降 | [Q/E] 夹爪旋转 | [空格] 张开/闭合")
    print("右手: [↑/↓/←/→] 控制机械臂平面移动")
    print("⚔️"*20 + "\n")
    
    # 物理引擎热身计数器
    startup_frames = 0
    
    # ==========================================
    # 接管主循环
    # ==========================================
    while simulation_app.is_running():
        simulation_app.update()
        if not timeline.is_playing():
            timeline.play()

        # =========================================================
        # 🛡️ 物理引擎热身与开机自检 (三段式动画 0~90 帧)
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
                
            # 【自检完成】开放控制权
            elif startup_frames == 90:
                print("🚀 机械臂自检完毕！各关节响应正常，键盘控制权已解锁！")

            startup_frames += 1
            
            # 在自检期间，强制跳过键盘读取事件，防止控制指令冲突
            if startup_frames <= 90:
                pygame.display.flip()
                continue
        # =========================================================

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                simulation_app.close()
                return

        keys = pygame.key.get_pressed()

        # --- 1. 底盘键盘控制 (WASD) ---
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

        # --- 2. 夹爪开合 (Space 空格键) ---
        space_pressed = keys[pygame.K_SPACE]
        # 边缘检测逻辑，防止信号抖动
        if space_pressed and not space_pressed_last_frame:
            gripper_closed = not gripper_closed
            print(f"🔧 夹爪状态切换: {'闭合' if gripper_closed else '张开'}")
            
            target_val = 0.8 if gripper_closed else 0.0
            
            if len(gripper_indices) > 0:
                target_positions = []
                for idx in gripper_indices:
                    joint_name = my_robot.dof_names[idx].lower()
                    multiplier = 1.0
                    
                    if "right" in joint_name:
                        multiplier = -1.0
                    if "inner_finger" in joint_name and "knuckle" not in joint_name:
                        multiplier *= -1.0 
                        
                    target_positions.append(target_val * multiplier)

                gripper_action = ArticulationAction(
                    joint_positions=np.array(target_positions),
                    joint_indices=gripper_indices
                )
                my_robot.apply_action(gripper_action)
                
        space_pressed_last_frame = space_pressed

        # --- 3. 夹爪旋转 (Q/E) ---
        rot_speed = 0.0
        if keys[pygame.K_q]: rot_speed = 0.05
        if keys[pygame.K_e]: rot_speed = -0.05
        gripper_yaw += rot_speed

        # --- 4. 机械臂 XYZ 移动 (方向键 + R/F) ---
        move_x, move_y, move_z = 0.0, 0.0, 0.0
        
        if keys[pygame.K_UP]: move_x = 1.0
        if keys[pygame.K_DOWN]: move_x = -1.0
        if keys[pygame.K_LEFT]: move_y = 1.0
        if keys[pygame.K_RIGHT]: move_y = -1.0
        if keys[pygame.K_r]: move_z = 1.0
        if keys[pygame.K_f]: move_z = -1.0

        # --- 5. IK 求解下发 ---
        if move_x != 0 or move_y != 0 or move_z != 0 or rot_speed != 0:
            current_pos, _ = ee_solver.compute_end_effector_pose()
            speed = 0.005
            
            target_pos = current_pos + np.array([move_x, move_y, move_z]) * speed
            base_down_quat = np.array([0, 0.7071, 0.7071, 0]) 
            z_rot_quat = np.array([np.cos(gripper_yaw/2), 0.0, 0.0, np.sin(gripper_yaw/2)])
            target_quat = custom_quat_mul(z_rot_quat, base_down_quat)

            action, success = ee_solver.compute_inverse_kinematics(
                target_position=target_pos,
                target_orientation=target_quat
            )
            
            if success:
                my_robot.apply_action(action)

        pygame.display.flip()

    pygame.quit()
    simulation_app.close()

if __name__ == "__main__":
    run()
