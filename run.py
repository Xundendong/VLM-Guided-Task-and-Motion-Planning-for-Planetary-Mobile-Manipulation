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

    # 引擎启动后，切断烦人的 PhysX 警告喇叭
    logging.getLogger("omni.physx.plugin").setLevel(logging.ERROR)
    carb.settings.get_settings().set_int("/log/level", 2)

    pygame.init()
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("Jackal pygame")

    import omni.usd
    from pxr import UsdPhysics
    import omni.timeline
    
    stage = omni.usd.get_context().get_stage()
    timeline = omni.timeline.get_timeline_interface()

    print("\n" + "⚔️"*20)
    print("【完美越野微调版】开始实施物理级降维打击...")
    
    # 拔管：切除官方后台的“幽灵刹车系统”
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

    # 劫持轮子，注入动力
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
    print("👉 请点击黑窗口，按下 W/S/A/D 在月球狂飙！")
    print("⚔️"*20 + "\n")

    # 接管主循环
    while simulation_app.is_running():
        simulation_app.update()
        if not timeline.is_playing():
            timeline.play()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                simulation_app.close()
                return

        # 捕获键盘
        keys = pygame.key.get_pressed()
        forward_vel = 0.0
        turn_vel = 0.0

        # ==========================================
        # 微调 1：提速！基础速度从 30 提到 60，转向从 20 提到 35
        # ==========================================
        if keys[pygame.K_w]: forward_vel = 100.0
        if keys[pygame.K_s]: forward_vel = -100.0
        if keys[pygame.K_a]: turn_vel = -45.0
        if keys[pygame.K_d]: turn_vel = 45.0

        # 往底层写速度
        if len(wheel_prims) > 0:
            for prim in wheel_prims:
                drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                # ==========================================
                # 微调 2：修正转向符号极性！
                # ==========================================
                if "left" in prim.GetName().lower():
                    # 以前是 forward_vel - turn_vel，现在改成 +
                    drive.GetTargetVelocityAttr().Set(forward_vel + turn_vel)
                else:
                    # 以前是 forward_vel + turn_vel，现在改成 -
                    drive.GetTargetVelocityAttr().Set(forward_vel - turn_vel)

        pygame.display.flip()

    pygame.quit()
    simulation_app.close()

if __name__ == "__main__":
    run()