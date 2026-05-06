import numpy as np

class JackalUR3eController:
    def __init__(self, stage, mg_ext_path=""):
        self.stage = stage
        self.robot = None
        self.wheel_prims = []
        self.arm_joint_indices = []
        
        # 👇 核心修复：用回你原版的“防穿模”安全角度！千万别用我之前给的 -1.57
        self.safe_home_angles = np.array([0.0, -1.0, 1.0, -1.0, -1.57, 0.0])

    def initialize(self):
        """初始化底层物理和关节"""
        from omni.isaac.core.articulations import Articulation
        
        self.robot = Articulation("/Robots/husky/jackal")
        self.robot.initialize()
        
        # 1. 绑定动力轮
        self._bind_wheels()
        # 2. 识别机械臂 6 个关节
        self._bind_arm()
        
        # 3. 立即执行一次硬锁定，防止落地瞬间“骨折”
        self.apply_home_pose()
        print("✅ [底层] 硬件驱动已就绪，UR3e 机械臂已锁定收起姿态")

    def _bind_wheels(self):
        from pxr import UsdPhysics
        for prim in self.stage.Traverse():
            if prim.GetTypeName() == "PhysicsRevoluteJoint" and "wheel" in prim.GetName().lower():
                self.wheel_prims.append(prim)
                drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                if not drive: drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive.GetStiffnessAttr().Set(0.0)
                drive.GetDampingAttr().Set(10000000.0)

    def _bind_arm(self):
        """自动识别 UR3e 的 6 个核心旋转关节"""
        keywords = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
        self.arm_joint_indices = []
        for kw in keywords:
            for idx, name in enumerate(self.robot.dof_names):
                if kw in name.lower():
                    self.arm_joint_indices.append(idx)
                    break

    def apply_home_pose(self):
        """强制将机械臂固定在 Home 位置 (物理引擎启动初期必调)"""
        if len(self.arm_joint_indices) == 6:
            self.robot.set_joint_positions(self.safe_home_angles, joint_indices=self.arm_joint_indices)
            self.robot.set_joint_velocities(np.zeros(6), joint_indices=self.arm_joint_indices)

    def set_base_velocity(self, forward_vel, turn_vel):
        from pxr import UsdPhysics
        for prim in self.wheel_prims:
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if "left" in prim.GetName().lower():
                drive.GetTargetVelocityAttr().Set(forward_vel + turn_vel)
            else:
                drive.GetTargetVelocityAttr().Set(forward_vel - turn_vel)