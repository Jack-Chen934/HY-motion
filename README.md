# HY-Motion + Genesis 人体实验

这个目录是 HY-Motion 动作在 Genesis 中回放的独立实验模块，不依赖也不修改 `gaze_vr_isaaclab`。

## 当前进度（2026-09-20）

当前已完成“HY-Motion 动作导入 → Genesis 人体代理 → 运动学回放 → 基础足部接触处理”的验证阶段。最新实现已经提交到本仓库的独立分支历史中，推荐使用 `replay_human_mjcf.py` 的 cubic 插值、足部锁定、渐进 IK 和诊断报告功能。

最近一次完整回放使用 `final_full_cooperation_seed2026.npz`，时长约 4.97 秒，Genesis 1.3.3、tavis 环境、CPU 后端，结果为：

- 源帧重复数：0；
- 脚锚点最大下穿：0 m；
- 父子骨段最大误差：约 `1.5e-7 m`；
- 碰撞步骤后的 qpos 漂移：0；
- 回归测试：3 项通过；
- 左/右脚累计滑移：约 15.8 cm / 9.1 cm；
- 支撑 IK 拒绝：4 次；
- 根部最大 jerk：约 `7649 m/s^3`；
- 关节最大 jerk：约 `43879 rad/s^3`。

最新报告为 `reports/final_smoothness_full.json`。报告包含源轨迹、100 Hz 命令轨迹和最终应用轨迹的速度/加速度/jerk，以及接触切换和 IK 跳变信息。

## 当前问题与目标关系

当前人体仍是固定骨长的 22 关节 MJCF 运动学代理，不是真实动力学人体。`set_qpos()` 直接写入姿态，因此尚未验证人体在重力、摩擦、接触力和关节扭矩下的自然运动。

当前主要限制如下：

- 腿部 IK 会降低脚部穿地，但接触切换仍会增加 jerk，并产生脚滑移；
- 支撑 IK 可能不可达，当前策略会拒绝该次硬约束以保护骨架结构；
- 上身旋转主要由骨段方向重建，绕骨段轴的 twist 不完全确定；
- GLB 木制人体目前主要用于视觉，MJCF 胶囊/球体是独立碰撞代理；
- 尚未加入人体 PD/扭矩控制、真实接触动力学和碰撞响应；
- 尚未接入 `gaze_vr_isaaclab` 中的 Genesis 机器人、协作任务、观测空间、奖励函数或训练接口。

因此，本仓库当前可以用于动作导入、人体代理可视化、骨架连续性和初步足部约束验证；它还不能直接作为最终的人体—机器人协同强化学习环境。下一阶段应先增加人体 PD 跟踪和动力学稳定性验证，再把人体与机器人放入同一个 Genesis 场景，最后建立协作控制和训练接口。

## 内容

- `export_hymotion_motion.py`：将 HY-Motion 的 SMPL-H 输出转换为 Genesis 关键点缓存。
- `replay_human_capsules.py`：使用 22 个关节和胶囊骨段进行运动学回放。
- `generate_human_mjcf.py`：根据动作静态骨架生成 22 关节 MJCF 人体。
- `replay_human_mjcf.py`：将 HY-Motion 关键点重定向为 MJCF 球关节姿态。
- `assets/human_22ball.xml`：22 关节人体和胶囊/球体碰撞几何。
- `load_wooden_human.py`：加载官方木制人体视觉模型，执行坐标、缩放和地面高度检查。
- `assets/wooden_model/boy_Rigging_smplx_tex.glb`：从 HY-Motion 官方 FBX 转换的 GLB 人体资产。
- `motions/prepared/final_full_cooperation_seed2026.npz`：已准备好的示例动作。
- `reports/`：关键回放和模型加载报告。

## 环境

当前验证环境为 Genesis 1.3.3，使用 `tavis` 环境和 CPU 后端。Genesis 运行前建议设置：

```bash
export QD_TMP_DIR=/tmp/hy_genesis_qd
export XDG_CACHE_HOME=/tmp/hy_genesis_cache
export MPLCONFIGDIR=/tmp/hy_genesis_mpl
export NUMBA_CACHE_DIR=/tmp/hy_genesis_numba
```

## 运行人体静态可视化

```bash
/var/local/sorry/conda/envs/tavis/bin/python \
  load_wooden_human.py --viewer
```

## 运行动作回放

```bash
/var/local/sorry/conda/envs/tavis/bin/python \
  replay_human_capsules.py \
  --motion motions/prepared/final_full_cooperation_seed2026.npz \
  --report reports/stage_a_replay.json
```

当前木制人体 GLB 仅作为视觉资产使用；碰撞代理仍是独立的胶囊骨架。真实人体关节链、动力学、PD 跟踪和机器人交互尚未包含在本目录中。

## 22 关节 MJCF 回放

生成 MJCF：

```bash
/var/local/sorry/conda/envs/tavis/bin/python \
  generate_human_mjcf.py
```

无界面回放：

```bash
/var/local/sorry/conda/envs/tavis/bin/python \
  replay_human_mjcf.py \
  --report reports/stage_b_mjcf_replay.json
```

打开 Genesis viewer：

```bash
/var/local/sorry/conda/envs/tavis/bin/python \
  replay_human_mjcf.py --viewer
```

开启碰撞几何进行短时测试：

```bash
/var/local/sorry/conda/envs/tavis/bin/python \
  replay_human_mjcf.py --seconds 0.5 --collision \
  --report reports/stage_b_mjcf_collision_smoke.json
```

开启足底接触检测、根部校正和地面防穿透：

```bash
/var/local/sorry/conda/envs/tavis/bin/python \
  replay_human_mjcf.py \
  --collision --foot-lock \
  --report reports/stage_b_foot_lock.json
```

开启球关节腿部数值 IK：

```bash
/var/local/sorry/conda/envs/tavis/bin/python \
  replay_human_mjcf.py \
  --foot-lock --leg-ik \
  --report reports/stage_b_leg_ik.json
```

平滑回放的推荐命令是：

```bash
export QD_TMP_DIR=/tmp/hy_genesis_qd
export XDG_CACHE_HOME=/tmp/hy_genesis_cache
export MPLCONFIGDIR=/tmp/hy_genesis_mpl
export NUMBA_CACHE_DIR=/tmp/hy_genesis_numba
export NUMBA_DISABLE_CACHING=1

/var/local/sorry/conda/envs/tavis/bin/python \
  replay_human_mjcf.py \
  --foot-lock --leg-ik --ik-weight 0.2 --foot-clearance 0.055 \
  --report reports/final_smoothness_full.json
```

平滑回放包含以下稳定性处理：30 FPS 动作到 100 Hz 仿真的三次位置插值和旋转样条、四元数符号连续化、结合脚部高度和水平速度的接触检测、接触进入/释放确认、支撑腿 warm-start，以及带根部正则化的统一支撑 IK。根部位置和球关节姿态采用时间连续的 cubic 插值，减少源帧边界的速度、加速度和 jerk 突变。双脚同时接触时优先保持已经稳定的支撑脚；若当前固定骨长代理无法达到支撑目标，则拒绝该次 IK 解并回退到连续源姿态，避免把腿拉入错误分支。`--ik-weight 0.2` 让 IK 以渐进方式写回，降低接触切换时的姿态跳变；IK 会携带上一帧根部修正和腿部解，并对根部、关节和摆动脚进行限幅；最后的地面安全钳位只在 IK 不可达时启用。可用下面的命令运行快速回归测试：

```bash
/var/local/sorry/conda/envs/tavis/bin/python -m pytest -q tests/test_replay_smoothness.py
```

最新完整回放报告为 `reports/final_smoothness_full.json`。报告中的
`support_ik_rejections` 表示因为固定骨长代理不可达而主动放弃的硬支撑约束次数；这不是运行失败，而是避免不稳定姿态的保护机制。当前回放仍然是运动学控制，不代表人体已经具备真实动力学接触力。

报告还包含 `smoothness_metrics`：分别给出源 30 FPS、插值后的 100 Hz 命令轨迹和最终应用轨迹的根部/关节速度、加速度与 jerk；`contact_transition_frames` 用于检查接触切换，`pre_post_ik_jump_metrics` 用于定位 IK 造成的局部跳变。当前最终回归的完整动作约 4.97 秒，源帧重复为 0，脚锚点最大下穿为 0，父子骨长误差约 `1.5e-7 m`，碰撞后的 qpos 漂移为 0。

足底逻辑是运动学约束：从脚部关键点检测接触，接触期间锁定脚的 XY，并读取 Genesis 的真实脚关节锚点修正根部位置。`--leg-ik` 使用独立的阻尼最小二乘数值 IK，针对当前 MJCF 的髋/膝/踝球关节计算脚锚点 Jacobian；摆动脚不加入支撑 IK，避免破坏行走腿的连续性。当前策略以骨架结构完整性优先：所有父子骨段长度每帧检查，腿部球关节有跨帧旋转限幅。它不是动力学接触求解，也不是 PD 控制。报告会记录骨段长度误差、接触帧数、最大穿地量、脚滑移距离、IK 误差和根部校正量。

推荐不要在运动学回放中使用 `--collision`。如果误加该选项，代码会在每步后重新写回目标 `qpos`，以防碰撞求解器改写姿态造成视觉上的关节链散开；这只能保护可视化结构，不能产生真实接触力。需要真实碰撞和接触动力学时，应改用 PD/扭矩控制器。

当前版本的关节链使用固定骨长和球关节，能作为 Genesis 结构/碰撞代理，但上身关键点重定向仍是骨段方向拟合。启用腿部 IK 后，接触脚可保持稳定，下一步应进行动力学 PD 跟踪和真实机器人/人体交互验证。

## 不包含

本仓库不包含 HY-Motion 模型权重、Python/Conda 环境、Hugging Face 缓存、临时文件和 `gaze_vr_isaaclab` 项目文件。
