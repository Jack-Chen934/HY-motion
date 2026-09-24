# HY-Motion + Genesis 人体实验

这个目录是 HY-Motion 动作在 Genesis 中回放的独立实验模块，不依赖也不修改 `gaze_vr_isaaclab`。

## 当前进度（2026-09-22）

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
- `render_hymotion_mesh.py`：使用官方 52 关节 WoodenMesh 线性蒙皮，生成完整人体网格动作 MP4/GIF。
- `replay_hymotion_mesh_genesis.py`：在 Genesis 中实时更新完整 WoodenMesh，并同步显示 22 关节 MJCF 碰撞代理。
- `CONTEXT.md`：记录人体视觉层、碰撞代理、交互窗口和任务成功等领域术语。
- `docs/hri_task_design.md`：人机交互基准任务、状态机、指标和实施边界。
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
export NUMBA_DISABLE_CACHING=1
```

## 运行人体静态可视化

```bash
/var/local/sorry/conda/envs/tavis/bin/python \
  load_wooden_human.py --viewer
```

注意：`load_wooden_human.py` 当前只加载 GLB 木制网格并进行静态展示，不会播放 HY-Motion 动作，也不会驱动 GLB 的骨骼蒙皮。

## 完整人体网格动作展示

当前推荐先使用官方 HY-Motion `WoodenMesh` 离线渲染路径验证网格动作。它直接读取原始
SMPL-H 的 `poses/trans`，使用 52 个关节和官方线性蒙皮数据生成 24,256 个顶点、
48,396 个三角面，因此不会经过 22 关节 MJCF 重定向，也不会受到 MJCF 固定骨长的影响。
相机沿人体根部跟随，适合查看完整行走。

```bash
cd /home/sorry/python_ws/hy-motion-workspace/genesis-human-experiment

export MPLCONFIGDIR=/tmp/hy_genesis_mpl

/var/local/sorry/conda/envs/tavis/bin/python \
  render_hymotion_mesh.py \
  --output-dir /home/sorry/python_ws/hy-motion-workspace/outputs/final \
  --prefix final_full_cooperation_seed2026_mesh
```

输出文件为：

- `outputs/final/final_full_cooperation_seed2026_mesh.mp4`：30 FPS 网格动作视频；
- `outputs/final/final_full_cooperation_seed2026_mesh.gif`：快速预览；
- `outputs/final/final_full_cooperation_seed2026_mesh_first_frame.png`：首帧；
- `outputs/final/final_full_cooperation_seed2026_mesh_report.json`：顶点、面数和坐标转换报告。

这个脚本是离线视觉基准，不会把网格作为 Genesis 的动力学碰撞体。Genesis 实时
回放版本见下方“Genesis 中的完整网格 + 动作回放”，其中继续保留
`human_22ball.xml` 作为碰撞代理，把蒙皮网格作为视觉层；这样可以避免高面数人体
网格直接参与碰撞导致仿真变慢或接触不稳定。

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

`replay_human_mjcf.py` 会读取默认的
`motions/prepared/final_full_cooperation_seed2026.npz`，因此这里播放的确实是
HY-Motion 生成的动作；但当前画面主体是 MJCF 胶囊/球体人体，而不是
`assets/wooden_model/boy_Rigging_smplx_tex.glb` 网格。相机默认跟随人体根部，适合查看完整行走；如需固定世界坐标相机，可增加 `--fixed-camera`。

完整行走可视化示例：

```bash
export QD_TMP_DIR=/tmp/hy_genesis_qd
export XDG_CACHE_HOME=/tmp/hy_genesis_cache
export MPLCONFIGDIR=/tmp/hy_genesis_mpl
export NUMBA_CACHE_DIR=/tmp/hy_genesis_numba
export NUMBA_DISABLE_CACHING=1

/var/local/sorry/conda/envs/tavis/bin/python \
  replay_human_mjcf.py \
  --viewer \
  --foot-lock \
  --foot-clearance 0.055 \
  --report reports/viewer_follow_walk.json
```

## Genesis 中的完整网格 + 动作回放

现在已经提供实时 Genesis 回放路径。它使用官方 `WoodenMesh` 根据每帧
`poses/trans` 生成 24,256 个网格顶点，然后通过 Genesis 1.3.3 的
`RigidEntity.set_vverts()` 更新视觉网格；同时保留 `human_22ball.xml` 作为
22 关节胶囊/球体代理。网格实体明确设置为 `collision=False`，因此不会让
4.8 万个三角面直接参与碰撞。

先运行 0.5 秒无界面冒烟测试：

```bash
export QD_TMP_DIR=/tmp/hy_genesis_qd
export XDG_CACHE_HOME=/tmp/hy_genesis_cache
export MPLCONFIGDIR=/tmp/hy_genesis_mpl
export NUMBA_CACHE_DIR=/tmp/hy_genesis_numba
export NUMBA_DISABLE_CACHING=1

/var/local/sorry/conda/envs/tavis/bin/python \
  replay_hymotion_mesh_genesis.py \
  --seconds 0.5 --collision \
  --report reports/hymotion_mesh_genesis_smoke.json
```

打开 Genesis viewer 进行完整行走展示：

```bash
/var/local/sorry/conda/envs/tavis/bin/python \
  replay_hymotion_mesh_genesis.py \
  --viewer --collision \
  --report reports/hymotion_mesh_genesis_full.json
```

脚本首次运行会在 `assets/generated/hymotion_mesh_initial.obj` 写入一份首帧
OBJ，仅用于 Genesis 分配视觉顶点缓冲；它属于运行时缓存，已加入 `.gitignore`。
报告会记录顶点/三角面数、顶点顺序校验、最大更新步长和网格—代理根部对齐误差。
当前验证结果为：Genesis 1.3.3、24,256 顶点、48,396 三角面、完整 4.97 秒
和 497 个仿真步成功，网格顶点顺序校验通过，根部对齐误差为 0。

这一步解决的是“完整人体网格在 Genesis 中跟随 HY-Motion 动作显示”。它仍然
不是人体动力学：网格是视觉层，MJCF 是运动学碰撞代理，尚未实现人体 PD/扭矩
 控制、真实接触响应，也尚未把人体和 `gaze_vr_isaaclab` 的机器人放进同一场景。

## 人机交互基准任务

当前默认任务是 `handover`：人行走并伸手递出一个可见物体，机器人在安全距离外
预抓取、低速接近、闭合夹爪、验证抓取、接管物体并撤回。旧的空手接近任务仍可用
`--profile strict` 作为基线。完整设计见 `docs/hri_task_design.md` 和
`docs/adr/0001-redesign-handover-task.md`。四阶段升级路线见
`docs/handover_four_phase_plan.md`。

当前动作分析得到左手腕约 `3.9~5.0 s` 的连续低速窗口，第一版保持要求为 `0.5 s`。
`handover` 目前是“运动学人体 + 分阶段物体控制”的可重复基线：交接前物体跟随
人体掌心局部坐标系，验证抓取后跟随机器人 TCP。它不是人体动力学接触，也不修改
`gaze_vr_isaaclab`。默认预抓取和慢速接近时长为 `16 s + 14 s`，完整任务约
`36.7 s`（包含人体稳定、夹爪闭合、验证和撤回阶段）。这是根据实际 TCP 收敛时间去掉旧版多余等待后的时序；不能再缩短到
机械臂尚未到达物体时就闭合夹爪。

### Handover 可视化

在有图形显示的本地终端运行。viewer 模式使用至少 30 Hz 的视觉步长；CPU 软件
渲染仍可能低于实时，但不会再按 100 Hz 全网格更新拖慢整个画面。交接物体挂在
拇指根部和食指根部之间、进一步向掌心核心偏移的位置，不再直接挂在腕关节；物体尺寸
当前为 `0.04 m`，姿态随掌面更新。viewer 模式会自动降低 IK 求解预算，并把接触查询
限制在安全检查频率；要获得最流畅的完整画面，仍推荐先保存轨迹再使用 `hri-replay`。

```bash
export QD_TMP_DIR=/tmp/hy_genesis_qd
export XDG_CACHE_HOME=/tmp/hy_genesis_cache
export MPLCONFIGDIR=/tmp/hy_genesis_mpl
export NUMBA_CACHE_DIR=/tmp/hy_genesis_numba
export NUMBA_DISABLE_CACHING=1

/var/local/sorry/conda/envs/tavis/bin/python \
  run_hri_walking_approach.py \
  --profile handover \
  --viewer \
  --camera-mode fixed \
  --viewer-hz 30 \
  --backend cpu \
  --ik-max-samples 4 \
  --ik-max-solver-iters 30 \
  --report reports/handover_viewer.json
```

在 CPU 软件渲染下，完整 `handover` viewer 仍可能只有约 14--16 FPS。推荐把任务
验收和画面展示分成两步：先无界面计算并保存严格轨迹，再用 `hri-replay` 回放。回放
不重新执行 IK、碰撞查询或控制器，只显示同一条严格轨迹，并同步显示掌心物体：

```bash
# 1. 严格验收并保存完整轨迹
/var/local/sorry/conda/envs/tavis/bin/python \
  run_hri_walking_approach.py \
  --profile handover \
  --backend cpu \
  --save-trajectory motions/cache/handover_full_20260922.npz \
  --report reports/handover_full_20260922.json

# 2. 流畅视觉回放
/var/local/sorry/conda/envs/tavis/bin/python \
  run_hri_walking_approach.py \
  --profile hri-replay \
  --trajectory motions/cache/handover_full_20260922.npz \
  --viewer \
  --camera-mode fixed \
  --viewer-hz 30 \
  --backend auto \
  --report reports/handover_full_20260922_replay.json
```

不要用 `--seconds 5` 期待看到交接：5 秒只覆盖人体动作和预抓取开始阶段。脚本会
明确打印完整任务所需时长，并在报告中写入 `truncated_before_full_handover: true`。
要看到夹爪闭合、交接和撤回，省略 `--seconds`，或至少运行到约 `36.7 s`。

无界面严格验收不会更新 24,256 顶点视觉网格，因此不会把网格写入性能瓶颈；这不
改变人体碰撞代理、机器人 IK、抓取验证或安全距离检查。`hri-replay` 的画面不是
新的安全验收，安全结论必须读取前一步的严格报告。

第一版执行器为 `run_hri_walking_approach.py`。它在同一个 Genesis scene 中加载
完整 WoodenMesh 视觉层、`human_22ball.xml` 碰撞代理和 R1 Pro URDF，仅从
`gaze_vr_isaaclab` 读取机器人资产，不修改该项目：

```bash
export QD_TMP_DIR=/tmp/hy_genesis_qd
export XDG_CACHE_HOME=/tmp/hy_genesis_cache
export MPLCONFIGDIR=/tmp/hy_genesis_mpl
export NUMBA_CACHE_DIR=/tmp/hy_genesis_numba
export NUMBA_DISABLE_CACHING=1

/var/local/sorry/conda/envs/tavis/bin/python \
  run_hri_walking_approach.py \
  --report reports/hri_walking_approach.json
```

查看器模式使用固定世界相机，不再默认跟随机器人；如确实需要跟随相机，显式增加
`--camera-mode follow`。`--backend auto` 会优先使用 CUDA，否则回退到 CPU；也可以
显式使用 `--backend gpu` 或 `--backend cpu`。

执行器现在提供四种用途明确分开的模式：

- `strict`（默认）用于 HRI 验收：保留 `0.01 s`、100 Hz 仿真和高预算 IK，报告中的
  碰撞、保持时间和末端误差可作为任务指标。CPU 后端可能低于实时。
- `smooth-viewer` 用于看动作：默认把展示和 Genesis 步进统一到至少 30 Hz，固定
  机器人底盘并降低 IK 预算，避免 CPU 上为不可见的 100 Hz 中间步付出成本。它是
  可视化配置，不替代严格任务验收，较大的仿真步长也不适合动力学结论。
- `human-viewer` 只加载人体碰撞代理和完整 WoodenMesh，不加载 R1 Pro；不执行 IK、
  人机距离检测或接触查询。这是最流畅的完整人体动作展示模式。
- `hri-viewer` 加载 R1 Pro 的固定中立视觉姿态，但关闭碰撞求解、IK、接触查询和
  安全检查，仅用于观察人与机器人空间布局。它不是 HRI 成功判定。
- `hri-replay` 读取 `strict` 已经算好的人体和机器人 qpos 轨迹缓存，只做视觉回放，
  不重新执行 IK、碰撞或安全检查。它适合在 CPU 上流畅查看完整人机动作，但不替代
  `strict` 验收。

因此当前共有四种用途不同的运行模式。严格模式负责计算和验收，回放模式负责把
同一份严格结果稳定地展示出来，避免为了看画面重复付出 IK 和碰撞查询成本。

推荐的可视化命令：

```bash
/var/local/sorry/conda/envs/tavis/bin/python \
  run_hri_walking_approach.py \
  --viewer --profile smooth-viewer --camera-mode fixed \
  --viewer-hz 30 --backend auto \
  --report reports/hri_walking_approach_smooth_viewer.json
```

只查看完整人体动作，推荐使用：

```bash
/var/local/sorry/conda/envs/tavis/bin/python \
  run_hri_walking_approach.py \
  --viewer --profile human-viewer --backend auto \
  --report reports/human_viewer.json
```

查看完整人体和固定中立姿态的 R1 Pro，使用：

```bash
/var/local/sorry/conda/envs/tavis/bin/python \
  run_hri_walking_approach.py \
  --viewer --profile hri-viewer --camera-mode fixed --backend auto \
  --report reports/hri_viewer.json
```

## 严格轨迹缓存与流畅回放

严格模式可以把每个仿真步的人体和机器人状态保存为压缩 NPZ。该缓存是运行时
产物，默认放在 `motions/cache/`，已由 `.gitignore` 忽略，不会混入仓库版本历史。
它只用于视觉回放，不改变严格模式的计算结果：

```bash
# 先执行一次真实 HRI 计算并保存轨迹
/var/local/sorry/conda/envs/tavis/bin/python \
  run_hri_walking_approach.py \
  --profile strict \
  --backend cpu \
  --save-trajectory motions/cache/hri_episode_20260922.npz \
  --report reports/hri_episode_20260922_strict.json

# 再回放已经计算好的完整人机动作
/var/local/sorry/conda/envs/tavis/bin/python \
  run_hri_walking_approach.py \
  --profile hri-replay \
  --trajectory motions/cache/hri_episode_20260922.npz \
  --viewer \
  --camera-mode fixed \
  --viewer-hz 30 \
  --backend auto \
  --report reports/hri_episode_20260922_replay.json
```

本次实际回放结果：动作时长 `4.9667 s`，回放步数 `150`，CPU 墙钟耗时约
`0.582 s`，实时倍率约 `8.54x`；人体网格更新 `150` 次，机器人状态更新 `150`
次，`ik_calls=0`、`safety_checks=0`、`collision_enabled=false`。这说明该模式
适合流畅展示，但安全指标必须引用对应的 `strict` 报告，不能把回放报告当作新的
安全验收。

如果只需要最流畅的人体动作，不需要机器人，继续使用 `human-viewer`；如果需要
验证严格 HRI 指标，继续使用 `strict`。当前 tavis 环境检测到 GPU 不可用，因此
上述实测使用 Genesis CPU 后端。

其中 `human-viewer` 与 `hri-viewer` 的报告会明确记录 `ik_calls: 0`、
`safety_checks: 0` 或 `collision_enabled: false`。要验证机器人真实跟随人体手部、
碰撞安全和保持时间，必须回到 `strict` 或性能较低但仍带控制逻辑的 `smooth-viewer`。

如果需要严格验收，使用：

```bash
/var/local/sorry/conda/envs/tavis/bin/python \
  run_hri_walking_approach.py \
  --viewer --profile strict --camera-mode fixed --backend gpu \
  --report reports/hri_walking_approach_strict_viewer.json
```

`--backend gpu` 只有在当前 Python 环境的 `torch.cuda.is_available()` 为 `True` 且
NVIDIA 驱动可通信时才可用。当前检查到的 `tavis` 环境是 CPU 后端；Genesis 日志
还报告 software rendering，因此即使没有异常，完整 R1 Pro 场景也可能低于实时。
这种情况下，最稳定的动作展示仍是前文的 `render_hymotion_mesh.py` 离线视频；
Genesis viewer 用于交互场景观察，不能把 CPU 低于实时解释为 HY-Motion 动作本身
不连续。

本次性能修复后的完整报告为
`reports/hri_walking_approach_optimized_final_v2.json`：任务状态 `success`，动作
时长 `4.9667 s`，任务段墙钟耗时 `30.71 s`，暂停窗口末端最大误差 `0.0863 m`，
碰撞步骤 `0`。当前 tavis 环境中 `torch.cuda.is_available()` 为 `False`，因此实际
使用的是 CPU；日志中的 Genesis 软件渲染也会使 viewer 低于实时。若服务器的
CUDA 环境可用，推荐显式验证：

```bash
/var/local/sorry/conda/envs/tavis/bin/python \
  run_hri_walking_approach.py \
  --viewer --camera-mode fixed --backend gpu \
  --report reports/hri_walking_approach_viewer_gpu.json
```

已完成完整 `4.9667 s` 回放验证：状态为 `success`，左手暂停窗口内末端最大
误差 `0.0863 m`，暂停窗口 IK 致命失败 `0`，实际接触步数 `0`，保持时间
`1.08 s`。报告中的 `post_hold_ik_failures` 是保持阶段求解器残差，实际末端
仍满足误差阈值，因此不作为任务失败条件。`min_human_robot_aabb_distance_m`
仅是保守 broad-phase 诊断值；靠近手部时 AABB 可能重叠，不能等同于精确表面
距离，安全判定以代理实际接触查询和末端误差为准。

报告路径：`reports/hri_walking_approach.json`。

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
