# HY-Motion + Genesis 人体实验

这个目录是 HY-Motion 动作在 Genesis 中回放的独立实验模块，不依赖也不修改 `gaze_vr_isaaclab`。

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

当前版本的关节链使用固定骨长和球关节，能作为 Genesis 结构/碰撞代理，但关键点重定向仍是骨段方向拟合。最近一次完整回放的最大关节位置误差约为 0.462 m，主要出现在右脚踝；下一步需要做带末端位置约束的 IK 和脚底接触修正。

## 不包含

本仓库不包含 HY-Motion 模型权重、Python/Conda 环境、Hugging Face 缓存、临时文件和 `gaze_vr_isaaclab` 项目文件。
