结论先说清楚：**你现在应该做的不是“发布一个新 benchmark”，而是做一个固定、可复现、可诊断的 MetaDrive-Parking 数据与评估协议。** 论文里不要把它包装成数据集贡献，而是写成：

> 为了系统研究视觉世界模型在自动泊车中的闭环规划能力，我们构建了一个基于 MetaDrive 的受控停车评估协议，包含固定 train/val/test seeds、ID/OOD splits、专家/扰动/恢复轨迹，以及与 E2E Parking 一致的停车成功率和最终误差指标。

这就足够了。

你现在 retrieval-guided MPPI 已经证明“纯随机 MPPI/CEM 搜索效率差”这个判断是对的；接下来最重要的是让训练数据覆盖 planner 会走到的状态分布，而不是继续在优化器上硬拧。

---

# 0. 我参考了哪些论文/benchmark，分别借鉴什么

你这个数据集设计不应该自嗨，应该明确站在这些工作上：

| 参考                      | 你该借鉴什么                                                                                                                                                                       |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **DINO-WM**             | 离线轨迹训练世界模型 + 测试时 action sequence optimization 的设定。DINO-WM 本身就是从 offline behavioral trajectories 学 latent dynamics，再用 action optimization 做 visual goal reaching。([arXiv][1]) |
| **E2E Parking Dataset** | 自动泊车指标、成功判定、测试任务数量、数据集迭代思路。它用 CARLA 建停车数据，并报告 TSR、APE、AOE 等指标。([arXiv][2])                                                                                                   |
| **MetaDrive**           | 用 procedural generation 和 held-out seeds 做泛化评估。MetaDrive 文档里明确给了 train/test 不同 seed range 的范式。([metadrive-simulator.readthedocs.io][3])                                      |
| **Bench2Drive**         | 闭环评估要按场景能力拆分，不能只看 open-loop L2；它强调 open-loop 指标无法充分反映驾驶性能，并按 scenario/weather/location 做系统评估。([arXiv][4])                                                                    |
| **SEG-Parking**         | 自动泊车可以构造专门停车数据集，并且可以用 expert policy + injected noise 收集数据；它还强调 OOD action 和泛化问题。([arXiv][5])                                                                                 |
| **ParkPredict+ / DLP**  | 停车场不是普通道路，存在非结构化规则、复杂停车 maneuver、缺少公开连续轨迹数据等问题；DLP 可以作为真实停车场 open-loop 补充。([arXiv][6])                                                                                       |

---

# 1. 你这个数据集的核心目的

你现在的数据集不是为了“训练一个会模仿 expert 的模型”，而是为了训练一个**在 planner 可能访问到的状态上也可靠的 world model**。

所以数据集要覆盖三类状态：

| 类型                    | 作用                     |
| --------------------- | ---------------------- |
| Expert success        | 告诉 WM 正确停车流形是什么        |
| Noisy expert          | 覆盖“接近正确但有偏差”的状态        |
| Recovery / off-policy | 覆盖 planner 走歪以后如何恢复的状态 |

这和 SEG-Parking 用 expert policy 加 injected noise 构建停车数据集的思路是一致的，只是你不是做 offline RL policy，而是做 world model + planning。([arXiv][5])

你现在的核心假设可以写成论文里的一个 observation：

> Expert-only offline trajectories are insufficient for world-model-based parking planning, because sampling-based planners frequently induce off-expert states. We therefore augment expert demonstrations with noisy and recovery trajectories to improve dynamics coverage around planner-induced deviations.

中文就是：

> 只用干净专家轨迹会让 WM 只在正确轨迹附近准，而 MPPI/CEM 实际会把车带到偏离状态。为了让 WM 在这些状态上也可靠，需要加入 noisy expert 和 recovery 数据。

---

# 2. 数据集总体结构：MetaDrive-Parking-v2

我建议你让 Codex 按这个名字做：

```text
MetaDrive-Parking-v2
```

它分成 4 个部分：

```text
1. train
   用于训练 world model 和构建 retrieval bank

2. val
   用于选 horizon、subgoal 数、retrieval 参数、MPPI 参数

3. test_id
   同分布闭环测试，不参与训练、不参与 retrieval

4. test_ood
   泛化测试，包括 layout / init pose / dynamics / visual / obstacle OOD
```

注意：**retrieval bank 只能从 train 里建，绝对不能用 val/test。**

---

# 3. 规模设计：先做中等规模，不要一开始 5000+

你现在最适合做两个版本。

## 版本 A：MVP 数据集，用来快速验证

```text
train: 1500 episodes
  - 750 expert success
  - 450 noisy expert
  - 300 recovery / off-policy

val: 200 episodes

test_id: 384 episodes

test_ood:
  - unseen_layout: 128 episodes
  - unseen_init_pose: 128 episodes
  - unseen_dynamics: 128 episodes
  - unseen_visual: 128 episodes
```

为什么 test_id 用 384？因为 E2E Parking Dataset 的测试协议里，一个 test epoch 是 96 个 parking tasks，最终对 best model 做 4 个 test epochs，也就是 384 tasks 来保证稳定性。你可以借鉴这个规模，不必完全一样，但 384 是一个很好解释的数字。([arXiv][2])

## 版本 B：正式训练数据集

如果 MVP 证明有效，再扩到：

```text
train: 5000 episodes
  - 2500 expert success
  - 1500 noisy expert
  - 1000 recovery / off-policy

val: 500 episodes

test_id: 384 或 500 episodes

test_ood:
  - unseen_layout: 200 episodes
  - unseen_init_pose: 200 episodes
  - unseen_dynamics: 200 episodes
  - unseen_visual: 200 episodes
  - obstacle_density: 200 episodes，可选
```

不要一开始就做 10000 条。你的首要目标是验证：**Expert-only → Expert+Noisy → Expert+Noisy+Recovery** 是否明显提升 WM rollout 和 planner success。

---

# 4. split 设计：一定要按 seed / scenario 分，而不是随机切 trajectory

这一点很重要。不能把同一个停车场、同一个车位、同一个初始姿态附近的 episode 随机分到 train 和 test。那样 test 会虚高。

MetaDrive 官方文档给的范式是：用不同 `start_seed` 和 `num_scenarios` 指定 train 和 test，例如 train 用 seeds `[1000,1999]`，test 用 `[0,199]`，从而评估 held-out scenarios 的泛化。([metadrive-simulator.readthedocs.io][3])

你可以让 Codex 固定下面这些 seed range：

```text
train scenario_seed:   10000 - 11999
val scenario_seed:     20000 - 20299
test_id scenario_seed: 30000 - 30383

ood_layout_seed:       40000 - 40199
ood_init_seed:         41000 - 41199
ood_dynamics_seed:     42000 - 42199
ood_visual_seed:       43000 - 43199
ood_obstacle_seed:     44000 - 44199
```

每个 episode 还要有一个 `route_seed`，用于控制初始姿态、目标车位、噪声模式：

```text
scenario_seed: 控制停车场 layout、停放车辆、视觉风格
route_seed: 控制 ego 初始位置、目标车位、专家轨迹扰动
noise_seed: 控制 noisy/recovery 扰动
```

Codex 需要输出固定 split 文件：

```text
data/metadrive_parking_v2/splits/train.json
data/metadrive_parking_v2/splits/val.json
data/metadrive_parking_v2/splits/test_id.json
data/metadrive_parking_v2/splits/test_ood_layout.json
data/metadrive_parking_v2/splits/test_ood_init.json
data/metadrive_parking_v2/splits/test_ood_dynamics.json
data/metadrive_parking_v2/splits/test_ood_visual.json
```

每个 split json 里存的是 task 列表，而不是已经跑出来的 episode 文件。

---

# 5. 场景设计：先别做太多停车类型

你第一版不要同时做垂直车位、斜列车位、平行泊车、动态车、多层车库。会失控。

我建议第一版只做：

```text
主任务：垂直车位 reverse parking / bay parking
可选：少量 forward parking
暂不做：parallel parking、moving agents、复杂交互
```

原因是你的研究核心是 world model planning，不是把所有停车类型都覆盖。E2E Parking / E2E Parking Dataset 也主要是在 CARLA 里围绕目标车位做受控闭环停车任务，并不是一上来覆盖所有真实停车类型。([arXiv][2])

---

# 6. 难度分级：必须有 easy / medium / hard

你需要让数据和测试都能分难度，否则结果很难解释。

## Easy

```text
aisle_width: 6.5m - 7.5m
neighbor_occupancy: 0 或 1 侧有车
start_distance_to_slot: 6m - 10m
initial_yaw_error: 0° - 20°
target_slot: 非边缘车位
obstacle_density: low
```

目的：验证模型是否能完成基本停车。

## Medium

```text
aisle_width: 5.2m - 6.5m
neighbor_occupancy: 两侧随机有车
start_distance_to_slot: 8m - 16m
initial_yaw_error: 15° - 45°
target_slot: 普通车位 + 少量近边缘车位
obstacle_density: medium
```

目的：主实验难度。

## Hard

```text
aisle_width: 4.5m - 5.2m
neighbor_occupancy: 两侧大概率有车
start_distance_to_slot: 12m - 20m
initial_yaw_error: 35° - 70°
target_slot: 边缘车位、窄通道车位、遮挡车位
obstacle_density: high
```

目的：泛化和失败分析。

训练集比例建议：

```text
easy: 30%
medium: 50%
hard: 20%
```

test_id 也用这个比例。test_ood 可以更偏 hard。

---

# 7. 初始位置设计：借鉴 E2E Parking Dataset 的“固定位置族”

E2E Parking Dataset 不是完全随机初始位置，它会设计不同初始位置，并且训练/验证有不同分布。例如 Gen 1A 里每个车位采 8 条 route，其中 6 个训练初始位置包括 far-left、middle-left、near-left、near-right、middle-right、far-right，验证初始位置则相对随机但约束在左右两侧。([arXiv][2])

你可以照这个思想做成 6 个 start families：

```text
start_family:
  0: far_left
  1: mid_left
  2: near_left
  3: near_right
  4: mid_right
  5: far_right
```

每个 family 在目标车位局部坐标系下定义：

```text
relative_to_target_slot:
  longitudinal_offset_along_aisle
  lateral_offset_from_slot
  yaw_error
```

建议范围：

```text
far_left / far_right:
  distance: 14m - 20m
  yaw_error: 20° - 60°

mid_left / mid_right:
  distance: 8m - 14m
  yaw_error: 10° - 45°

near_left / near_right:
  distance: 4m - 8m
  yaw_error: 0° - 30°
```

这样做有两个好处：

1. 你可以均衡覆盖不同起点；
2. test 可以明确说“每个目标车位评估多个初始位置”，更像标准协议。

---

# 8. 三类训练轨迹怎么采

这是最关键的部分。

## 8.1 Expert success trajectories

目标：给 WM 正确停车流形，给 retrieval bank 提供高质量 maneuver。

采集方式：

```text
for each task:
  reset env with scenario_seed, route_seed
  run expert/controller
  record obs, state, action, target_slot, goal_pose
  if success and no collision:
      save as expert_success
  else:
      discard or save separately as expert_failed_debug
```

过滤条件：

```text
success == True
collision == False
outbound == False
final lateral error <= 0.6m
final longitudinal error <= 1.0m
final yaw error <= 10°
```

这个成功阈值直接借鉴 E2E Parking Dataset：车中心相对目标车位中心横向 0.6m、纵向 1m 以内，姿态误差不超过 10°。([arXiv][2])

注意：expert set 只保留成功轨迹。失败 expert 不要混进主训练，除非单独作为 debug 或 recovery source。

---

## 8.2 Noisy expert trajectories

目标：覆盖 expert tube 附近的偏移状态，让 WM 学会“动作稍微不准时车会怎么走”。

采集方式：

```text
a_noisy = clip(a_expert + noise, action_low, action_high)
```

噪声不要太大。建议三档：

```text
small noise: 70%
  steer_std = 0.05
  throttle_std = 0.03

medium noise: 25%
  steer_std = 0.10
  throttle_std = 0.05

large noise: 5%
  steer_std = 0.18
  throttle_std = 0.08
```

如果你的 action 是 `[steer, throttle]`，且 throttle 正负表示前进/倒车，就直接对 throttle 加噪声。如果你有 gear/reverse，建议单独存 `gear`，不要只靠 throttle 正负隐式表示。

执行策略建议用两种：

### A. 全程 noisy expert

```text
每一步执行 expert_action + noise
直到成功 / 碰撞 / 超时
```

这类数据可能会失败，但对 WM 仍然有价值，因为 WM 需要学偏离后的动态。

### B. Noisy + expert correction

```text
前 k 步执行 noisy action
偏离后切回 expert/controller recovery
```

这类数据对 retrieval 和 recovery 都有价值，因为它包含“偏了以后怎么拉回来”。

保存标签：

```text
source_type = "noisy_expert"
noise_level = small / medium / large
is_recovered = True / False
final_success = True / False
```

Noisy 数据可以包含失败，但比例不能太高。建议控制：

```text
noisy trajectories 中成功/恢复成功比例 >= 60%
碰撞样本 <= 10%
```

否则模型会学到太多无意义碰撞状态。

---

## 8.3 Recovery / off-policy trajectories

目标：覆盖 planner 最常进入的偏离状态。

这类数据最重要，也最容易决定你后面 retrieval-guided MPPI 能不能真正成功。

Recovery source 分三种。

### Source A：专家轨迹中间状态扰动

从 expert trajectory 的中间帧采样状态，然后加扰动：

```text
position perturbation:
  lateral: ±0.3m, ±0.6m, ±1.0m
  longitudinal: ±0.5m, ±1.0m, ±2.0m

yaw perturbation:
  ±5°, ±10°, ±20°

velocity perturbation:
  0 - 2 m/s

steer perturbation:
  ±0.1, ±0.3, ±0.5
```

然后让 expert/controller 从这个状态恢复到目标车位。

这类样本最接近你说的“第 1 阶段偏一点，第 2 阶段开始分布外”。

### Source B：错误动作前缀

从正常起点开始，故意执行 3–10 步错误动作：

```text
wrong_action_type:
  opposite_steer
  too_much_steer
  wrong_gear_or_wrong_throttle_sign
  stop_and_go
  oscillating_steer
```

然后切回 expert recovery。

这能覆盖 MPPI 常见的乱动、原地抖、方向打反。

### Source C：当前 planner 失败日志

你已经有 retrieval-guided MPPI 的失败 case。下一版数据集应该把这些失败状态收集起来：

```text
run current planner on train-only tasks
when failure / stage drift occurs:
  save failure state
  reset env to that state
  run expert/controller recovery
  save recovery trajectory
```

注意：只能从 train split 的任务里挖失败状态，不能从 test 里挖。否则 test 泄漏。

保存标签：

```text
source_type = "recovery"
recovery_source = expert_perturb / wrong_prefix / planner_failure
perturbation_level = small / medium / large
recovered_success = True / False
```

---

# 9. 数据文件应该怎么存

Codex 应该统一保存两层文件：task 文件和 episode 文件。

## 9.1 task json

每个 task 是一个可复现任务定义：

```text
{
  "task_id": "train_000123",
  "split": "train",
  "scenario_seed": 10123,
  "route_seed": 50123,
  "noise_seed": 90123,

  "parking_type": "perpendicular_reverse",
  "difficulty": "medium",
  "start_family": "mid_left",

  "layout_params": {
    "num_rows": 2,
    "slots_per_row": 10,
    "slot_width": 2.5,
    "slot_length": 5.0,
    "aisle_width": 5.8,
    "neighbor_occupancy": 0.7
  },

  "ego_init": {
    "x": ...,
    "y": ...,
    "yaw": ...,
    "v": ...,
    "steer": ...
  },

  "target_slot": {
    "slot_id": "...",
    "center_x": ...,
    "center_y": ...,
    "yaw": ...,
    "polygon": [...]
  },

  "ood_type": "none"
}
```

## 9.2 episode npz

每条采集出来的轨迹：

```text
episode_id
task_id
source_type              # expert_success / noisy_expert / recovery
obs_rgb                  # [T, H, W, 3]
state                    # [T, state_dim]
action                   # [T, action_dim]
raw_action               # 原始控制量
goal_state
target_slot_polygon
subgoal_indices
subgoal_states
subgoal_images
collision_flags
outbound_flags
success_flag
stage_id_per_step
metadata_json
```

state 至少包括：

```text
x, y, yaw
v_longitudinal
v_lateral，可选
steer
gear/reverse 或 throttle_sign
distance_to_goal
heading_error_to_goal
```

action 建议同时存两份：

```text
model_action:
  你当前模型用的 action，例如 [steer, throttle]

raw_control:
  steer, throttle, brake, reverse
```

E2E Parking Dataset 里明确记录了控制标签，包括 throttle、brake、steering、reverse，并列出了范围与分辨率；你不一定照它离散化，但“把 reverse/gear 显式存出来”很值得借鉴。([arXiv][2])

---

# 10. subgoal 该怎么存

你现在是分层规划，所以数据里必须直接存 subgoal，而不是每次临时算。

建议每个 expert trajectory 提取 4 个关键 subgoal：

```text
subgoal_0: approach
  接近车位入口，车辆仍主要前进

subgoal_1: setup
  准备倒车的位置，接近换挡/大转向开始点

subgoal_2: entry
  车尾进入车位，车身角度开始对齐

subgoal_3: final
  目标停车姿态
```

Codex 可以先用规则提取：

```text
setup point:
  第一次 reverse / throttle 变负之前的附近帧

entry point:
  后轴或车尾进入 target slot extended boundary 的帧

final point:
  成功停车末帧

approach point:
  start 到 setup 之间按 progress 取 40%-60% 的帧
```

如果当前没有 gear/reverse，就用 longitudinal velocity 或 throttle sign 判断换挡点。

Noisy/recovery trajectory 的 subgoal 不一定从自身提取，而可以继承对应 task 的 expert reference subgoals。这样每个 task 都有稳定的阶段目标。

---

# 11. retrieval bank 怎么建

你的 retrieval-guided MPPI 不应该直接从所有训练片段里检索。应该建一个干净的 `retrieval_bank`。

包含：

```text
expert_success trajectories
recovered_success trajectories
```

不包含：

```text
noisy failed trajectories
collision trajectories
test / val trajectories
```

每个 retrieval item 存：

```text
bank_item_id
task_id
source_episode_id
stage_id
start_state
end_state
start_latent，可选
end_latent，可选
action_sequence
duration
success_after_segment
difficulty
parking_type
```

检索距离第一版可以用 state-space，不要先上复杂 latent：

```text
D =
w_start_xy     * ||current_xy - bank_start_xy||
+ w_start_yaw  * yaw_error(current_yaw, bank_start_yaw)
+ w_goal_xy    * ||subgoal_xy - bank_end_xy||
+ w_goal_yaw   * yaw_error(subgoal_yaw, bank_end_yaw)
+ w_v          * |current_v - bank_start_v|
+ w_steer      * |current_steer - bank_start_steer|
```

注意你自己已经发现：同样 subgoal 位移下，速度、车身角、前轮角不同，动作效果差很多。所以 `v` 和 `steer` 一定要进检索 key。

---

# 12. 训练前必须做的数据诊断

这一步不要省。它能直接避免“训练完才发现数据还是偏”。

让 Codex 写：

```text
tools/diagnose_parking_dataset.py
```

输出这些图和表：

## 12.1 动作分布

```text
steer histogram
throttle histogram
gear/reverse ratio
brake ratio，如果有
```

重点看：

```text
abs(steer) > 0.3
abs(steer) > 0.5
abs(steer) > 0.8
reverse ratio
low-speed large-steer ratio
```

## 12.2 状态覆盖

```text
relative x/y to target slot
relative yaw to target slot
velocity distribution
steer state distribution
distance_to_goal distribution
```

尤其要画：

```text
expert vs noisy vs recovery 的 relative pose scatter
```

你要看到 recovery 数据确实覆盖了 expert tube 周围，而不是和 expert 重合。

## 12.3 阶段覆盖

```text
stage 0/1/2/3 sample count
每个 stage 的平均长度
每个 stage 的 reverse ratio
每个 stage 的 large-steer ratio
```

如果 stage 2/3 样本太少，planner 后半段一定容易失败。

## 12.4 检索覆盖

在 val/test_id 上，计算每个 subproblem 到 train retrieval bank 的最近距离：

```text
min_retrieval_distance
top1 / top5 retrieval distance
retrieval stage match rate
```

如果失败 case 的 `min_retrieval_distance` 明显更大，就证明你的问题确实是数据覆盖不足。

这可以成为论文里很有价值的分析图。

---

# 13. 评估指标：直接对齐 E2E Parking Dataset

闭环主指标建议用这些：

| 指标  | 含义                                     |
| --- | -------------------------------------- |
| TSR | Target Success Rate，成功停入目标车位           |
| TFR | 到了目标车位但误差超阈值                           |
| NTR | 停到了非目标车位                               |
| CR  | Collision Rate                         |
| OR  | Outbound Rate                          |
| TR  | Timeout Rate                           |
| APE | Average Position Error                 |
| AOE | Average Orientation Error              |
| APT | Average Parking Time                   |
| AIT | Average Inference Time / planning time |

E2E Parking Dataset 对这些指标有明确定义：TSR 是成功停入指定车位；成功阈值是横向 0.6m、纵向 1m、姿态误差 10°；同时还定义了 TFR、NTR、CR、OR、TR、APE、AOE、APT、AIT。([arXiv][2])

你可以额外加两个你自己方法相关的指标：

```text
Subgoal Success Rate
Stage Failure Distribution
```

例如：

```text
stage_0_success
stage_1_success
stage_2_success
stage_3_success

failure_stage:
  fail_before_setup
  fail_during_reverse
  fail_final_alignment
```

这能说明数据集改进后到底是哪个阶段变好了。

---

# 14. OOD 测试怎么做

你的 OOD 不要做太多，先做 4 个就够。

## OOD-Layout

改变停车场几何：

```text
aisle_width 变窄或变宽
slot_width / slot_length 变化
slots_per_row 变化
target slot 更靠边
道路入口方向变化
```

目标：证明不是只记住一个停车场 layout。

## OOD-Init

改变初始姿态：

```text
更远起点
更大 yaw error
更极端 start_family
更接近边缘的初始位置
```

目标：证明 subgoal + retrieval 对起点变化有泛化。

## OOD-Dynamics

改变车辆动力学：

```text
wheelbase ±10%
max_steering ±10%
friction ±20%
control_delay 0.1s / 0.2s
max_speed 变化
```

目标：证明方法不是卡死在某个车辆参数上。

## OOD-Visual

改变视觉域：

```text
ground texture
vehicle color
camera height
camera yaw/pitch 小扰动
lighting
shadow
```

E2E Parking Dataset 也专门迭代了阴影、天气/光照和 seed 分配，它的 Gen 1C/2A/2B 里加入了 shadow slots 和不同 weather/seed 设置，这说明视觉变化确实是自动泊车数据构建里需要考虑的因素。([arXiv][2])

---

# 15. Codex 具体任务单

你可以直接把下面这段给 Codex。

## 任务 1：定义配置文件

```text
conf/metadrive_parking_dataset_v2.yaml
```

包含：

```text
dataset_root
image_size
fps
frameskip
episode_max_steps

splits:
  train:
    n_episodes: 1500
    scenario_seed_start: 10000
  val:
    n_episodes: 200
    scenario_seed_start: 20000
  test_id:
    n_episodes: 384
    scenario_seed_start: 30000
  test_ood_layout:
    n_episodes: 128
    scenario_seed_start: 40000
  test_ood_init:
    n_episodes: 128
    scenario_seed_start: 41000
  test_ood_dynamics:
    n_episodes: 128
    scenario_seed_start: 42000
  test_ood_visual:
    n_episodes: 128
    scenario_seed_start: 43000

train_composition:
  expert_success: 0.50
  noisy_expert: 0.30
  recovery: 0.20

difficulty_mix:
  easy: 0.30
  medium: 0.50
  hard: 0.20
```

---

## 任务 2：生成 task split

```text
tools/generate_parking_tasks_v2.py
```

输入：

```text
--config conf/metadrive_parking_dataset_v2.yaml
--split train/val/test_id/test_ood_layout/...
```

输出：

```text
data/metadrive_parking_v2/splits/*.json
```

要求：

```text
1. 每个 task 有 task_id, split, scenario_seed, route_seed, noise_seed
2. 每个 task 有 difficulty, start_family, target_slot, layout_params
3. train/val/test 的 scenario_seed 不重叠
4. OOD split 只改变一个主要因素
5. 所有 task 可复现
```

---

## 任务 3：采 expert success

```text
tools/collect_parking_data_v2.py --mode expert_success
```

要求：

```text
1. 读取 split task json
2. 调用 expert/controller
3. 保存成功 episode
4. 失败则记录到 failed_tasks.json
5. 保存完整 metadata
```

---

## 任务 4：采 noisy expert

```text
tools/collect_parking_data_v2.py --mode noisy_expert
```

要求：

```text
1. 从 train tasks 采 noisy expert
2. 支持 small/medium/large noise
3. 支持 full noisy rollout 和 noisy+correction rollout
4. 失败也可保存，但必须打标签
5. 控制 collision 样本比例不要过高
```

---

## 任务 5：采 recovery

```text
tools/collect_parking_data_v2.py --mode recovery
```

支持三类 recovery：

```text
1. expert_midpoint_perturb
2. wrong_action_prefix
3. planner_failure_recovery
```

其中 planner failure 只能来自 train tasks。

---

## 任务 6：构建 retrieval bank

```text
tools/build_retrieval_bank_v2.py
```

要求：

```text
1. 只使用 train split
2. 只使用 expert_success 和 recovered_success
3. 不使用 val/test
4. 按 stage 切片
5. 保存 start_state, end_state, action_sequence, stage_id
6. 支持 state-space retrieval key
```

---

## 任务 7：数据诊断

```text
tools/diagnose_parking_dataset_v2.py
```

输出：

```text
diagnostics/action_histograms.png
diagnostics/state_coverage.png
diagnostics/relative_pose_scatter.png
diagnostics/stage_distribution.png
diagnostics/retrieval_coverage_val.png
diagnostics/summary.json
```

summary 至少包括：

```text
num_episodes_by_source
num_steps_by_source
success_rate_by_source
collision_rate_by_source
reverse_ratio
large_steer_ratio
low_speed_large_steer_ratio
stage_counts
mean_episode_length
retrieval_min_dist_mean_on_val
retrieval_min_dist_p90_on_val
```

---

## 任务 8：统一闭环评估

```text
tools/eval_parking_suite_v2.py
```

输入：

```text
--split test_id / test_ood_layout / ...
--planner flat_mppi / hierarchical_mppi / retrieval_only / retrieval_guided_mppi
--model_ckpt ...
--retrieval_bank ...
```

输出：

```text
results/{planner}/{split}/metrics.json
results/{planner}/{split}/episodes.csv
results/{planner}/{split}/videos/
```

metrics：

```text
TSR
TFR
NTR
CR
OR
TR
APE
AOE
APT
AIT
stage_success_rate
failure_stage_distribution
```

---

# 16. 最小论文实验表：不要做太复杂

你现在不做 Structured Maneuver Perturbation 是对的，手工痕迹太重。下一版论文实验可以压缩成三张表。

## 表 1：数据集 ablation

| Training data             | TSR ↑ | APE ↓ | AOE ↓ | CR ↓ | TR ↓ |
| ------------------------- | ----: | ----: | ----: | ---: | ---: |
| Expert only               |       |       |       |      |      |
| Expert + Noisy            |       |       |       |      |      |
| Expert + Noisy + Recovery |       |       |       |      |      |

这张表回答核心问题：

> 数据覆盖是否解决 planner-induced off-expert 状态的问题？

## 表 2：planner ablation

| Planner               | Hierarchy | Retrieval | WM scoring | TSR ↑ | APE ↓ | AOE ↓ |
| --------------------- | --------: | --------: | ---------: | ----: | ----: | ----: |
| Flat MPPI             |         否 |         否 |          是 |       |       |       |
| Hierarchical MPPI     |         是 |         否 |          是 |       |       |       |
| Retrieval-only        |         是 |         是 |          否 |       |       |       |
| Retrieval-guided MPPI |         是 |         是 |          是 |       |       |       |

这张表回答：

> retrieval prior 有用吗？WM scoring 真的有贡献吗？

## 表 3：泛化测试

| Test split   | TSR ↑ | APE ↓ | AOE ↓ | CR ↓ | TR ↓ |
| ------------ | ----: | ----: | ----: | ---: | ---: |
| ID           |       |       |       |      |      |
| OOD-Layout   |       |       |       |      |      |
| OOD-Init     |       |       |       |      |      |
| OOD-Dynamics |       |       |       |      |      |
| OOD-Visual   |       |       |       |      |      |

这张表回答：

> MetaDrive 自建数据是不是只在自己调出来的场景里有效？

这三张表就够了，不需要再做 top-k、ratio、sigma、opt_steps 全部网格。`opt_steps=0/1/3/5` 可以做一个小图或 appendix，用来说明 WM refinement 有时会改坏 prior，但不要作为主消融。

---

# 17. 论文里怎么讲这个数据集

不要写：

> We propose a new benchmark.

这样会把审稿人注意力引到“你的 benchmark 是否足够大、是否公开、是否比 CARLA 好”。

建议写：

> We construct a controlled MetaDrive-Parking evaluation suite to systematically study world-model-based parking planning. Following the closed-loop evaluation practice in E2E Parking and the held-out seed protocol in MetaDrive, we use fixed train/validation/test splits, ID/OOD scenarios, and parking-specific success/failure metrics.

中文：

> 我们构建了一个受控的 MetaDrive-Parking 评估协议，用来系统研究基于世界模型的自动泊车规划。参考 E2E Parking 的闭环停车指标和 MetaDrive 的 held-out seed 泛化评估方式，我们固定训练/验证/测试划分，并设置 ID/OOD 场景与停车专用成功/失败指标。

这样就不会显得你在自创 benchmark，也不会显得 MetaDrive 太 low。

---

# 18. 你现在下一步最实际的路线

我建议你按这个顺序推进：

```text
Step 1:
做 MetaDrive-Parking-v2 MVP：
1500 train + 200 val + 384 test_id + 4×128 OOD

Step 2:
先训练 Expert-only 和 Expert+Noisy+Recovery 两个 WM
不要一开始训太多版本

Step 3:
在 val 上看：
- open-loop rollout error
- retrieval coverage
- stage-wise prediction error

Step 4:
在 test_id 上跑：
- Flat MPPI
- Hierarchical MPPI
- Retrieval-only
- Retrieval-guided MPPI

Step 5:
如果确实提升，再扩展到 5000 train

Step 6:
最后跑 OOD splits
```

你现在最需要证明的是：

> Expert-only 数据不足以支撑 closed-loop WM planning；加入 noisy/recovery 数据后，WM 对 planner-induced off-expert states 的预测更可靠，retrieval-guided MPPI 的阶段成功率和最终停车成功率随之提升。

这个故事非常合理，而且比继续堆 optimizer 更像论文。

[1]: https://arxiv.org/html/2411.04983v2?utm_source=chatgpt.com "DINO-WM: World Models on Pre-trained Visual Features ..."
[2]: https://arxiv.org/html/2504.10812v1 "E2E Parking Dataset: An Open Benchmark for End-to-End Autonomous Parking"
[3]: https://metadrive-simulator.readthedocs.io/en/latest/rl_environments.html "Environments — MetaDrive 0.1.1 documentation"
[4]: https://arxiv.org/abs/2406.03877 "[2406.03877] Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop End-To-End Autonomous Driving"
[5]: https://arxiv.org/html/2509.13956v1 "SEG-Parking: Towards Safe, Efficient, and Generalizable Autonomous Parking via End-to-End Offline Reinforcement Learning"
[6]: https://arxiv.org/abs/2204.10777?utm_source=chatgpt.com "ParkPredict+: Multimodal Intent and Motion Prediction for Vehicles in Parking Lots with CNN and Transformer"
