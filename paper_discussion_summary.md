# DINO-WM Parking Research Notes for Paper Discussion

更新日期：2026-05-14

这份笔记整理了当前对话中形成的核心研究思路、代码框架、实验设计和阶段性结论，方便后续继续讨论论文创新点与故事线。当前项目基于 DINO-WM，在 MetaDrive 停车任务上做闭环规划，并尝试将模型迁移/适配到真实停车场数据集 DLP 上做开环预测验证。

## 1. 当前研究定位

本项目的主线不是单纯复现 DINO-WM，而是把 DINO-WM 的视觉世界模型用于自动泊车任务，重点关注：

1. 停车场景中长程规划困难的问题。
2. 分层 subgoal 规划对世界模型预测误差累积的缓解作用。
3. 在世界模型较准但优化器搜索困难时，引入 expert-guided MPPI/CEM 作为更有效的 subplanner。
4. 从仿真 MetaDrive 到真实 DLP 停车场数据的视觉/动力学适配。

当前最稳妥的论文定位是：

- **MetaDrive**：主实验平台，负责闭环规划、成功率、分层规划、guided planning 等核心结果。
- **DLP**：真实数据补充实验，负责验证模型经过轻量适配后能否在真实停车场视觉域上进行开环预测。

需要避免过度声明：

- 不建议声称 DLP 上实现了真实闭环控制，因为 DLP 不是交互式环境。
- 不建议声称 MetaDrive 模型可以直接 zero-shot 到 DLP。当前实验已经证明直接迁移效果很差。
- 更合适的说法是：MetaDrive 预训练世界模型不能直接零样本迁移到 DLP，但可以作为真实数据适配的初始化，在少量 DLP fine-tune 后获得更好的真实停车场开环预测能力。

## 2. MetaDrive 部分：当前代码框架

主要文件：

- `plan_park_3.py`：早期分层停车规划主脚本，包含多阶段 subgoal、MPC/CEM/MPPI 等逻辑。
- `plan_park_3_guided.py`：当前更重要的 guided planning 版本，基于分层框架加入 expert-guided MPPI。
- `test_guided_mppi.py`：单独测试 guided MPPI 的脚本，用来验证 expert-guided prior 是否能在短 horizon 内提升优化效果。
- `conf/plan_park.yaml`：原始停车规划配置。
- `conf/plan_park_guided.yaml`：guided planning 的配置文件。
- `planning/`：子规划器相关实现，包含 CEM/MPPI 等优化器。

### 2.1 分层规划思路

早期发现：

- 单段长程规划很难完成完整停车。
- 世界模型在长 horizon 下预测误差累积明显。
- 更短 MPC horizon 可以缓解误差累积，但不能根治优化器搜索困难。

因此采用分层思路：

1. 从专家轨迹中抽取 subgoals。
2. 每个阶段只规划到下一个 subgoal。
3. 每个阶段使用较短 horizon，例如 15 帧 / 3 个 macro steps。
4. 特殊的 setup point 不再强行插入到某个阶段内部，而是作为独立或更平滑的 subgoal 处理，避免局部 subgoal 被扰乱。

这个思路的论文价值在于：

- 将完整泊车任务拆成短 horizon、局部可预测的子任务。
- 让世界模型更容易在可信预测长度内工作。
- 减少长程视觉预测误差对控制优化的破坏。

### 2.2 发现的问题：不是 WM 完全不会动力学，而是优化器搜索不到

关键实验：

- 直接把专家动作输入世界模型和仿真环境，完整停车可以成功。
- 在 WM 想象中，车辆状态变化和仿真环境非常接近，畸变不明显。

这说明：

- WM 在专家轨迹附近确实学到了一定的车辆运动动力学。
- 问题不完全是“模型不会预测”。
- 更大的瓶颈是 CEM/MPPI 在高维动作空间里搜索不到正确控制序列，尤其是低速大转向、倒车、细微修正这类动作。

换句话说：

> 当前 WM 更像是局部有效的动力学模型，但普通随机优化器很难在复杂停车动作空间中找到 expert-like sequence。

### 2.3 Expert-guided MPPI 的定位

曾经直接使用 expert action 作为“作弊”上限实验：

- 每个阶段给定当前起点和 subgoal。
- 让 expert/controller 在短 horizon 内生成动作。
- 将动作同时作用于 WM 和仿真环境。
- 结果能成功，证明分层 subgoal 和 WM 在 expert 附近是可行的。

后来改为 expert-guided MPPI，而不是纯执行 expert：

- expert 不再直接决定最终动作。
- expert/action prior 作为 MPPI 的 proposal center 或 candidate guidance。
- MPPI 仍然基于 WM objective 对采样动作进行打分和优化。

论文里可以这样区分：

- **直接 expert action**：上限实验，证明任务拆分与 WM 局部预测可行，但作弊感强。
- **expert-guided MPPI**：使用先验引导动作搜索，减少随机采样难度，仍保留 WM-based planning 结构。

更进一步减少作弊感的方向：

1. 将 expert prior 替换为训练集检索得到的相似轨迹片段。
2. 将 expert prior 替换为 BC policy 输出。
3. 将 prior 只作为动作分布初始化，而非强制执行。

这样可以把故事从“作弊”改成：

> 停车任务具有强结构化动作模式，纯随机 shooting 效率低，因此引入数据驱动的 action prior 来提升世界模型规划的搜索效率。

## 3. MetaDrive 实验设计建议

MetaDrive 是论文主实验，应优先做扎实。

### 3.1 主实验

建议比较：

1. Flat CEM / MPPI：不分层，直接规划完整目标。
2. Hierarchical CEM / MPPI：分层 subgoal，但无 expert guidance。
3. Hierarchical expert upper bound：直接执行 expert/controller 动作，上限实验。
4. Hierarchical expert-guided MPPI：当前重点方法。

主要指标：

- 停车成功率。
- 最终 pose 误差。
- 最终 heading 误差。
- 到达每个 subgoal 的成功率。
- 每个阶段的规划步数/重规划次数。
- 动作平滑性或控制幅度统计。

### 3.2 消融实验

可以做：

- 不同 horizon H。
- 不同 subgoal 数量。
- 有无 setup point 特殊处理。
- 有无 expert/action prior。
- 不同 prior 权重。
- CEM vs MPPI vs guided MPPI。

### 3.3 重要论文论点

MetaDrive 部分可以支撑以下论点：

1. 停车任务中的长程视觉规划难度高，直接 DINO-WM planning 成功率不足。
2. WM 在 expert 轨迹附近预测能力较好，但普通采样优化器搜索效率不足。
3. 分层 subgoal 可以把长程停车拆成短 horizon 局部规划。
4. Expert-guided/data-guided action prior 可以显著改善子规划器在停车动作空间中的搜索效率。

## 4. DLP 数据集部分：目标和限制

DLP 是真实停车场数据集，包含：

- 原始视频。
- `agents.json`
- `instances.json`
- `frames.json`
- `obstacles.json`
- `scene.json`

数据位置：

- 项目软链接：`dlp_dataset/data`
- 实际数据盘：`/root/autodl-tmp/dlp_dataset/data`

当前已经下载 DJI_0001 到 DJI_0030。

### 4.1 DLP 可以做什么

DLP 适合做：

- 真实停车场视觉域的开环预测。
- sim-to-real/domain adaptation 的补充验证。
- 从真实轨迹中构造 BEV-like 输入，测试 WM 是否能预测未来画面。

DLP 不适合直接做：

- 真正闭环控制。
- 真实环境交互式规划。

原因：

- DLP 是离线记录数据。
- 没有可交互仿真器。
- 模型动作无法改变后续真实视频。

所以论文中应明确：

> DLP experiment is open-loop visual prediction on real-world parking data, not closed-loop control.

### 4.2 DLP 视觉预处理

早期直接使用作者的处理函数失败，原因包括：

1. 视角不一致：MetaDrive 是固定视角，DLP 原始跟踪某个 agent 时像 agent 不动的 BEV 视角。
2. 颜色风格不一致：MetaDrive 中自车、周车、背景颜色和真实 DLP 差异很大。
3. DLP 轨迹有残影，需要去掉。
4. 停车场背景变化很大，不同 scene 的视觉分布差异明显。
5. DLP 中车辆类型很多，不同车辆外观也会增加视觉建模难度。

因此我们新建脚本在 `tools/` 下生成 MetaDrive-like DLP 视频/npz，而不是直接接入原始图像。

重要脚本：

- `tools/prepare_dlp_open_loop.py`

当前生成的 DLP npz 数据集：

- `/root/autodl-tmp/dlp_datasets/dlp_metadrive_npz_stage1_full_recon`
- 580 episodes
- 每个 episode 约 101 raw frames
- 训练/验证大致 522 / 58 split

### 4.3 DLP 与 MetaDrive 的关系

已经观察到：

- MetaDrive 训练的 WM 直接用于 DLP，会很快把场景预测回 MetaDrive 风格。
- 说明视觉域差异对 latent dynamics 影响很大。
- DINO-WM 的 zero-shot 是任务层面的 zero-shot，不等于任意真实视觉域的 sim-to-real zero-shot。

更合理的故事是：

> DINOv2 特征和 MetaDrive 训练提供了较好的初始化，但 DLP 仍需要视觉/latent 适配。

## 5. DLP 训练阶段设计

### 5.1 Stage 0：数据准备

目标：

- 从 DLP json/video 生成类似 MetaDrive 输入的 npz。
- 统一图像尺寸、颜色风格、视角裁剪。
- 输出到数据盘，避免占用项目盘。

输出：

- `/root/autodl-tmp/dlp_datasets/dlp_metadrive_npz_stage1_full_recon`

### 5.2 Stage 1：只训练 DLP decoder

配置：

- `conf/train_dlp_recon_stage1.yaml`

目标：

- 冻结 DINO encoder 和 MetaDrive predictor。
- 只训练 decoder，让模型能重建 DLP 风格图像。

结果：

- 训练 80 epochs。
- Training loss 从约 0.0156 降到 0.0011。
- Validation loss 从约 0.0395 降到 0.0378。

结论：

- 训练集重建明显变好。
- 验证集提升有限。
- decoder-only 能改善视觉重建，但不能解决 predictor rollout。

论文意义：

- 证明单纯视觉解码适配不够。
- DLP 的挑战不只是像素重建，还有 latent dynamics 对齐。

### 5.3 Stage 1.5：冻结 MetaDrive predictor，训练 decoder 对齐 predictor 输出

配置：

- `conf/train_dlp_decoder_align_stage15.yaml`

目标：

- 使用 MetaDrive predictor 的 rollout latent。
- 冻结 predictor。
- 训练 DLP decoder 去解码 predictor 输出的 latent。

这和 Stage 1 的区别：

- Stage 1 decoder 学的是“真实 DLP latent -> DLP image”。
- Stage 1.5 decoder 学的是“MetaDrive predictor 输出 latent -> DLP image”。

直观解释：

> Stage 1 让 decoder 会画 DLP；Stage 1.5 让 decoder 会画 MetaDrive predictor 想象出来的 DLP。

结果：

- 有一定改善，但并不足以让长 rollout 稳定。
- 说明 predictor 输出 latent 与 DLP 真实未来 latent 仍有较大偏差。

论文意义：

- 证明仅对 decoder 做适配仍然有限。
- 进一步支持需要轻量 predictor fine-tune。

### 5.4 Stage 2：从 MetaDrive 初始化，轻量 fine-tune predictor

配置：

- `conf/train_dlp_predictor_finetune_stage2.yaml`

脚本：

- `tools/train_dlp_predictor_finetune_stage2.py`

初始化：

- Predictor 使用 MetaDrive checkpoint。
- Decoder 使用 Stage 1.5 初始化。

训练策略：

- 冻结 DINO encoder。
- 训练 predictor / action encoder / proprio encoder / decoder。
- Predictor 学习率较小。
- Decoder 学习率更小，避免破坏已有 DLP 风格重建。

用户训练：

- 40 epochs。

训练日志末期大致：

- train loss：约 0.02078
- train latent：约 0.02013
- train pred pixel：约 0.00108
- train recon：约 0.00107
- valid loss：约 0.07372
- valid latent：约 0.05113
- valid pred pixel：约 0.03775
- valid recon：约 0.03724

结论：

- Stage 2 是目前 DLP 上最有效的方案。
- 它不等于完全重训，因为：
  - 使用 MetaDrive predictor 初始化。
  - 冻结 DINO encoder。
  - fine-tune 成本低于从零训练。
  - 可以和 from-scratch DLP training 做对比，证明 MetaDrive 初始化是否有价值。

论文表达：

> We use MetaDrive pretrained dynamics as initialization and adapt it to DLP with lightweight fine-tuning, showing that simulation-trained world dynamics can provide a useful prior for real parking open-loop prediction.

## 6. DLP 评估：重要 frameskip 修正

训练配置中有：

- `frameskip = 5`
- `num_hist = 3`
- `num_pred = 1`
- 每个 model step 对应 5 个 raw frames。
- action token 是 5 帧动作拼接后的 macro action，维度为 `5 * 2 = 10`。

`data_loader_park.py` 的行为：

- 图像和 state：`start:end:frameskip` 下采样。
- action：raw action reshape 成 macro action。

早期评估 bug：

- WM 预测是 model timestep。
- target video/metric 用的是 raw DLP frames。
- 导致 WM 视频看起来比 target 走得快约 5 倍。
- 旧的无 `_aligned` 输出不能作为正式结果。

已修正脚本：

- `tools/test_dlp_hybrid_predictor_decoder.py`
- `tools/export_dlp_wm_videos.py`

修正方式：

- 读取 raw frames 后按 `frameskip=5` 下采样 target image/state。
- 将 raw actions reshape 成 macro actions。
- 保证 target video 和 WM prediction 都在 model timestep 上比较。

正式使用的结果目录应带 `_aligned`：

- `tools/outputs/dlp_hybrid_predictor_decoder_stage2_epoch40_aligned/norm_metadrive`
- `tools/outputs/dlp_stage2_video_compare_aligned`

不要使用这些旧结果作为论文依据：

- `tools/outputs/dlp_hybrid_predictor_decoder_stage2_epoch40/norm_metadrive`
- `tools/outputs/dlp_stage2_video_compare`

## 7. DLP Stage 2 当前修正后结果

修正 frameskip 后，Stage 2 epoch40 的小规模评估结果明显好转：

输出目录：

- `tools/outputs/dlp_hybrid_predictor_decoder_stage2_epoch40_aligned/norm_metadrive`

summary：

```text
mean_future_wm_mse: 0.011227403185330331
mean_reconstruction_mse: 0.00576111853782398
mean_future_persistence_mse: 0.02351167642821868
mean_future_wm_over_persistence: 0.4756754090388616
mean_future_wm_better_than_persistence_pct: 83.33333333333333
```

解释：

- WM 未来预测 MSE 约为 persistence baseline 的 47.6%。
- 约 83.3% clips 中 WM 优于 persistence。
- 说明 Stage 2 已经不是“自嗨式视频”，而是有定量优势。

但注意：

- 当前只是少量 clips。
- 仍有失败 outlier。
- 需要扩大 clip 数量后才能作为正式论文统计。

### 7.1 可视化输出

视频输出目录：

- `tools/outputs/dlp_stage2_video_compare_aligned`

每个 clip 包含：

- `target_video.mp4`：真实 DLP 下采样后的目标视频。
- `wm_pred_video.mp4`：WM 开环预测视频。
- `metrics.json`：该 clip 的 MSE 和 baseline 指标。

当前例子：

- `clip_0000_episode_000001_DJI_0028_3cd22e8c_06710`
- `clip_0003_episode_000004_DJI_0013_dda69adc_01210`

由于 DLP 每个 npz 约 101 raw frames，frameskip=5，因此可比较长度约为 20 model steps。

## 8. DLP 下一步正式实验

为了支撑论文，建议补以下实验。

### 8.1 大规模开环统计

使用修正后的 aligned evaluation，在更多 DLP clips 上跑：

- 50 clips
- 100 clips
- 如果时间允许，200 clips

报告：

- WM open-loop MSE
- persistence baseline MSE
- WM / persistence ratio
- better-than-persistence percentage
- horizon-wise error curve

尤其需要画：

- 误差随预测 horizon 增长曲线。
- Stage0/1/1.5/2 与 persistence 的柱状图或折线图。

### 8.2 Ablation

建议对比：

1. MetaDrive WM zero-shot on DLP。
2. Stage 1 decoder-only。
3. Stage 1.5 decoder alignment。
4. Stage 2 predictor fine-tune。
5. Persistence baseline。

可选：

- Stage 2 from scratch on DLP。
- Stage 2 without MetaDrive initialization。
- Stage 2 with decoder reinitialized vs Stage1.5 initialized。

最关键的是证明：

> MetaDrive initialization + lightweight DLP adaptation 优于直接 zero-shot，也优于只训练 decoder。

如果能加 from-scratch 对比，就能进一步说明 MetaDrive 预训练是否真的有价值。

### 8.3 视频案例

论文中建议放：

- 成功案例。
- 中等案例。
- 失败案例。

失败案例也有价值，可以说明：

- 车辆外观变化。
- 背景复杂。
- 遮挡。
- 大角度运动。
- 稀有车辆类型。

这会让论文更可信。

## 9. 关于 zero-shot 的谨慎表述

DINO-WM 原文强调 zero-shot，主要指：

- 测试时不需要针对下游任务训练策略。
- 给定当前图像和目标图像后，可以通过 WM planning 达到目标。
- 对新任务/新目标有一定泛化。

但这不等于：

- 从仿真视觉域到真实停车场视觉域可以完全 zero-shot。
- 不同相机、不同渲染风格、不同车辆外观下无需任何适配。

因此本项目建议这样写：

不推荐：

> Our MetaDrive-trained model zero-shot transfers to DLP.

推荐：

> Direct zero-shot transfer from MetaDrive to DLP is challenging due to substantial visual domain shift. However, the simulation-trained model provides a useful initialization for real-data adaptation. With lightweight fine-tuning on DLP, the world model achieves better open-loop prediction than persistence baselines.

中文含义：

> 我们不宣称纯零样本 sim-to-real 成功，而是证明仿真预训练提供了可迁移的初始化，经过轻量真实数据适配后，可以在真实停车场数据上获得有效开环预测能力。

## 10. 可能的论文创新点组织

可以考虑将创新点组织为三层。

### 创新点 1：面向泊车任务的分层视觉世界模型规划框架

泊车任务长程、低速、大转向、倒车修正多，直接 DINO-WM planning 容易失败。本文将完整泊车过程分解为若干 subgoal-guided stages，让每个 stage 落在 WM 更可信的短预测长度内。

关键词：

- Hierarchical visual planning
- Subgoal decomposition
- Reliable prediction horizon
- Parking-specific long-horizon decomposition

### 创新点 2：Expert/Data-guided MPPI 提升停车动作搜索效率

实验发现 WM 在 expert 附近能较好预测，但普通 CEM/MPPI 搜索不到有效动作。本文引入 expert-guided 或 data-guided action prior，把低效随机搜索变成先验引导的采样优化。

关键词：

- Guided action prior
- Expert-guided MPPI
- Data-guided planning
- Structured parking maneuver prior

需要注意：

- 如果最终论文使用 expert-guided，要弱化“直接 expert”色彩。
- 更好的论文版本是训练集检索 prior 或 BC policy prior。

### 创新点 3：真实停车场 DLP 开环适配验证

本文进一步在真实 DLP 停车场数据上测试模型迁移。直接 zero-shot 失败，但通过 decoder adaptation 和 predictor fine-tuning，可以获得真实视觉域上的开环预测能力。

关键词：

- Real-world parking open-loop prediction
- Sim-to-real world model adaptation
- Decoder adaptation
- Predictor fine-tuning

这部分不是主闭环贡献，而是增强真实数据可信度。

## 11. 当前最推荐的论文故事线

一条比较稳的故事线：

1. DINO-WM 在通用控制任务上表现好，但自动泊车有特殊挑战：低速、长程、大转向、倒车、目标姿态精确。
2. 直接 flat planning 在泊车中失败，主要原因包括长程预测误差和动作搜索困难。
3. 通过专家动作上限实验发现，WM 在 expert 轨迹附近其实可以预测较好，说明问题不完全是动力学没学会。
4. 因此提出分层 subgoal + guided MPPI：前者控制预测长度，后者提升动作搜索效率。
5. 在 MetaDrive 中进行闭环评估，展示成功率提升。
6. 为了验证真实视觉域的潜力，引入 DLP 数据集做开环预测实验。
7. 发现直接 sim-to-real zero-shot 不成立，但 MetaDrive 预训练经过轻量 DLP 适配后可以显著优于 persistence baseline。
8. 结论：视觉世界模型用于自动泊车具有潜力，但真实部署仍需要域适配和更强的闭环环境支持。

## 12. 当前代码/实验清单

MetaDrive planning：

- `plan_park_3.py`
- `plan_park_3_guided.py`
- `test_guided_mppi.py`
- `conf/plan_park.yaml`
- `conf/plan_park_guided.yaml`

DLP data preparation：

- `tools/prepare_dlp_open_loop.py`
- DLP data：`/root/autodl-tmp/dlp_dataset/data`
- DLP npz：`/root/autodl-tmp/dlp_datasets/dlp_metadrive_npz_stage1_full_recon`

DLP training：

- `conf/train_dlp_recon_stage1.yaml`
- `conf/train_dlp_decoder_align_stage15.yaml`
- `conf/train_dlp_predictor_finetune_stage2.yaml`
- `tools/train_dlp_predictor_finetune_stage2.py`

DLP evaluation/export：

- `tools/test_dlp_hybrid_predictor_decoder.py`
- `tools/export_dlp_wm_videos.py`
- Correct aligned output:
  - `tools/outputs/dlp_hybrid_predictor_decoder_stage2_epoch40_aligned/norm_metadrive`
  - `tools/outputs/dlp_stage2_video_compare_aligned`

旧的未对齐输出不要用于论文：

- `tools/outputs/dlp_hybrid_predictor_decoder_stage2_epoch40/norm_metadrive`
- `tools/outputs/dlp_stage2_video_compare`

## 13. 后续优先级

最高优先级：

1. 对 DLP Stage2 做更大规模 aligned open-loop 评估。
2. 对 Stage0/1/1.5/2 做同口径 ablation。
3. 在 MetaDrive 中整理 guided MPPI 的正式成功率表。
4. 导出 MetaDrive 成功/失败视频和 DLP target-vs-WM 视频。

中等优先级：

1. 做 horizon-wise prediction error curve。
2. 做 DLP failure case 分类。
3. 尝试训练集检索 prior 或 BC prior，替代直接 expert-guided，降低作弊感。

低优先级：

1. DLP 伪闭环。
2. CARLA 迁移。
3. 3DGS 场景闭环。

原因：

- DLP 没有真正交互环境，伪闭环说服力有限。
- CARLA 本质仍是仿真，和 MetaDrive 的论文增量有限。
- 3DGS 闭环实现成本高，短期风险大。

## 14. 一句话总结

当前最稳的论文表述是：

> 本文面向自动泊车任务，提出一种分层 subgoal 引导的视觉世界模型规划框架，并通过 expert/data-guided MPPI 缓解停车动作空间中的搜索困难。在 MetaDrive 中验证闭环规划效果，在 DLP 真实停车场数据上验证经过轻量适配后的开环预测能力。实验表明，纯 sim-to-real zero-shot 在真实停车场视觉域上并不可靠，但仿真预训练世界模型可以作为有效初始化，经过少量真实数据适配后显著优于简单 persistence baseline。

