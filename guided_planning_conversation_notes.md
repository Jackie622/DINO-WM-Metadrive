# Guided Parking Planning Conversation Notes

Date: 2026-05-11

This note summarizes the current discussion and implementation state for the DINO-WM MetaDrive parking planning experiments. It is not a verbatim transcript because the chat context was compacted, but it preserves the important technical reasoning, decisions, files, and experiment results.

## Background

The original parking planner struggled with hierarchical planning. Shorter MPC horizons reduced long-horizon WM rollout error, but did not solve the core issue: CEM/MPPI often failed to find the narrow expert-like action sequence needed for low-speed steering/reversing maneuvers.

The user initially wanted to:

- Diagnose the dataset action distribution.
- Try a "cheating" expert upper-bound by executing expert/controller actions.
- Avoid expensive retraining unless necessary.
- Build a thesis-feasible method that improves success rate while still being defensible.

The conda environment is:

```bash
conda activate dino_wm
```

or directly:

```bash
/root/miniconda3/envs/dino_wm/bin/python
```

## Earlier Plan Categories

### Plan A: Engineering / Oracle Baseline

Purpose: make the full pipeline run and measure an upper bound.

- Add expert/controller prior actions into candidate pools.
- Use WM plus geometry proxy reranking.
- Use more conservative subgoals.
- Add hybrid controller fallback.

This is useful as an upper-bound experiment, but has obvious "cheating" concerns.

### Plan B: Model Improvement

Purpose: make the WM genuinely learn parking dynamics.

- Add more data for large steering, reverse parking, low-speed turning, close-to-slot correction.
- Diagnose action distribution:
  - `abs(steer) > 0.3 / 0.5 / 0.8`
  - reverse steer distribution
  - large-turn segment length
  - samples near Phase1-like states
- Add action-conditioned auxiliary losses.
- Add pose/proprio auxiliary losses:
  - ego pose delta
  - heading delta
  - speed
  - action-conditioned next proprio
- Reweight multi-step losses toward early rollout accuracy.

Cost concern: retraining 5000 trajectories previously took about 4 days and about 800 RMB server cost.

### Plan C: Practical Middle Route

Chosen route:

1. Diagnose training data distribution.
2. Run expert/controller upper-bound.
3. If the upper-bound succeeds, identify WM/planner bottleneck.
4. Keep a controller/demo prior as an engineering baseline.
5. Later retrain a modest WM if needed.

## Important Experimental Findings

### Expert Oracle Works

An expert/action oracle was implemented in `plan_park_3.py`. It directly uses expert action segments from the current episode trace.

Result:

- It succeeds in both the simulation environment and WM imagination.
- In WM imagination, the vehicle almost does not deform and stays close to the simulator trajectory.

Interpretation:

- The WM can predict accurately near expert-like actions.
- The bottleneck is not simply that WM dynamics are impossible.
- The bigger issue is that CEM/MPPI cannot discover the narrow successful action manifold from random sampling.

### Expert + Noise Test

A separate script was created:

```text
test_expert_noise.py
```

It tests expert actions, noisy expert actions, and random actions.

Smoke result:

```text
expert success=1.0, env_dist=0.187, env_head=0.006
expert_noise_0.050 success=0.0, env_dist=1.290
random_uniform success=0.0, env_dist=8.914
```

Interpretation:

- The successful action manifold is very narrow.
- Even small noise around the expert can fail.
- WM can be locally accurate near expert, but uninformed search is weak.

### Guided MPPI Test

A separate test script was created:

```text
test_guided_mppi.py
```

It uses expert/demo actions as a prior and performs bounded residual MPPI in physical action space.

Smoke command:

```bash
env WANDB_MODE=disabled /root/miniconda3/envs/dino_wm/bin/python test_guided_mppi.py \
  hydra.run.dir=plan_outputs/guided_mppi_smoke \
  +guided_mppi.num_samples=32 \
  +guided_mppi.opt_steps=1 \
  +guided_mppi.eval_batch_size=32 \
  +guided_mppi.save_phase_video=false \
  +guided_mppi.save_final_video=false
```

Smoke result:

```text
success=1.000 dist=0.635 head=0.007 visual_div=101.241 proprio_div=0.540
```

Interpretation:

- Expert-guided residual MPPI can work.
- The selected actions remain close to expert prior.
- This supports using expert/demo prior as a practical planning improvement.

## Current Guided Pipeline

New files were created instead of modifying the original baseline directly:

```text
plan_park_3_guided.py
planning/guided_mppi.py
planning/mpc_park_guided.py
conf/plan_park_guided.yaml
```

The original baseline files remain conceptually separate:

```text
plan_park_3.py
conf/plan_park.yaml
```

### `planning/guided_mppi.py`

Defines:

```python
GuidedMPPIPlanner
```

Core behavior:

- Takes `actions` as a normalized expert/demo prior with shape `(B, H, action_dim)`.
- Converts normalized macro actions to physical action space.
- Samples bounded residuals around the prior.
- Clips physical actions to `[-1, 1]`.
- Converts candidates back to normalized actions.
- Scores candidates using WM rollout objective.
- Adds residual and smoothness regularization.
- Updates selected trajectory with MPPI softmax weights.

Important config fields:

```yaml
horizon: 10
num_samples: 512
opt_steps: 5
eval_every: 1
return_best: False
eval_batch_size: 100
temperature: 0.7
sigma_steer: 0.04
sigma_throttle: 0.06
clip_steer: 0.12
clip_throttle: 0.18
residual_weight: 0.2
smooth_weight: 0.05
save_iter_images: True
save_iter_video: False
```

It now saves internal optimization rollout images like:

```text
mpc_phase0_plan_0_output_1.png
mpc_phase0_plan_0_output_2.png
...
```

This matches the style of the original CEM/MPPI/GD subplanner outputs.

### `planning/mpc_park_guided.py`

Defines:

```python
MPCPlannerGuided
```

Core changes:

- Supports `prior_actions`.
- On the first MPC iteration, passes `self.prior_actions` to the subplanner.
- On later iterations, passes remaining memoized actions.
- Supports phase-level video/image control.
- Prints when rollout images are saved.

### `plan_park_3_guided.py`

Hydra entry point:

```python
@hydra.main(config_path="conf", config_name="plan_park_guided")
```

Important guided behavior:

- Checks `expert_guided.enabled`.
- Builds phase expert prior actions using `_phase_expert_prior_actions(phase_idx)`.
- Passes prior to `MPCPlannerGuided`.
- Runs hierarchical phase planning.
- Final evaluation saves a merged video if enabled.

Relevant functions:

```python
_phase_expert_prior_actions
_apply_phase_config
perform_planning
_final_eval_actions
```

### `conf/plan_park_guided.yaml`

Important config:

```yaml
expert_oracle:
  enabled: False

expert_guided:
  enabled: True

diagnostics:
  print_action_stats: False
  save_final_merged_video: True
  save_planner_rollout_images: True
  plot_full_rollout_images: False

planner:
  _target_: planning.mpc_park_guided.MPCPlannerGuided
  max_iter: null
  n_taken_actions: 5
  save_video: False
  sub_planner:
    target: planning.guided_mppi.GuidedMPPIPlanner
    horizon: 10
    num_samples: 512
    opt_steps: 5
    eval_every: 1
    return_best: False
    save_iter_images: True
    save_iter_video: False
```

Current logging/video behavior:

- Middle `[Action Stats]`, `[Action Seq]`, `[Action Detail]` are closed by default for guided runs.
- Intermediate phase videos are disabled by default.
- Internal subplanner images are still saved:

```text
mpc_phase0_plan_0_output_1.png
```

- Final merged video is saved:

```text
mpc_phase7_merged_final_0_success.mp4
```

## Validation Commands Already Run

Syntax checks passed:

```bash
python -m py_compile planning/guided_mppi.py planning/mpc_park_guided.py plan_park_3_guided.py
python -m py_compile planning/evaluator_park.py plan_park_3_guided.py planning/mpc_park_guided.py planning/guided_mppi.py
```

Config read checks passed in `dino_wm`:

```bash
/root/miniconda3/envs/dino_wm/bin/python -c "from omegaconf import OmegaConf; cfg=OmegaConf.load('conf/plan_park_guided.yaml'); print(cfg.diagnostics.print_action_stats, cfg.diagnostics.save_final_merged_video, cfg.planner.save_video, cfg.planner.sub_planner.save_iter_images)"
```

Output:

```text
False True False True
```

Full guided smoke command previously run:

```bash
env WANDB_MODE=disabled /root/miniconda3/envs/dino_wm/bin/python plan_park_3_guided.py \
  hydra.run.dir=plan_outputs/guided_full_smoke \
  planner.save_video=false \
  planner.sub_planner.num_samples=32 \
  planner.sub_planner.opt_steps=1 \
  planner.sub_planner.eval_batch_size=32
```

Result:

```text
Success rate: 1.0
```

## Discussion: Why WM Is Accurate But CEM/MPPI Fails

Important conclusion:

> WM can predict the expert trajectory accurately, but uninformed CEM/MPPI cannot search the expert-like action sequence.

Reasons:

- Parking has a narrow successful action manifold.
- Low-speed steering/reversing requires coordinated steer/throttle sequences.
- Short-term objective may reject actions that temporarily move away from the target but are needed for alignment.
- CEM/MPPI random sampling often explores action regions that WM can predict but that are dynamically irrelevant or unsuccessful.
- Expert-like action sequences are rare under uninformed Gaussian/uniform sampling.

So the main bottleneck is:

```text
action search / prior distribution
```

not purely:

```text
WM prediction accuracy
```

## Discussion: MPC Confusion

The user raised a valid concern:

If `H=5` and `n_taken_actions=1`, then near the goal the planner still plans five steps. It may overshoot instead of planning one step to the goal and four steps staying still. This makes fixed-horizon MPC feel awkward.

Clarified view:

- Standard MPC works when the objective rewards reaching the goal and staying there.
- It also assumes the planner can select braking/holding actions after reaching the goal.
- If the action/objective does not support "arrive then hold", fixed `H` can be bad near the goal.
- In parking, fixed-horizon MPC is especially awkward:
  - far from goal: horizon may be too short
  - near goal: horizon may be too long
  - necessary maneuvers may temporarily increase distance
  - frequent replanning can create unstable throttle/steer behavior

Better framing:

```text
stage-wise short-horizon trajectory optimization
```

rather than relying on generic fixed-horizon MPC as the main novelty.

## Relation To Original DINO-WM MPC Results

The original paper reports higher success for MPC than CEM/GD on tasks like PointMaze, Push-T, Wall, Rope, Granular.

This does not contradict the parking observations.

Original MPC likely helps because:

- It reduces compounding WM rollout error.
- It uses real feedback after short executions.
- Those tasks benefit from local correction.

Parking differs because:

- It is highly structured geometrically.
- It has narrow feasible action manifolds.
- Some correct actions temporarily move away from the target.
- Expert-like action search is the harder issue.

Suggested paper framing:

> Prior works show that MPC improves world-model planning by reducing long-horizon compounding errors. However, in structured parking, the main challenge is not only compounding prediction error but also the narrow feasible action manifold induced by low-speed steering and reversing maneuvers. Therefore, we retain the receding/subgoal planning idea, but replace uninformed random shooting with demonstration-guided residual optimization.

Possible method names:

```text
Demonstration-Guided Stage-wise MPC
Expert-Guided Residual MPPI for Hierarchical Parking
```

## Expert-Guided vs Direct Expert Action

Current guided prior still uses current episode expert trace, so it is still oracle-like.

Difference:

- Direct expert action: execute expert exactly.
- Expert-guided residual MPPI: use expert as a prior, then optimize residual actions using WM objective.

But if the prior comes from the current test episode expert trace, it remains a strong oracle. It is best treated as an upper-bound or proof-of-concept.

To reduce "cheating" in the paper:

### Training-set retrieval prior

Instead of using current episode expert actions:

- Search training demonstrations for a similar `(current pose, subgoal pose, stage)`.
- Use the retrieved action segment as the prior.
- MPPI then optimizes residuals around it.

This can be described as:

```text
retrieval-based demonstration prior
```

### BC policy prior

Train a small behavior cloning policy:

```text
pi_BC(obs, subgoal) -> action sequence
```

Then use its predicted sequence as the prior for residual MPPI.

This can be described as:

```text
learned behavior prior / BC-guided residual MPC
```

These options reduce test-time oracle concerns.

## Reliable Prediction Horizon

The user wants to estimate the WM's reliable prediction length.

Important nuance:

- Near expert trajectories, WM may accurately predict the complete parking process.
- But reliable prediction length should not be measured only near expert actions.
- It should account for action perturbations and state distribution.

Possible estimate:

1. Collect rollout pairs from simulator and WM.
2. Evaluate multiple horizons.
3. Measure:
   - pose error
   - heading error
   - visual latent error
   - proprio error
   - success/ranking consistency
4. Test both:
   - expert actions
   - perturbed expert actions
   - planner-generated actions
5. Define the largest `H` whose errors stay below thresholds.

Useful formula for adaptive planning:

```text
H_phase = min(H_required_by_geometry, H_reliable_by_WM)
```

If:

```text
H_required_by_geometry > H_reliable_by_WM
```

then insert more subgoals instead of forcing long-horizon planning.

## Recommended Thesis Framing

A strong narrative:

1. DINO-WM can model parking dynamics locally around valid trajectories.
2. Uninformed action search fails because parking has a narrow feasible action manifold.
3. Hierarchical subgoals keep each phase within the WM's reliable prediction range.
4. Demonstration-guided residual MPPI supplies a plausible action prior.
5. Residual optimization still uses WM planning, so it is not merely replaying expert actions.
6. Future non-oracle versions can use retrieval or BC priors instead of current episode expert traces.

Suggested method statement:

> We propose a hierarchical demonstration-guided residual planning framework for autonomous parking. The long-horizon parking task is decomposed into short phases bounded by subgoals. For each phase, a demonstration prior initializes a residual MPPI planner, which optimizes bounded action residuals under the learned world model. This combines the stability of demonstration priors with the flexibility of model-based planning.

## Important Current Caveats

- Current guided prior source is still current episode expert trace.
- This is acceptable for debugging and upper-bound experiments.
- For a defensible paper experiment, replace it with:
  - training-set retrieval prior, or
  - BC policy prior.
- Current subgoals may still come from expert trajectory. That can also be considered oracle-like depending on the final paper framing.
- If presenting as an upper-bound/analysis study, be explicit.

## Useful Run Command

Typical guided run:

```bash
env WANDB_MODE=disabled /root/miniconda3/envs/dino_wm/bin/python plan_park_3_guided.py
```

Fast smoke run:

```bash
env WANDB_MODE=disabled /root/miniconda3/envs/dino_wm/bin/python plan_park_3_guided.py \
  hydra.run.dir=plan_outputs/guided_full_smoke \
  planner.sub_planner.num_samples=32 \
  planner.sub_planner.opt_steps=1 \
  planner.sub_planner.eval_batch_size=32
```

Expected key outputs:

```text
subgoal_plan.png
mpc_phase0_plan_0_output_1.png
mpc_phase0_plan_0_output_2.png
...
mpc_phase7_merged_final_0_success.mp4
```

