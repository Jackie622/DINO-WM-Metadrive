"""Verify WM reconstruction of phase2_obs_0.

Usage:
    CUDA_VISIBLE_DEVICES=0 python debug_wm_recon.py
"""
import os, sys, cv2
import torch
import numpy as np
import hydra
from pathlib import Path
from omegaconf import OmegaConf, open_dict

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import cfg_to_dict, seed
from preprocessor import Preprocessor

RENDER_RES = (672, 672)
REAL_RES = (224, 224)
BEV_SCALING = 24.0
MAP_CENTER = (29.0, 1.0)


# ========== copy the bits we need from plan_park_3.py ==========

def load_ckpt(snapshot_path, device):
    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location=device, weights_only=False)
    ALL_MODEL_KEYS = ["encoder", "predictor", "decoder", "proprio_encoder", "action_encoder"]
    result = {k: v.to(device) for k, v in payload.items() if k in ALL_MODEL_KEYS}
    result["epoch"] = payload["epoch"]
    return result


def load_model(model_ckpt, train_cfg, num_action_repeat, device):
    result = {}
    if model_ckpt.exists():
        result = load_ckpt(model_ckpt, device)
        print(f"Resuming from epoch {result['epoch']}: {model_ckpt}")

    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(train_cfg.encoder)
    if "predictor" not in result:
        raise ValueError("Predictor not found in model checkpoint")

    if train_cfg.has_decoder and "decoder" not in result:
        base_path = os.path.dirname(os.path.abspath(__file__))
        if train_cfg.env.decoder_path is not None:
            decoder_path = os.path.join(base_path, train_cfg.env.decoder_path)
            ckpt = torch.load(decoder_path)
            result["decoder"] = ckpt["decoder"] if isinstance(ckpt, dict) else torch.load(decoder_path)
        else:
            raise ValueError("Decoder path not found")
    elif not train_cfg.has_decoder:
        result["decoder"] = None

    model = hydra.utils.instantiate(
        train_cfg.model,
        encoder=result["encoder"],
        proprio_encoder=result["proprio_encoder"],
        action_encoder=result["action_encoder"],
        predictor=result["predictor"],
        decoder=result["decoder"],
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
        concat_dim=train_cfg.concat_dim,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=train_cfg.num_proprio_repeat,
    )
    model.to(device)
    model.eval()
    return model


# ========== main ==========

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load model config & checkpoint
    ckpt_base_path = "/root/DINO-WM-Metadrive/checkpoints"
    model_name = "metadrive_park_3"
    model_path = f"{ckpt_base_path}/outputs/{model_name}/"
    with open(os.path.join(model_path, "hydra.yaml"), "r") as f:
        model_cfg = OmegaConf.load(f)

    num_action_repeat = model_cfg.num_action_repeat
    model_ckpt = Path(model_path) / "checkpoints" / "model_100.pth"
    wm = load_model(model_ckpt, model_cfg, num_action_repeat, device=device)
    print("WM loaded successfully.")

    # 2. Load dataset to get preprocessor stats & transform
    datasets, traj_dsets = hydra.utils.call(
        model_cfg.env.dataset, num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred, frameskip=model_cfg.frameskip
    )
    dset = traj_dsets["valid"]
    real_dset = datasets["train"]
    dset.action_mean = real_dset.action_mean
    dset.action_std = real_dset.action_std
    dset.state_mean = real_dset.state_mean
    dset.state_std = real_dset.state_std
    dset.proprio_mean = real_dset.proprio_mean
    dset.proprio_std = real_dset.proprio_std

    preprocessor = Preprocessor(
        action_mean=dset.raw_action_mean, action_std=dset.raw_action_std,
        state_mean=dset.state_mean, state_std=dset.state_std,
        proprio_mean=dset.proprio_mean, proprio_std=dset.proprio_std,
        transform=dset.transform,
    )
    print("Preprocessor ready.")

    # 3. Create environment (single env)
    from plan_park_3 import ParkingDinoWrapper
    from metadrive.envs.marl_envs import MultiAgentParkingLotEnv

    frameskip = model_cfg.frameskip
    act_mean = dset.raw_action_mean.numpy()
    act_std = dset.raw_action_std.numpy()

    def create_env():
        env_config = {
            "use_render": False, "num_agents": 1, "start_seed": 400,
            "allow_respawn": False, "window_size": RENDER_RES,
            "out_of_road_done": False, "crash_vehicle_done": False,
            "vehicle_config": {"lidar": {"num_lasers": 0}, "show_navi_mark": False},
        }
        env = MultiAgentParkingLotEnv(env_config)
        return ParkingDinoWrapper(env, frameskip=frameskip, action_mean=act_mean, action_std=act_std)

    # Use single env directly (not SubprocVectorEnv) for simplicity
    env = create_env()
    print("Environment ready.")

    # 4. Get a real Phase 1 endpoint (setup_pose) and goal via sample_random_init_goal_states
    seed_val = 55
    print(f"\n=== Running sample_random_init_goal_states with seed={seed_val} ===")
    obs_init, obs_subgoal, obs_goal = env.sample_random_init_goal_states(seed_val)

    # Phase 1 MPC endpoint = sub_goal pose (setup pose, near the parking slot)
    start_pose = obs_subgoal['raw_pose']  # (3,) numpy
    goal_pose = obs_goal['raw_pose']      # (3,) numpy
    print(f"Phase 1 endpoint (setup_pose): x={start_pose[0]:.3f}, y={start_pose[1]:.3f}, h={start_pose[2]:.3f}")
    print(f"Goal (slot):                  x={goal_pose[0]:.3f}, y={goal_pose[1]:.3f}, h={goal_pose[2]:.3f}")

    # 5. Run generate_phase2_wm_start with real poses
    print(f"\n=== Running generate_phase2_wm_start ===")
    phase2_obs_0, expert_init_actions = env.generate_phase2_wm_start(seed_val, start_pose, goal_pose)

    print(f"phase2_obs_0 keys: {list(phase2_obs_0.keys())}")
    print(f"  visual shape: {phase2_obs_0['visual'].shape}  (expect t=3, H=224, W=224, C=3)")
    print(f"  proprio shape: {phase2_obs_0['proprio'].shape}")
    print(f"expert_init_actions shape: {expert_init_actions.shape}")

    # 5. Add batch dim -> (B=1, t, H, W, C)
    obs_batched = {}
    for k, v in phase2_obs_0.items():
        obs_batched[k] = np.expand_dims(v, axis=0)
    print(f"After expand_dims: visual shape = {obs_batched['visual'].shape}")

    # 6. Transform obs -> (B, t, C, H, W) in [-1, 1], then resized for encoder
    trans_obs = preprocessor.transform_obs(obs_batched)
    trans_obs = {k: v.to(device) for k, v in trans_obs.items()}
    print(f"After transform: visual shape = {trans_obs['visual'].shape}")

    # 7. Encode & decode
    with torch.no_grad():
        z = wm.encode_obs(trans_obs)
        print(f"Latent z['visual'] shape: {z['visual'].shape}")

        recon_dict, diff = wm.decode_obs(z)
        recon_vis = recon_dict["visual"]  # (B, t, C, H, W) in [-1, 1]
        print(f"Recon visual shape: {recon_vis.shape}")

    # 8. Save images
    os.makedirs("debug_output", exist_ok=True)

    # Save original frames
    orig_vis = obs_batched["visual"]  # (1, 3, 224, 224, 3)
    for t_idx in range(orig_vis.shape[1]):
        img = orig_vis[0, t_idx]  # (224, 224, 3) RGB
        cv2.imwrite(f"debug_output/orig_t{t_idx}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"Saved original frame {t_idx}")

    # Save reconstructed frames (denormalize from [-1,1] to [0,255])
    recon_vis_cpu = recon_vis.cpu()  # (1, t, 3, 224, 224) in [-1, 1]
    for t_idx in range(recon_vis_cpu.shape[1]):
        img_tensor = recon_vis_cpu[0, t_idx]  # (3, 224, 224) in [-1, 1]
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255.0  # -> (H, W, C) in [0,255]
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        cv2.imwrite(f"debug_output/recon_t{t_idx}.png", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        print(f"Saved reconstructed frame {t_idx}")

    # Also save side-by-side comparison
    for t_idx in range(min(orig_vis.shape[1], recon_vis_cpu.shape[1])):
        orig = orig_vis[0, t_idx]
        recon = (recon_vis_cpu[0, t_idx].permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255.0
        recon = np.clip(recon, 0, 255).astype(np.uint8)
        compare = np.concatenate([orig, recon], axis=1)  # side by side
        cv2.imwrite(f"debug_output/compare_t{t_idx}.png", cv2.cvtColor(compare, cv2.COLOR_RGB2BGR))
        print(f"Saved comparison frame {t_idx}")

    # If decoder returned diff, check it
    if diff is not None:
        print(f"Decode diff: {diff.item():.6f}")

    print("\nDone! Check debug_output/ for the images.")
    print("  - orig_t0.png .. orig_t2.png  = original frames from env")
    print("  - recon_t0.png .. recon_t2.png = WM reconstructed frames")
    print("  - compare_t0.png .. = side-by-side")

    env.close()


if __name__ == "__main__":
    main()
