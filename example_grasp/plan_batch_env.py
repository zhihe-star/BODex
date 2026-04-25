# Standard Library
import time
from typing import Dict, List
import datetime
import os
from copy import deepcopy

# Third Party
import torch
import numpy as np
import argparse

# CuRobo
import curobo.geom.sdf.world
from curobo.wrap.reacher.grasp_solver import GraspSolver, GraspSolverConfig
from curobo.util.world_cfg_generator import get_world_config_dataloader
from curobo.util.logger import setup_logger, log_warn
from curobo.util.save_helper import SaveHelper
from curobo.util_file import (
    get_manip_configs_path,
    join_path,
    load_yaml,
)
from sideaware.presets import (
    FINGER_ORDER,
    build_contact_points,
    build_finger_sets,
    get_preset_by_name,
)
from sideaware.label_utils import (
    build_joint_update_mask,
    build_side_profile_q_ref,
    build_sideaware_label,
)

torch.backends.cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import numpy as np
import random

seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)


def process_grasp_result(result, save_debug, save_data, save_id):
    traj = result.debug_info["solver"]["steps"][0]
    all_traj = torch.cat(traj, dim=1)  # [b*n, h, q]
    batch, horizon = all_traj.shape[:2]

    if save_data == "all":
        select_horizon_lst = list(range(0, horizon))
    elif "select_" in save_data:
        part_num = int(save_data.split("select_")[-1])
        select_horizon_lst = list(range(0, horizon, horizon // (part_num - 1)))
        select_horizon_lst[-1] = horizon - 1
    elif save_data == "init":
        select_horizon_lst = [0]
    elif save_data == "final" or save_data == "final_and_mid":
        select_horizon_lst = [-1]
    else:
        raise NotImplementedError

    if save_id is None:
        save_id_lst = list(range(0, batch))
    elif isinstance(save_id, List):
        save_id_lst = save_id
    else:
        raise NotImplementedError

    save_traj = all_traj[:, select_horizon_lst]
    save_traj = save_traj[save_id_lst, :]

    if save_debug:
        n_num = torch.stack(result.debug_info["solver"]["hp"][0]).shape[-2]
        o_num = torch.stack(result.debug_info["solver"]["op"][0]).shape[-2]
        hp_traj = torch.stack(result.debug_info["solver"]["hp"][0], dim=1).view(-1, n_num, 3)
        grad_traj = torch.stack(result.debug_info["solver"]["grad"][0], dim=1).view(-1, n_num, 3)
        op_traj = torch.stack(result.debug_info["solver"]["op"][0], dim=1).view(-1, o_num, 3)
        posi_traj = torch.stack(result.debug_info["solver"]["debug_posi"][0], dim=1).view(
            -1, o_num, 3
        )
        normal_traj = torch.stack(result.debug_info["solver"]["debug_normal"][0], dim=1).view(
            -1, o_num, 3
        )

        debug_info = {
            "hp": hp_traj,
            "grad": grad_traj * 100,
            "op": op_traj,
            "debug_posi": posi_traj,
            "debug_normal": normal_traj,
        }

        for k, v in debug_info.items():
            debug_info[k] = v.view((all_traj.shape[0], -1) + v.shape[1:])[:, select_horizon_lst]
            debug_info[k] = debug_info[k][save_id_lst, :]
            debug_info[k] = debug_info[k].view((-1,) + v.shape[1:])
    else:
        debug_info = None
        if save_data == "final_and_mid":
            mid_robot_pose = torch.cat(result.debug_info["solver"]["mid_result"][0], dim=1)
            mid_robot_pose = mid_robot_pose[save_id_lst, :]
            save_traj = torch.cat([mid_robot_pose, save_traj], dim=-2)

    return save_traj, debug_info


def _normalize_finger_set(finger_set: List[str]) -> List[str]:
    if finger_set is None:
        return None
    normalized = [finger.upper() for finger in finger_set]
    invalid = [finger for finger in normalized if finger not in FINGER_ORDER]
    if invalid:
        raise ValueError(f"Unknown finger in --finger_set: {invalid}")
    return normalized


def _build_sideaware_spec(args, manip_config_data: Dict):
    from_cli = (
        args.sideaware_preset is not None
        or args.finger_set is not None
        or args.num_fingers is not None
    )
    from_cfg = (
        "sideaware" in manip_config_data
        and isinstance(manip_config_data["sideaware"], Dict)
        and manip_config_data["sideaware"].get("enabled", False)
    )
    if not from_cli and not from_cfg:
        return None

    if args.sideaware_preset is not None:
        preset = get_preset_by_name(args.sideaware_preset)
        active_fingers = preset["active_fingers"]
        side_per_finger = preset["side_per_finger"]
        preset_name = preset["name"]
    elif from_cli:
        if args.finger_set is None:
            n_fingers = 3 if args.num_fingers is None else args.num_fingers
            active_fingers = build_finger_sets(n_fingers)[0]
        else:
            active_fingers = _normalize_finger_set(args.finger_set)
        side_per_finger = {finger: args.side for finger in active_fingers}
        preset_name = f"N{len(active_fingers)}_{'_'.join(active_fingers)}_{args.side}"
    else:
        cfg_sideaware = manip_config_data["sideaware"]
        active_fingers = _normalize_finger_set(cfg_sideaware.get("active_fingers", []))
        side_per_finger = {
            finger.upper(): side for finger, side in cfg_sideaware.get("side_per_finger", {}).items()
        }
        if not active_fingers:
            raise ValueError("sideaware.enabled=True but no active_fingers in config.")
        for finger in active_fingers:
            if finger not in side_per_finger:
                raise ValueError(f"Missing side assignment for finger: {finger}")
        preset_name = cfg_sideaware.get(
            "preset_name", f"N{len(active_fingers)}_{'_'.join(active_fingers)}_custom"
        )

    contact_points_name = build_contact_points(active_fingers, side_per_finger)
    label = build_sideaware_label(active_fingers, side_per_finger)
    q_ref = build_side_profile_q_ref(active_fingers, side_per_finger)
    joint_update_mask = build_joint_update_mask(active_fingers)

    return {
        "enabled": True,
        "preset_name": preset_name,
        "num_fingers": len(active_fingers),
        "active_fingers": active_fingers,
        "side_per_finger": side_per_finger,
        "contact_points_name": contact_points_name,
        "label": label,
        "q_ref": q_ref,
        "joint_update_mask": joint_update_mask,
    }


def apply_sideaware_overrides(manip_config_data: Dict, args):
    sideaware_spec = _build_sideaware_spec(args, manip_config_data)
    if sideaware_spec is None:
        return manip_config_data, None

    manip_config_data["grasp_contact_strategy"]["contact_points_name"] = sideaware_spec[
        "contact_points_name"
    ]
    manip_config_data["seeder_cfg"]["q"] = sideaware_spec["q_ref"]
    manip_config_data["mogen_init"] = sideaware_spec["q_ref"]
    manip_config_data["sideaware"] = {
        "enabled": True,
        "preset_name": sideaware_spec["preset_name"],
        "num_fingers": sideaware_spec["num_fingers"],
        "active_fingers": sideaware_spec["active_fingers"],
        "side_per_finger": sideaware_spec["side_per_finger"],
        "contact_points_name": sideaware_spec["contact_points_name"],
        "joint_update_mask": sideaware_spec["joint_update_mask"],
        **sideaware_spec["label"],
    }

    old_exp_name = manip_config_data.get("exp_name", None)
    if old_exp_name:
        manip_config_data["exp_name"] = f"{old_exp_name}_{sideaware_spec['preset_name']}"
    else:
        manip_config_data["exp_name"] = f"sideaware_{sideaware_spec['preset_name']}"

    return manip_config_data, manip_config_data["sideaware"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--manip_cfg_file",
        type=str,
        default="fc_leap.yml",
        help="config file path",
    )

    parser.add_argument(
        "-f",
        "--save_folder",
        type=str,
        default=None,
        help="If None, use join_path(manip_cfg_file[:-4], $TIME) as save_folder",
    )

    parser.add_argument(
        "-m",
        "--save_mode",
        choices=["usd", "npy", "usd+npy", "none"],
        default="npy",
        help="Method to save results",
    )

    parser.add_argument(
        "-d",
        "--save_data",
        # choices=['all', 'final', 'final_and_mid', 'init', 'select_{$INT}'],
        default="final_and_mid",
        help="Which results to save",
    )

    parser.add_argument(
        "-i",
        "--save_id",
        type=int,
        nargs="+",
        default=None,
        help="Which results to save",
    )

    parser.add_argument(
        "-debug",
        "--save_debug",
        action="store_true",
        help="Which to save contact normal for debug",
    )

    parser.add_argument(
        "-w",
        "--parallel_world",
        type=int,
        default=20,
        help="parallel world num.",
    )
    parser.add_argument(
        "--sideaware_preset",
        type=str,
        default=None,
        help="Preset name from sideaware/presets.py, e.g. N3_TH_FF_MF_palmar",
    )
    parser.add_argument(
        "--num_fingers",
        type=int,
        choices=[2, 3, 4],
        default=None,
        help="Used when --sideaware_preset is None and --finger_set is not provided.",
    )
    parser.add_argument(
        "--side",
        type=str,
        choices=["palmar", "dorsal"],
        default="palmar",
        help="Used when --sideaware_preset is None.",
    )
    parser.add_argument(
        "--finger_set",
        nargs="+",
        default=None,
        help="Optional finger subset, e.g. --finger_set TH FF MF",
    )

    parser.add_argument(
        "-k",
        "--skip",
        action="store_false",
        help="If True, skip existing files. (default: True)",
    )

    setup_logger("warn")

    args = parser.parse_args()
    manip_config_data = load_yaml(join_path(get_manip_configs_path(), args.manip_cfg_file))
    manip_config_data, sideaware_meta = apply_sideaware_overrides(manip_config_data, args)

    world_generator = get_world_config_dataloader(manip_config_data["world"], args.parallel_world)

    if args.save_folder is not None:
        save_folder = os.path.join(args.save_folder, "graspdata")
    elif manip_config_data["exp_name"] is not None:
        save_folder = os.path.join(
            args.manip_cfg_file[:-4], manip_config_data["exp_name"], "graspdata"
        )
    else:
        save_folder = os.path.join(
            args.manip_cfg_file[:-4],
            datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
            "graspdata",
        )

    save_helper = SaveHelper(
        robot_file=manip_config_data["robot_file"],
        save_folder=save_folder,
        task_name="grasp",
        mode=args.save_mode,
    )
    tst = time.time()
    grasp_solver = None
    for world_info_dict in world_generator:
        sst = time.time()
        if args.skip and save_helper.exist_piece(world_info_dict["save_prefix"]):
            log_warn(f"skip {world_info_dict['save_prefix']}")
            continue

        if grasp_solver is None:
            grasp_config = GraspSolverConfig.load_from_robot_config(
                world_model=world_info_dict["world_cfg"],
                manip_name_list=world_info_dict["manip_name"],
                manip_config_data=manip_config_data,
                obj_gravity_center=world_info_dict["obj_gravity_center"],
                obj_obb_length=world_info_dict["obj_obb_length"],
                use_cuda_graph=False,
                store_debug=args.save_debug,
            )
            grasp_solver = GraspSolver(grasp_config)
            world_info_dict["world_model"] = grasp_solver.world_coll_checker.world_model
        else:
            world_info_dict["world_model"] = world_model = [
                curobo.geom.sdf.world.WorldConfig.from_dict(world_cfg) for world_cfg in world_info_dict["world_cfg"]
            ]
            grasp_solver.update_world(
                world_model,
                world_info_dict["obj_gravity_center"],
                world_info_dict["obj_obb_length"],
                world_info_dict["manip_name"],
            )

        result = grasp_solver.solve_batch_env(return_seeds=grasp_solver.num_seeds)

        if args.save_debug:
            robot_pose, debug_info = process_grasp_result(
                result, args.save_debug, args.save_data, args.save_id
            )
            world_info_dict["debug_info"] = debug_info
            world_info_dict["robot_pose"] = robot_pose.reshape(
                (len(world_info_dict["world_model"]), -1) + robot_pose.shape[1:]
            )
        else:
            squeeze_pose_qpos = torch.cat(
                [
                    result.solution[..., 1, :7],
                    result.solution[..., 1, 7:] * 2 - result.solution[..., 0, 7:],
                ],
                dim=-1,
            )
            all_hand_pose_qpos = torch.cat(
                [result.solution, squeeze_pose_qpos.unsqueeze(-2)], dim=-2
            )
            world_info_dict["robot_pose"] = all_hand_pose_qpos
            world_info_dict["contact_point"] = result.contact_point
            world_info_dict["contact_frame"] = result.contact_frame
            world_info_dict["contact_force"] = result.contact_force
            world_info_dict["grasp_error"] = result.grasp_error
            world_info_dict["dist_error"] = result.dist_error
        if sideaware_meta is not None:
            world_info_dict["sideaware"] = [
                deepcopy(sideaware_meta) for _ in range(len(world_info_dict["save_prefix"]))
            ]
        log_warn(f"Sinlge Time: {time.time()-sst}")
        save_helper.save_piece(world_info_dict)

    log_warn(f"Total Time: {time.time()-tst}")
