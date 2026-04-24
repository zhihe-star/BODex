#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BODex .npy grasp diagnostics for hand_f
Author: ChatGPT

Assumptions tailored to your hand_f setup:
- Finger order: FF, MF, RF, LF, TH
- contact_point tail shape: (..., 5, 3)
- contact_frame tail shape: (..., 5, 3, 3)
- contact_force tail shape: (..., 6, 5, 3)   # default hard-finger contact
- grasp_error tail shape: (..., 6)
- dist_error tail shape: (..., K) or (...,)

This script is designed to be robust to extra leading singleton dims,
such as:
    contact_point: (1, 20, 1, 5, 3)
    contact_frame: (1, 20, 5, 3, 3)
    contact_force: (1, 20, 6, 5, 3)
    grasp_error:   (1, 20, 6)
    dist_error:    (1, 20, 2)
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Any, Dict, List, Tuple

import numpy as np


FINGER_NAMES = ["FF", "MF", "RF", "LF", "TH"]
THUMB_INDEX = 4

# 若你的 wrench 顺序就是常见的 6 个单位力方向，可保留这个标签。
# 若你自己的 task_dict / TWS 不是这个顺序，只需要改这里。
DEFAULT_WRENCH_LABELS_6 = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]


def safe_norm(x: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    return np.linalg.norm(x, axis=axis, keepdims=keepdims) + 1e-12


def unit(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / safe_norm(x, axis=axis, keepdims=True)


def load_npy_dict(path: str) -> Dict[str, Any]:
    obj = np.load(path, allow_pickle=True)

    if isinstance(obj, np.lib.npyio.NpzFile):
        return {k: obj[k] for k in obj.files}

    # 常见情况：0-d object array containing a dict
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        try:
            item = obj.item()
            if isinstance(item, dict):
                return item
        except Exception:
            pass

    if isinstance(obj, dict):
        return obj

    raise ValueError(
        f"无法把文件解析成 dict: {path}\n"
        f"type={type(obj)}, shape={getattr(obj, 'shape', None)}, dtype={getattr(obj, 'dtype', None)}"
    )


def reshape_tail(arr: np.ndarray, tail_shape: Tuple[int, ...], name: str) -> np.ndarray:
    """
    Reshape array to (-1, *tail_shape), requiring that the last len(tail_shape) dims match exactly.
    """
    arr = np.asarray(arr)
    if arr.ndim < len(tail_shape):
        raise ValueError(f"{name} 维度过少: shape={arr.shape}, expected tail={tail_shape}")
    if tuple(arr.shape[-len(tail_shape):]) != tuple(tail_shape):
        raise ValueError(f"{name} 尾部维度不匹配: shape={arr.shape}, expected tail={tail_shape}")
    return arr.reshape(-1, *tail_shape)


def reshape_dist(arr: np.ndarray) -> np.ndarray:
    """
    Dist error may be scalar or vector in the last dim.
    Converts to shape (-1, K), where K>=1.
    """
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr.reshape(-1, arr.shape[-1])


def broadcast_or_fail(arr: np.ndarray, target_n: int, name: str) -> np.ndarray:
    """
    If arr has 1 sample and target_n > 1, repeat it.
    Otherwise require arr.shape[0] == target_n.
    """
    if arr.shape[0] == target_n:
        return arr
    if arr.shape[0] == 1 and target_n > 1:
        reps = [target_n] + [1] * (arr.ndim - 1)
        return np.tile(arr, reps)
    raise ValueError(f"{name} 样本数不一致: got {arr.shape[0]}, expected {target_n}")


def get_wrench_labels(n_wrench: int) -> List[str]:
    if n_wrench == 6:
        return DEFAULT_WRENCH_LABELS_6
    return [f"W{i}" for i in range(n_wrench)]


def extract_arrays(data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    required = ["contact_point", "contact_frame", "contact_force", "grasp_error", "dist_error"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"缺少字段: {missing}. 当前可用字段: {list(data.keys())}")

    cp = reshape_tail(np.asarray(data["contact_point"]), (5, 3), "contact_point")
    cf = reshape_tail(np.asarray(data["contact_frame"]), (5, 3, 3), "contact_frame")
    force = np.asarray(data["contact_force"])
    ge = np.asarray(data["grasp_error"])
    de = reshape_dist(np.asarray(data["dist_error"]))

    # contact_force: (..., W, 5, 3)
    if force.ndim < 3 or force.shape[-2:] != (5, 3):
        raise ValueError(f"contact_force 末尾维度异常: shape={force.shape}, expected (..., W, 5, 3)")
    n_wrench = force.shape[-3]
    force = force.reshape(-1, n_wrench, 5, 3)

    # grasp_error: (..., W)
    if ge.ndim < 1 or ge.shape[-1] != n_wrench:
        raise ValueError(f"grasp_error 末尾维度异常: shape={ge.shape}, expected (..., {n_wrench})")
    ge = ge.reshape(-1, n_wrench)

    n = cp.shape[0]
    cf = broadcast_or_fail(cf, n, "contact_frame")
    force = broadcast_or_fail(force, n, "contact_force")
    ge = broadcast_or_fail(ge, n, "grasp_error")
    de = broadcast_or_fail(de, n, "dist_error")

    return {
        "contact_point": cp,
        "contact_frame": cf,
        "contact_force": force,
        "grasp_error": ge,
        "dist_error": de,
    }


def finger_force_stats(force_local: np.ndarray) -> Dict[str, np.ndarray]:
    """
    force_local: [W, 5, 3]
    3 dims are local contact-frame coefficients:
      [normal_pressure, tangential_1, tangential_2]
    """
    normal = np.abs(force_local[..., 0])              # [W, 5]
    tang = np.linalg.norm(force_local[..., 1:], axis=-1)  # [W, 5]
    total = np.linalg.norm(force_local, axis=-1)      # [W, 5]

    mean_normal = normal.mean(axis=0)                 # [5]
    mean_tang = tang.mean(axis=0)                     # [5]
    mean_total = total.mean(axis=0)                   # [5]

    total_sum = mean_total.sum() + 1e-12
    share = mean_total / total_sum

    tang_ratio = mean_tang / (mean_normal + 1e-12)

    return {
        "mean_normal": mean_normal,
        "mean_tang": mean_tang,
        "mean_total": mean_total,
        "share": share,
        "tang_ratio": tang_ratio,
    }


def detect_inactive_fingers(mean_total: np.ndarray) -> List[int]:
    """
    Relative criterion is more reliable than absolute magnitude,
    because contact-force variables are normalized internal QP variables.
    """
    max_f = float(np.max(mean_total))
    if max_f < 1e-6:
        return list(range(len(mean_total)))

    inactive = []
    for i, v in enumerate(mean_total):
        if v < max(1e-4, 0.15 * max_f):
            inactive.append(i)
    return inactive


def detect_scraping_fingers(tang_ratio: np.ndarray) -> List[int]:
    """
    tang_ratio > 1 means tangential components dominate normal pressure.
    Often indicates side-wall/edge rubbing instead of good fingertip pressing.
    """
    return [i for i, r in enumerate(tang_ratio) if r > 1.0]


def compute_thumb_opposition(
    points: np.ndarray,
    frames: np.ndarray,
    mean_force: np.ndarray,
    share: np.ndarray,
) -> Dict[str, Any]:
    """
    points: [5,3]   object-side contact points
    frames: [5,3,3] local contact frame, columns are [normal, tangent1, tangent2]
    mean_force: [5]
    share: [5]

    Heuristic:
    - thumb must carry non-trivial force share
    - thumb normal should oppose average four-finger normals
    - thumb point should be sufficiently separated from four-finger centroid
    """
    thumb_pt = points[THUMB_INDEX]
    other_pts = points[:THUMB_INDEX]

    thumb_n = unit(frames[THUMB_INDEX, :, 0])
    other_ns = unit(frames[:THUMB_INDEX, :, 0])
    mean_other_n = unit(other_ns.mean(axis=0))

    dot = float(np.dot(thumb_n, mean_other_n))  # negative is better opposition
    thumb_to_centroid = float(np.linalg.norm(thumb_pt - other_pts.mean(axis=0)))

    # scale-free spacing ratio
    other_centroid = other_pts.mean(axis=0)
    other_spread = float(np.mean(np.linalg.norm(other_pts - other_centroid, axis=1)) + 1e-12)
    spacing_ratio = thumb_to_centroid / other_spread if other_spread > 1e-12 else math.inf

    thumb_force = float(mean_force[THUMB_INDEX])
    thumb_share = float(share[THUMB_INDEX])

    # Heuristic scoring
    active = thumb_force > max(1e-4, 0.15 * float(np.max(mean_force)))
    enough_share = thumb_share > 0.10
    good_normal = dot < -0.15
    strong_normal = dot < -0.35
    good_spacing = spacing_ratio > 1.2

    if active and enough_share and strong_normal and good_spacing:
        status = "有效"
    elif active and (good_normal or good_spacing):
        status = "较弱但存在"
    elif active:
        status = "接触到了，但没有形成有效对掌"
    else:
        status = "无效/基本没参与"

    return {
        "status": status,
        "thumb_force": thumb_force,
        "thumb_share": thumb_share,
        "normal_dot": dot,
        "thumb_to_centroid": thumb_to_centroid,
        "spacing_ratio": spacing_ratio,
        "active": active,
    }


def grasp_worst_directions(grasp_error: np.ndarray, topk: int = 3) -> List[Tuple[int, float]]:
    idx = np.argsort(-grasp_error)[:topk]
    return [(int(i), float(grasp_error[i])) for i in idx]


def contact_distribution_comment(shares: np.ndarray) -> str:
    sorted_idx = np.argsort(-shares)
    top1 = float(shares[sorted_idx[0]])
    top2 = float(shares[sorted_idx[0]] + shares[sorted_idx[1]])

    if top1 > 0.60:
        return f"受力严重集中在单指（{FINGER_NAMES[sorted_idx[0]]}）"
    if top2 > 0.80:
        return f"受力主要集中在两指（{FINGER_NAMES[sorted_idx[0]]}, {FINGER_NAMES[sorted_idx[1]]}）"
    return "受力分布相对均衡"


def point_spread_comment(points: np.ndarray) -> str:
    """
    Very rough geometric comment based on object-side contact-point distribution.
    """
    centroid = points.mean(axis=0)
    d = np.linalg.norm(points - centroid, axis=1)
    spread = float(np.mean(d))

    # Thumb relative position
    thumb_d = float(np.linalg.norm(points[THUMB_INDEX] - centroid))
    others_mean = float(np.mean(d[:THUMB_INDEX]))

    if spread < 1e-3:
        return "5个接触点非常集中，可能都挤在同一区域"
    if thumb_d < 0.6 * max(others_mean, 1e-12):
        return "拇指接触点离整体接触簇不够远，拇指对侧分布偏弱"
    return "接触点分布看起来有一定包裹性"


def dist_comment(dist_vec: np.ndarray) -> str:
    vals = ", ".join(f"{v:.6f}" for v in dist_vec.tolist())
    if np.all(np.abs(dist_vec) < 0.003):
        return f"dist_error 很小 [{vals}]，几何接触总体较好"
    if np.max(np.abs(dist_vec)) > 0.01:
        return f"dist_error 偏大 [{vals}]，几何接触仍不理想"
    return f"dist_error 中等 [{vals}]"


def candidate_score(grasp_error: np.ndarray, dist_vec: np.ndarray) -> float:
    """
    Lower is better. Used only to choose a candidate to inspect first.
    """
    return float(np.max(grasp_error) + 0.5 * np.mean(np.abs(dist_vec)))


def analyze_candidate(
    idx: int,
    points: np.ndarray,
    frames: np.ndarray,
    forces: np.ndarray,
    grasp_error: np.ndarray,
    dist_vec: np.ndarray,
    wrench_labels: List[str],
) -> Dict[str, Any]:
    stats = finger_force_stats(forces)
    inactive = detect_inactive_fingers(stats["mean_total"])
    scraping = detect_scraping_fingers(stats["tang_ratio"])
    thumb = compute_thumb_opposition(points, frames, stats["mean_total"], stats["share"])
    worst = grasp_worst_directions(grasp_error, topk=min(3, len(grasp_error)))

    worst_labels = [f"{wrench_labels[i]}({v:.4f})" for i, v in worst]
    inactive_names = [FINGER_NAMES[i] for i in inactive]
    scraping_names = [FINGER_NAMES[i] for i in scraping]

    return {
        "idx": idx,
        "score": candidate_score(grasp_error, dist_vec),
        "inactive": inactive_names,
        "scraping": scraping_names,
        "thumb": thumb,
        "worst": worst,
        "worst_labels": worst_labels,
        "mean_total": stats["mean_total"],
        "share": stats["share"],
        "tang_ratio": stats["tang_ratio"],
        "grasp_error": grasp_error,
        "dist_vec": dist_vec,
        "distribution_comment": contact_distribution_comment(stats["share"]),
        "point_comment": point_spread_comment(points),
        "dist_comment": dist_comment(dist_vec),
    }


def print_candidate_report(rep: Dict[str, Any]) -> None:
    idx = rep["idx"]
    print("=" * 88)
    print(f"Candidate #{idx:02d}")
    print(f"综合排查分数（越小越好）: {rep['score']:.6f}")
    print(f"最差 wrench 方向: {', '.join(rep['worst_labels'])}")
    print(f"grasp_error 全部方向: {np.array2string(rep['grasp_error'], precision=5, suppress_small=False)}")
    print(f"dist_error: {np.array2string(rep['dist_vec'], precision=6, suppress_small=False)}")
    print(f"几何结论: {rep['dist_comment']}")
    print(f"接触点分布: {rep['point_comment']}")
    print(f"受力分布: {rep['distribution_comment']}")

    print("\n每根手指平均受力（局部 contact-frame 系数范数）:")
    for name, f, s, tr in zip(FINGER_NAMES, rep["mean_total"], rep["share"], rep["tang_ratio"]):
        print(f"  {name}: mean_force={f:.6f}, share={100.0*s:6.2f}%, tang/normal={tr:.3f}")

    inactive_msg = "无" if not rep["inactive"] else ", ".join(rep["inactive"])
    scraping_msg = "无" if not rep["scraping"] else ", ".join(rep["scraping"])

    print(f"\n哪根手指没参与: {inactive_msg}")
    print(f"疑似擦碰型接触: {scraping_msg}")

    th = rep["thumb"]
    print(
        "\n拇指是否有效对掌: "
        f"{th['status']} | "
        f"thumb_share={100.0*th['thumb_share']:.2f}%, "
        f"thumb_force={th['thumb_force']:.6f}, "
        f"thumb_normal·mean_other_normals={th['normal_dot']:.4f}, "
        f"thumb_spacing_ratio={th['spacing_ratio']:.3f}"
    )

    print("\n简要结论:")
    if rep["inactive"]:
        print(f"  - 基本没参与的手指: {inactive_msg}")
    else:
        print("  - 5根手指都在一定程度参与。")

    if rep["scraping"]:
        print(f"  - 这些手指切向分量偏强，像是在侧擦而不是正压: {scraping_msg}")
    else:
        print("  - 没有明显的擦碰型接触。")

    print(f"  - 最差 wrench 方向: {', '.join(rep['worst_labels'])}")
    print(f"  - 拇指对掌判断: {th['status']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze BODex grasp .npy for hand_f")
    parser.add_argument("npy_path", type=str, help="Path to .npy result file")
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Print top-k candidates ranked by (max grasp_error + 0.5 * mean|dist_error|)"
    )
    parser.add_argument(
        "--candidate",
        type=int,
        default=None,
        help="Print only one candidate index"
    )
    args = parser.parse_args()

    if not os.path.exists(args.npy_path):
        raise FileNotFoundError(args.npy_path)

    data = load_npy_dict(args.npy_path)
    arrs = extract_arrays(data)

    cp = arrs["contact_point"]     # [N, 5, 3]
    cf = arrs["contact_frame"]     # [N, 5, 3, 3]
    force = arrs["contact_force"]  # [N, W, 5, 3]
    ge = arrs["grasp_error"]       # [N, W]
    de = arrs["dist_error"]        # [N, K]

    n_candidates = cp.shape[0]
    n_wrench = ge.shape[1]
    wrench_labels = get_wrench_labels(n_wrench)

    reports = []
    for i in range(n_candidates):
        rep = analyze_candidate(
            idx=i,
            points=cp[i],
            frames=cf[i],
            forces=force[i],
            grasp_error=ge[i],
            dist_vec=de[i],
            wrench_labels=wrench_labels,
        )
        reports.append(rep)

    reports.sort(key=lambda r: r["score"])

    print("#" * 88)
    print(f"文件: {args.npy_path}")
    print(f"共解析出 {n_candidates} 个 candidate")
    print(f"wrench 数量: {n_wrench} -> labels: {wrench_labels}")
    print(f"手指顺序固定按 hand_f: {FINGER_NAMES}")
    print("#" * 88)

    if args.candidate is not None:
        match = [r for r in reports if r["idx"] == args.candidate]
        if not match:
            raise IndexError(f"candidate={args.candidate} 不存在，合法范围: [0, {n_candidates-1}]")
        print_candidate_report(match[0])
        return

    print("\n建议优先查看的 candidate 排名（越靠前越值得先看）:")
    for r in reports[: min(args.topk, len(reports))]:
        print(
            f"  #{r['idx']:02d} | score={r['score']:.6f} | "
            f"worst={r['worst_labels'][0]} | thumb={r['thumb']['status']} | "
            f"inactive={','.join(r['inactive']) if r['inactive'] else 'None'}"
        )

    print("\n详细报告:")
    for r in reports[: min(args.topk, len(reports))]:
        print_candidate_report(r)


if __name__ == "__main__":
    main()