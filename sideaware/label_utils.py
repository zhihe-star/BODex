from typing import Dict, List

from .presets import FINGER_ORDER

# hand_f cspace order:
# [rh_FFJ3, rh_FFJ2, rh_FFJ1,
#  rh_MFJ3, rh_MFJ2, rh_MFJ1,
#  rh_RFJ3, rh_RFJ2, rh_RFJ1,
#  rh_LFJ3, rh_LFJ2, rh_LFJ1,
#  rh_THJ4, rh_THJ3, rh_THJ2, rh_THJ1]
FINGER_JOINT_IDS = {
    "FF": [0, 1, 2],
    "MF": [3, 4, 5],
    "RF": [6, 7, 8],
    "LF": [9, 10, 11],
    "TH": [12, 13, 14, 15],
}

Q_OPEN = [0.0] * 16
Q_PALMAR = [
    0.00,
    0.26,
    0.20,
    0.00,
    0.26,
    0.20,
    0.00,
    0.24,
    0.18,
    0.00,
    0.21,
    0.15,
    0.00,
    0.42,
    0.14,
    -0.08,
]
Q_DORSAL = [
    0.00,
    -0.26,
    -0.20,
    0.00,
    -0.26,
    -0.20,
    0.00,
    -0.24,
    -0.18,
    0.00,
    -0.21,
    -0.15,
    0.00,
    -0.42,
    -0.14,
    0.08,
]


def build_sideaware_label(active_fingers: List[str], side_per_finger: Dict[str, str]) -> Dict:
    finger_mask = []
    side_mask = []

    for finger in FINGER_ORDER:
        if finger not in active_fingers:
            finger_mask.append(0)
            side_mask.append(0)
            continue

        finger_mask.append(1)
        side = side_per_finger[finger]
        if side == "palmar":
            side_mask.append(1)
        elif side == "dorsal":
            side_mask.append(-1)
        else:
            raise ValueError(f"Unknown side: {side}")

    return {
        "num_fingers": len(active_fingers),
        "finger_order": FINGER_ORDER,
        "finger_mask": finger_mask,
        "side_mask": side_mask,
        "active_fingers": active_fingers,
        "side_per_finger": side_per_finger,
    }


def build_joint_update_mask(active_fingers: List[str]) -> List[float]:
    joint_mask = [0.0] * len(Q_OPEN)
    for finger in active_fingers:
        for index in FINGER_JOINT_IDS[finger]:
            joint_mask[index] = 1.0
    return joint_mask


def build_side_profile_q_ref(
    active_fingers: List[str],
    side_per_finger: Dict[str, str],
    q_open: List[float] = None,
    q_palmar: List[float] = None,
    q_dorsal: List[float] = None,
) -> List[float]:
    q_open = Q_OPEN if q_open is None else q_open
    q_palmar = Q_PALMAR if q_palmar is None else q_palmar
    q_dorsal = Q_DORSAL if q_dorsal is None else q_dorsal

    q_ref = list(q_open)
    for finger in active_fingers:
        joint_ids = FINGER_JOINT_IDS[finger]
        side = side_per_finger[finger]
        src = q_palmar if side == "palmar" else q_dorsal
        for index in joint_ids:
            q_ref[index] = src[index]

    return q_ref
