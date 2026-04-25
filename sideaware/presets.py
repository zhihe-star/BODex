from itertools import combinations
from typing import Dict, List

FINGER_ORDER = ["TH", "FF", "MF", "RF", "LF"]
BASE_FINGERS = ["FF", "MF", "RF", "LF"]
THUMB = "TH"

FINGER_LINK = {
    "TH": "rh_thdistal",
    "FF": "rh_ffdistal",
    "MF": "rh_mfdistal",
    "RF": "rh_rfdistal",
    "LF": "rh_lfdistal",
}

SIDE_ID = {
    "palmar": 0,
    "dorsal": 1,
}


def build_finger_sets(num_fingers: int) -> List[List[str]]:
    if num_fingers not in [2, 3, 4]:
        raise ValueError(f"Unsupported num_fingers: {num_fingers}")

    other_num = num_fingers - 1
    result = []
    for others in combinations(BASE_FINGERS, other_num):
        result.append([THUMB] + list(others))
    return result


def build_contact_points(active_fingers: List[str], side_per_finger: Dict[str, str]) -> List[str]:
    names = []
    for finger in active_fingers:
        link = FINGER_LINK[finger]
        side = side_per_finger[finger]
        idx = SIDE_ID[side]
        names.append(f"{link}/{idx}")
    return names


def build_single_side_presets() -> List[Dict]:
    presets = []
    for n_fingers in [2, 3, 4]:
        for active_fingers in build_finger_sets(n_fingers):
            for side in ["palmar", "dorsal"]:
                side_per_finger = {finger: side for finger in active_fingers}
                preset_name = f"N{n_fingers}_{'_'.join(active_fingers)}_{side}"
                presets.append(
                    {
                        "name": preset_name,
                        "num_fingers": n_fingers,
                        "active_fingers": active_fingers,
                        "side_per_finger": side_per_finger,
                        "contact_points_name": build_contact_points(
                            active_fingers=active_fingers,
                            side_per_finger=side_per_finger,
                        ),
                    }
                )
    return presets


def get_preset_by_name(name: str) -> Dict:
    for preset in build_single_side_presets():
        if preset["name"] == name:
            return preset
    raise ValueError(f"Unknown side-aware preset: {name}")
