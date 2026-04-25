import subprocess

from sideaware.presets import build_single_side_presets

CONFIG = "sim_hand_f/sideaware_fc.yml"


def main():
    presets = build_single_side_presets()
    for preset in presets:
        cmd = [
            "python3",
            "example_grasp/plan_batch_env.py",
            "-c",
            CONFIG,
            "-w",
            "1",
            "-m",
            "npy",
            "--sideaware_preset",
            preset["name"],
        ]
        print("=" * 80)
        print("Running preset:", preset["name"])
        print("Command:", " ".join(cmd))
        print("=" * 80)
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
