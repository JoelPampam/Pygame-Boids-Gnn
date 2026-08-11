
import subprocess
import sys

# =========================
# CONFIGURATION
# =========================

NUMBER_OF_RUNS = 1

# The script you want to run
EXPERIMENT_SCRIPT = "Experiments.py"


# =========================
# RUN EXPERIMENTS
# =========================

def main():
    for i in range(1, NUMBER_OF_RUNS + 1):

        print()
        print("=" * 50)
        print(f"Starting experiment {i}/{NUMBER_OF_RUNS}")
        print("=" * 50)

        subprocess.run(
            [sys.executable, EXPERIMENT_SCRIPT],
            check=True
        )

        print(f"Finished experiment {i}/{NUMBER_OF_RUNS}")

    print()
    print("=" * 50)
    print(f"ALL {NUMBER_OF_RUNS} EXPERIMENTS COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()

