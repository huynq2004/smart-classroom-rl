# main.py (update)
import argparse
import subprocess, sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, choices=["bdq","noisynet","multihead","arq"], default="bdq")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    if args.algo == "bdq":
        cmd = ["python", "experiments/run_bdq.py"]
    elif args.algo == "noisynet":
        cmd = ["python", "experiments/run_noisynet.py"]
    elif args.algo == "multihead":
        cmd = ["python", "experiments/run_multi_head.py"]
    else:
        cmd = ["python", "experiments/run_arq.py"]

    if args.config:
        cmd += ["--config", args.config]
    sys.exit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
