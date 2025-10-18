import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, choices=["bdq", "multihead", "arq"], default="bdq")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    if args.algo == "bdq":
        cmd = ["python", "experiments/run_bdq.py"]
        if args.config: cmd += ["--config", args.config]
    elif args.algo == "ppo":
        cmd = ["python", "experiments/run_ppo.py"]
        if args.config: cmd += ["--config", args.config]
    else:
        cmd = ["python", "experiments/run_arq.py"]
        if args.config: cmd += ["--config", args.config]

    sys.exit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
