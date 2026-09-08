"""Run one finite independent-reference queue on physical GPU 2 or 3."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--dataset", choices=["pathmnist", "hyperkvasir"], required=True)
    parser.add_argument("--gpu", choices=["2", "3"], required=True)
    args = parser.parse_args()
    batch = args.batch.resolve()
    tasks = range(2, 5) if args.dataset == "pathmnist" else range(2, 11)
    data_root = ("/remote-home/wangbomin/root-migrated/FCL/medminist_data"
                 if args.dataset == "pathmnist" else
                 "/remote-home/wangbomin/root-migrated/FCL/Medmnist_new_local_pgsfedtorch/data/hyperkvasir20")
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED="1",
                       TMPDIR=str(batch / "tmp"), XDG_CACHE_HOME=str(batch / "cache"),
                       TORCH_HOME=str(batch / "cache/torch"),
                       MPLCONFIGDIR=str(batch / "cache/matplotlib"),
                       SWANLAB_LOG_DIR=str(batch / "logs/swanlab"),
                       OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    jobs = [{"task": task, "status": "pending"} for task in tasks]
    status_path = batch / f"queue_{args.dataset}.json"
    def save():
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(dict(dataset=args.dataset, gpu=args.gpu,
                                              updated=datetime.now(timezone.utc).isoformat(),
                                              jobs=jobs), indent=2) + "\n")
        temporary.replace(status_path)
    save()
    for job in jobs:
        task = job["task"]
        output = batch / f"{args.dataset}_task{task:02d}_seed2025"
        logfile = batch / "logs" / f"{args.dataset}_task{task:02d}.log"
        command = [sys.executable, "-u", str(batch / "code/run_independent.py"),
                   "--dataset", args.dataset, "--task", str(task),
                   "--source-root", str(batch / "base_source"),
                   "--data-root", data_root,
                   "--client-index", str(batch / "indexes" / f"{args.dataset}_train_alpha0.3.npy"),
                   "--output", str(output)]
        job.update(status="running", output=str(output), log=str(logfile), command=command)
        save()
        with logfile.open("x") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=environment, cwd=batch / "base_source")
            job["pid"] = process.pid
            save()
            code = process.wait()
        job.update(exit_code=code, status="complete" if code == 0 and
                   (output / "result.json").is_file() else "failed")
        save()
        if job["status"] == "failed":
            raise SystemExit(f"Queue stopped at {args.dataset} task {task}; see {logfile}")


if __name__ == "__main__":
    main()
