"""python -m hf_dft {run,calibrate,estimate} job.yaml --out <dir or gs://...>"""
from __future__ import annotations

import argparse
import sys

from . import runner


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hf_dft", description=__doc__)
    ap.add_argument("mode", choices=["run", "calibrate", "estimate"])
    ap.add_argument("job", help="job.yaml (ローカルまたは gs://)")
    ap.add_argument("--out", default="out", help="結果の保存先 (ローカルまたは gs://)")
    ap.add_argument("--machine", default="n2-highmem-32", help="estimate 用のマシンタイプ")
    args = ap.parse_args(argv)

    job = runner.load_job(args.job)
    if args.mode == "estimate":
        print(runner.run_estimate(job, args.machine))
        return 0
    if args.mode == "calibrate":
        runner.run_calibrate(job, args.out)
        return 0
    runner.run_job(job, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
