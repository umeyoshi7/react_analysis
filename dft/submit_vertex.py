"""job.yaml を GCS に置き、Vertex AI カスタムジョブとして投入する.

    python dft/submit_vertex.py run jobs/my_job.yaml \\
        --project my-project --region asia-northeast1 \\
        --bucket gs://my-bucket/hf-dft \\
        --image asia-northeast1-docker.pkg.dev/my-project/hf-dft/hf-dft:latest \\
        --machine-type n2-highmem-32 --spot

結果は <bucket>/<job_name>/<タイムスタンプ>/ 以下に保存される。校正ファイルは
<bucket>/calibration/ に置き、run ジョブは job.yaml の `calibration:` でそれを指す。
--dry-run を付けると、投入内容と費用の見積もりを表示するだけで何も送らない。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from hf_dft import cost, runner, storage  # noqa: E402


def build_worker_pool(image: str, machine_type: str, args: list[str], threads: int,
                      disk_gb: int) -> list[dict]:
    return [{
        "machine_spec": {"machine_type": machine_type},
        "replica_count": 1,
        "container_spec": {
            "image_uri": image,
            "args": args,
            "env": [{"name": "OMP_NUM_THREADS", "value": str(threads)}],
        },
        "disk_spec": {"boot_disk_type": "pd-ssd", "boot_disk_size_gb": disk_gb},
    }]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["run", "calibrate"])
    ap.add_argument("job", type=Path, help="ローカルの job.yaml")
    ap.add_argument("--project", required=True)
    ap.add_argument("--region", default="asia-northeast1")
    ap.add_argument("--bucket", required=True, help="結果とジョブ定義の置き場 (gs://bucket/prefix)")
    ap.add_argument("--image", required=True, help="Artifact Registry 上の実行イメージ")
    ap.add_argument("--machine-type", default="n2-highmem-32")
    ap.add_argument("--service-account", default=None)
    ap.add_argument("--spot", action="store_true", help="Spot VM を使う (安いが中断されうる。中断時は自動再実行し、計算済みの化合物は飛ばす)")
    ap.add_argument("--timeout-hours", type=float, default=24.0)
    ap.add_argument("--disk-gb", type=int, default=200)
    ap.add_argument("--wait", action="store_true", help="完了まで待つ")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if not storage.is_gcs(a.bucket):
        ap.error("--bucket は gs:// で始めてください")

    job = runner.load_job(str(a.job))
    if a.machine_type not in cost.MACHINES:
        print(f"注意: {a.machine_type} は見積もり表にありません (見積もりは省略)")
    else:
        vcpu = cost.MACHINES[a.machine_type][0]
        if job["resources"].get("threads", vcpu) > vcpu:
            ap.error(f"resources.threads が vCPU 数 ({vcpu}) を超えています")
        job["resources"].setdefault("threads", vcpu)
        if a.mode == "run":
            print(runner.run_estimate(job, a.machine_type))

    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = storage.join(a.bucket, job["job_name"], stamp)
    job_uri = storage.join(base, "job.yaml")
    out_uri = storage.join(base, "out") if a.mode == "run" else storage.join(a.bucket, "calibration")
    args = [a.mode, job_uri, "--out", out_uri]
    pool = build_worker_pool(a.image, a.machine_type, args, int(job["resources"].get("threads", 1)), a.disk_gb)

    print(f"ジョブ定義: {job_uri}\n結果の保存先: {out_uri}\nコンテナ引数: {args}\n"
          f"マシン: {a.machine_type} {'(Spot)' if a.spot else '(オンデマンド)'}")
    if a.dry_run:
        print("--dry-run のため送信しません")
        return 0

    import yaml
    storage.write_text(job_uri, yaml.safe_dump(job, allow_unicode=True, sort_keys=False))

    from google.cloud import aiplatform
    from google.cloud.aiplatform.compat.types import custom_job as gca

    aiplatform.init(project=a.project, location=a.region, staging_bucket=a.bucket)
    cj = aiplatform.CustomJob(
        display_name=f"hf-dft-{a.mode}-{job['job_name']}-{stamp}"[:120],
        worker_pool_specs=pool,
        base_output_dir=base,
        labels={"app": "hf-dft", "mode": a.mode},
    )
    cj.submit(
        service_account=a.service_account,
        timeout=int(a.timeout_hours * 3600),
        restart_job_on_worker_restart=a.spot,
        scheduling_strategy=gca.Scheduling.Strategy.SPOT if a.spot else None,
    )
    print(f"投入しました: {cj.resource_name}")
    print(f"コンソール: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={a.project}")
    if a.wait:
        cj.wait()
    return 0


if __name__ == "__main__":
    sys.exit(main())
