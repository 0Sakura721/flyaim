"""A 线内部工具:从 Google Storage 公桶流式下载 MaleCNS v1.0 数据。

特性:
- 断点续传(HTTP Range),失败自动重试并退避
- 流式写盘,不占内存
- 记录 bytes / sha256 / 耗时 / 重试次数 -> .cache/download_report.json

不是交付产物,只是构建期脚本(放在 flyaim/data/ 写范围内)。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
CACHE = Path(r"D:\dsh\autoaim\.cache")

TARGETS = [
    {
        "key": "annotations",
        "file": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        "required": True,
    },
    {
        "key": "weights",
        "file": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        "required": True,
    },
    {
        "key": "neurotransmitters",
        "file": "body-neurotransmitters-male-cns-v1.0.feather",
        "required": False,
    },
    {
        "key": "body_stats",
        "file": "body-stats-male-cns-v1.0-minconf-0.5.feather",
        "required": False,
    },
    {
        "key": "weights_traced_only",
        "file": "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather",
        "required": False,
    },
    {
        "key": "weights_significant_only",
        "file": "connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather",
        "required": False,
    },
]

CHUNK = 4 * 1024 * 1024
MAX_ATTEMPTS = 12
READ_TIMEOUT = 90.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=READ_TIMEOUT) as r:
            cl = r.headers.get("Content-Length")
            return int(cl) if cl else None
    except Exception as e:  # noqa: BLE001
        log(f"  HEAD 失败: {e}")
        return None


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download_one(t: dict) -> dict:
    url = f"{BASE}/{t['file']}"
    final = CACHE / t["file"]
    part = CACHE / (t["file"] + ".part")
    CACHE.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    total = remote_size(url)
    log(f"开始 {t['key']}: {t['file']}  远端大小={total}")

    if final.exists() and total is not None and final.stat().st_size == total:
        log(f"  已存在且大小匹配,跳过下载")
        return {
            "key": t["key"],
            "url": url,
            "path": str(final),
            "bytes": final.stat().st_size,
            "sha256": sha256_of(final),
            "seconds": round(time.time() - t0, 2),
            "attempts": 0,
            "status": "cached",
        }

    attempts = 0
    last_err = ""
    while attempts < MAX_ATTEMPTS:
        attempts += 1
        have = part.stat().st_size if part.exists() else 0
        if total is not None and have == total:
            break
        headers = {"User-Agent": "FlyAim/0.1 (research)"}
        mode = "wb"
        if have > 0:
            headers["Range"] = f"bytes={have}-"
            mode = "ab"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=READ_TIMEOUT) as r:
                # 服务器不支持 Range 时会返回 200,必须从头写
                if have > 0 and r.status == 200:
                    log("  服务器忽略 Range,从头重新下载")
                    have = 0
                    mode = "wb"
                if total is None:
                    cl = r.headers.get("Content-Length")
                    if cl:
                        total = int(cl) + have
                last_log = time.time()
                with open(part, mode) as f:
                    while True:
                        chunk = r.read(CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)
                        if time.time() - last_log > 15:
                            last_log = time.time()
                            pct = f"{100.0 * have / total:5.1f}%" if total else "  ?  "
                            log(f"  {t['key']}: {have/1e6:8.1f} MB / "
                                f"{'-' if total is None else f'{total/1e6:.1f}'} MB  {pct}"
                                f"  {have/1e6/max(time.time()-t0,1e-9):5.1f} MB/s")
            if total is None or part.stat().st_size == total:
                break
            log(f"  连接中断: {part.stat().st_size}/{total},重试 {attempts}")
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            log(f"  第 {attempts} 次尝试失败: {last_err};10s 后重试")
            time.sleep(10)
    else:
        log(f"  放弃 {t['key']}(超过 {MAX_ATTEMPTS} 次)")
        return {
            "key": t["key"],
            "url": url,
            "path": None,
            "bytes": part.stat().st_size if part.exists() else 0,
            "sha256": None,
            "seconds": round(time.time() - t0, 2),
            "attempts": attempts,
            "status": "failed",
            "error": last_err,
        }

    if final.exists():
        final.unlink()
    os.replace(part, final)
    size = final.stat().st_size
    log(f"  完成 {t['key']}: {size/1e6:.2f} MB,{time.time()-t0:.1f}s,校验 sha256 ...")
    digest = sha256_of(final)
    log(f"  sha256={digest}")
    return {
        "key": t["key"],
        "url": url,
        "path": str(final),
        "bytes": size,
        "sha256": digest,
        "seconds": round(time.time() - t0, 2),
        "attempts": attempts,
        "status": "ok",
    }


def main() -> int:
    results = {}
    report = CACHE / "download_report.json"
    if report.exists():
        try:
            results = json.loads(report.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            results = {}
    only = sys.argv[1:] or None
    for t in TARGETS:
        if only and t["key"] not in only:
            continue
        existing = results.get(t["key"])
        if existing and existing.get("status") in ("ok", "cached") and existing.get("sha256"):
            log(f"跳过 {t['key']}(report 中已记录)")
            continue
        res = download_one(t)
        results[t["key"]] = res
        report.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        if res["status"] == "failed" and t["required"]:
            log(f"必需文件 {t['key']} 下载失败,继续尝试后续文件")
    log("下载阶段结束")
    for k, v in results.items():
        log(f"  {k:18s} {v['status']:8s} {v['bytes']/1e6:10.2f} MB  {v.get('sha256')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
