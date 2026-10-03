"""高速并行下载器(纯标准库,零依赖)。

为什么需要它
------------
PyPI 及其国内镜像对**单连接**限速(实测 0.02~0.07 MB/s),
但同一文件的多个 Range 请求**不共享**该限速预算。
把文件切成小段、多个连接并发拉,再拼回去 —— 实测聚合峰值可达 47 MB/s。

**拖尾(stragler)问题与对策**
第一版用 2MB 分块 + 32 连接,实测平均只有 0.68 MB/s:
前 60% 以 48 MB/s 冲完,然后被少数几个卡住的块拖到 100 分钟 ETA。
对策(本版):
  1. **更小的分块**(默认 512KB)→ 负载更均衡,单个慢块的影响面缩小 4 倍;
  2. **停滞超时**:socket timeout + 每块墙钟上限,超时即放弃该块并重试
     —— 重连往往会落到不同的 CDN 边缘节点,慢节点就被绕开了;
  3. **指数退避重试**,失败块重新入队(work-stealing 队列)。

用法::

    <捆绑 python> tools/fetch_fast.py <url> <out> [--conn 64] [--chunk-kb 512]
    <捆绑 python> tools/fetch_fast.py --batch urls.txt --dest DIR [--conn 64]
"""

from __future__ import annotations

import argparse
import hashlib
import queue
import sys
import threading
import time
import urllib.request
from pathlib import Path

UA = {"User-Agent": "pip/24"}


def head_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD", headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers.get("Content-Length", 0))


def fetch_chunk(url: str, start: int, end: int, out: str,
                sock_timeout: float = 20.0, deadline_s: float = 60.0,
                retries: int = 4) -> int:
    """下载 [start, end] 一段并写到 out 的对应偏移。超时/短读会重试。"""
    n_bytes = end - start + 1
    for attempt in range(retries):
        off = start
        try:
            req = urllib.request.Request(
                url, headers={**UA, "Range": f"bytes={off}-{end}"})
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=sock_timeout) as r:
                if r.status not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status}")
                with open(out, "r+b") as f:
                    f.seek(off)
                    while off <= end:
                        if time.time() - t0 > deadline_s:
                            raise TimeoutError(
                                f"块超时 {time.time()-t0:.0f}s @ {off-start}/{n_bytes}B")
                        b = r.read(min(65536, end + 1 - off))
                        if not b:
                            break
                        f.write(b)
                        off += len(b)
            if off == end + 1:
                return n_bytes
            raise IOError(f"短读 {off-start}/{n_bytes}B")
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(min(0.4 * (2 ** attempt), 6.0))
    return 0


def download(url: str, out: Path, conn: int = 64, chunk_kb: int = 512,
             sock_timeout: float = 20.0, deadline_s: float = 60.0) -> dict:
    t0 = time.perf_counter()
    total = head_size(url)
    if total <= 0:
        raise RuntimeError(f"拿不到 Content-Length: {url}")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(out.suffix + ".part")
    if part.exists() and part.stat().st_size != total:
        part.unlink()
    if not part.exists():
        with open(part, "wb") as f:
            f.truncate(total)

    chunk = max(1, int(chunk_kb)) * 1024
    ranges = [(a, min(a + chunk - 1, total - 1)) for a in range(0, total, chunk)]
    print(f"[fetch] {out.name}: {total/1e6:.1f} MB, {len(ranges)} 块 × "
          f"{chunk_kb}KB, {conn} 连接", flush=True)

    q: queue.Queue = queue.Queue()
    for rg in ranges:
        q.put(rg)
    done = [0]
    failed: list = []
    last = [time.perf_counter(), 0.0]
    lock = threading.Lock()

    def worker():
        while True:
            try:
                a, b = q.get_nowait()
            except queue.Empty:
                return
            try:
                n = fetch_chunk(url, a, b, str(part), sock_timeout, deadline_s)
            except Exception as e:
                with lock:
                    failed.append((a, b, f"{type(e).__name__}: {e}"))
                q.task_done()
                continue
            with lock:
                done[0] += n
                now = time.perf_counter()
                if now - last[0] > 1.0:
                    rate = (done[0] - last[1]) / (now - last[0]) / 1e6
                    pct = 100.0 * done[0] / total
                    eta = (total - done[0]) / max(rate * 1e6, 1) / 60.0
                    print(f"  {pct:5.1f}%  {done[0]/1e6:7.1f}/{total/1e6:.1f} MB  "
                          f"{rate:6.2f} MB/s  ETA {eta:4.1f} 分", flush=True)
                    last[0], last[1] = now, done[0]
            q.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(int(conn))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if failed:
        raise RuntimeError(
            f"{len(failed)} 个块最终失败,例: {failed[0][2]}")
    if part.stat().st_size != total:
        raise RuntimeError(f"大小不符 {part.stat().st_size} != {total}")
    part.replace(out)
    dt = time.perf_counter() - t0
    rate = total / dt / 1e6
    print(f"[fetch] OK {out}  {total/1e6:.1f} MB  {dt:.1f}s  {rate:.2f} MB/s", flush=True)
    return {"path": str(out), "bytes": total, "seconds": dt, "rate_mbps": rate}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while (b := f.read(1 << 20)):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--batch", help="每行 'name|url' 的清单文件")
    ap.add_argument("--dest", default=".cache/wheels")
    ap.add_argument("--conn", type=int, default=64)
    ap.add_argument("--chunk-kb", type=int, default=512)
    ap.add_argument("--sock-timeout", type=float, default=20.0)
    ap.add_argument("--deadline", type=float, default=60.0)
    ap.add_argument("--sha256", default=None)
    args = ap.parse_args()

    if args.batch:
        lines = [l.strip() for l in Path(args.batch).read_text(encoding="utf-8").splitlines()
                 if l.strip() and "|" in l]
        dest = Path(args.dest)
        print(f"[fetch] 批量 {len(lines)} 个文件 -> {dest}", flush=True)
        ok = 0
        for line in lines:
            name, url = line.split("|", 1)
            fname = url.split("/")[-1].split("#")[0]
            target = dest / fname
            if target.exists() and target.stat().st_size > 0:
                print(f"[fetch] 跳过已存在: {fname}", flush=True)
                ok += 1
                continue
            try:
                download(url, target, args.conn, args.chunk_kb,
                         args.sock_timeout, args.deadline)
                ok += 1
            except Exception as e:
                print(f"[fetch] {name} 失败: {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
        print(f"[fetch] 批量完成 {ok}/{len(lines)}", flush=True)
        return 0 if ok == len(lines) else 1

    if not args.url or not args.out:
        ap.error("需要 url 和 out,或 --batch")
    try:
        download(args.url, Path(args.out), args.conn, args.chunk_kb,
                 args.sock_timeout, args.deadline)
    except Exception as e:
        print(f"[fetch] 失败: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    if args.sha256:
        got = sha256(Path(args.out))
        ok = got.lower() == args.sha256.lower()
        print(f"[fetch] sha256 {'OK' if ok else 'MISMATCH'} {got}")
        return 0 if ok else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
