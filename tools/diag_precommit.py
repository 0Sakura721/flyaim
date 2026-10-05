"""提交前体检:精确列出 `git add -A` 会收进什么、多大、有没有不该进的。

用法::
    & $py tools/_precommit_audit.py
"""
import collections
import os
import subprocess

ROOT = r"D:\dsh\autoaim"

# 不该进仓库的东西:大文件、可再生产物、局数据
SUSPECT_EXT = {".mp4", ".raw", ".png", ".csv", ".npz", ".parquet", ".jsonl",
               ".pkl", ".log", ".exe", ".zip"}
BIG = 200 * 1024          # 超过这个大小就点名


def git(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return r.stdout


def main() -> int:
    out = git("add", "-A", "--dry-run")
    entries: list[tuple[str, str]] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        act = line.split()[0]
        if "'" not in line:
            continue
        path = line.split("'", 1)[1].rsplit("'", 1)[0]
        entries.append((act, path))

    print("动作分布:", dict(collections.Counter(a for a, _ in entries)))
    print()

    files = [(a, p) for a, p in entries if not p.endswith("/")]
    total = 0
    per_dir: collections.Counter = collections.Counter()
    big: list[tuple[int, str]] = []
    suspect: list[tuple[int, str]] = []
    for _act, p in files:
        fp = os.path.join(ROOT, p.replace("/", os.sep))
        if not os.path.isfile(fp):
            continue
        sz = os.path.getsize(fp)
        total += sz
        per_dir[p.split("/")[0] if "/" in p else "(root)"] += sz
        if sz > BIG:
            big.append((sz, p))
        if os.path.splitext(p)[1].lower() in SUSPECT_EXT:
            suspect.append((sz, p))

    print(f"文件数 {len(files)}   总大小 {total / 1024 / 1024:.2f} MB")
    print("按顶层目录:")
    for d, sz in per_dir.most_common():
        print(f"  {sz / 1024 / 1024:8.3f} MB  {d}")
    print()
    print(f"⚠️ 可疑扩展名({len(suspect)} 个,合计 "
          f"{sum(s for s, _ in suspect) / 1024 / 1024:.2f} MB):")
    for sz, p in sorted(suspect, reverse=True)[:30]:
        print(f"  {sz / 1024 / 1024:8.3f} MB  {p}")
    print()
    print(f"⚠️ 单文件 >200KB({len(big)} 个):")
    for sz, p in sorted(big, reverse=True):
        print(f"  {sz / 1024 / 1024:8.3f} MB  {p}")
    print()
    print("=== 全部待提交文件 ===")
    for _a, p in sorted(files, key=lambda t: t[1]):
        print("   ", p)
    print()
    print("=== 删除/重命名 ===")
    for a, p in entries:
        if a != "add":
            print(f"    {a:8s} {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
