"""ANN 训练仪表盘:实时脑活动点云 + KPI + 曲线(浏览器 Canvas,不抢焦点)。

    & $py tools/ann_dashboard.py --state .cache/ann2_state.json

参考网传"果蝇玩 XX"项目的可视化风格:
    - 顶部 4 张 KPI 大数字卡(帧数/loss/拍频/靶距);
    - 左侧训练曲线(loss + 评估散点);
    - 右侧**脑形点云**:16.7 万神经元的解剖分组布局,颜色=变化(蓝↔橙),
      大小/亮度=活动度;
    - 底部状态行。

**布局诚实声明**:MaleCNS v1.0 不含 soma x/y/z 坐标。点云位置来自
"解剖分组簇布局"(每个 superclass 一个手选簇心,簇内确定性抖动),
形状仿脑+VNC 轮廓,**不是真实脑坐标**。前端标题已如实标注。
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# 解剖分组 → (簇心 x,y, 半径, 数量份额)(归一化坐标,仿脑+VNC 轮廓)
_CLUSTERS = {
    "cb_central":   (0.50, 0.26, 0.150),   # 中央脑:cb_intrinsic
    "cb_top":       (0.50, 0.15, 0.090),   # cb_sensory
    "cb_motor":     (0.60, 0.36, 0.045),
    "cb_endocrine": (0.40, 0.37, 0.035),
    "ol_left":      (0.22, 0.36, 0.130),   # 左视叶:ol_intrinsic 前半
    "ol_right":     (0.78, 0.36, 0.130),   # 右视叶
    "eye_left":     (0.11, 0.44, 0.060),   # ol_sensory 左
    "eye_right":    (0.89, 0.44, 0.060),   # ol_sensory 右
    "vis_proj":     (0.50, 0.44, 0.110),   # visual_projection/centrifugal
    "connector":    (0.50, 0.56, 0.150),   # 上下行/感觉上行(脑↔VNC 通道)
    "vnc":          (0.50, 0.82, 0.130),   # 腹神经索
    "ens":          (0.50, 0.95, 0.040),   # 内分泌/ENS
}

# superclass → 簇(按份额切分大组)
_MAP = {
    "ol_sensory": ["eye_left", "eye_right"],
    "ol_intrinsic": ["ol_left", "ol_right"],
    "cb_intrinsic": ["cb_central"],
    "cb_sensory": ["cb_top"],
    "cb_motor": ["cb_motor", "cb_endocrine"],
    "cb_endocrine": ["cb_endocrine"],
    "visual_projection": ["vis_proj"],
    "visual_centrifugal": ["vis_proj"],
    "ascending_neuron": ["connector"],
    "descending_neuron": ["connector"],
    "sensory_ascending": ["connector"],
    "vnc_intrinsic": ["vnc"],
    "vnc_sensory": ["vnc"],
    "vnc_motor": ["vnc"],
    "vnc_efferent": ["vnc"],
    "ENS": ["ens"],
}


def build_layout(n_neurons: int, superclasses: np.ndarray, seed: int = 20261004,
                 sample: int = 4096) -> dict:
    """返回 {x, y, group, sample_idx}(均为抽样子集)。"""
    rng = np.random.default_rng(seed)
    sup = np.asarray([str(s) for s in superclasses])
    # 每个 superclass 抽固定数量(按占比,总共约 sample)
    uniq, counts = np.unique(sup, return_counts=True)
    take = {}
    for u, c in zip(uniq, counts):
        take[u] = max(8, int(round(sample * c / len(sup))))
    idx_all = []
    for u in uniq:
        pool = np.flatnonzero(sup == u)
        k = min(take[u], pool.size)
        idx_all.append(rng.choice(pool, size=k, replace=False))
    sidx = np.sort(np.concatenate(idx_all))
    groups = sup[sidx]

    xs = np.zeros(sidx.size, dtype=np.float32)
    ys = np.zeros(sidx.size, dtype=np.float32)
    for u in np.unique(groups):
        sel = np.flatnonzero(groups == u)
        targets = _MAP.get(u, ["connector"])
        parts = np.array_split(sel, len(targets))
        for part, tname in zip(parts, targets):
            cx, cy, rad = _CLUSTERS[tname]
            r = rad * np.sqrt(rng.random(part.size))
            th = rng.random(part.size) * 2 * np.pi
            xs[part] = cx + r * np.cos(th)
            ys[part] = cy + r * np.sin(th)
    return {"x": xs.tolist(), "y": ys.tolist(), "group": groups.tolist(),
            "sample_idx": sidx.astype(np.int64).tolist()}


def _make_html(layout: dict, title: str) -> str:
    slim = {"x": [round(v, 4) for v in layout["x"]],
            "y": [round(v, 4) for v in layout["y"]],
            "group": layout["group"]}
    payload = json.dumps(slim, ensure_ascii=False)
    return f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>{title}</title>
<style>
 body{{margin:0;background:#0e1216;color:#dbe4ee;font-family:"Microsoft YaHei",system-ui,sans-serif}}
 .hdr{{display:flex;align-items:center;gap:14px;padding:10px 18px;background:#131a22;border-bottom:1px solid #223}}
 .hdr h1{{font-size:17px;margin:0;font-weight:600}}
 .badge{{font-size:12px;color:#7fa;border:1px solid #2a4;border-radius:10px;padding:2px 8px}}
 .note{{font-size:12px;color:#678}}
 .kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;padding:12px 18px}}
 .kpi{{background:#161e27;border:1px solid #223;border-radius:8px;padding:10px 14px}}
 .kpi .v{{font-size:30px;font-weight:700;color:#eaf2ff;font-variant-numeric:tabular-nums}}
 .kpi .l{{font-size:12px;color:#7b8ba0;margin-top:2px}}
 .kpi .u{{font-size:14px;color:#8fa;margin-left:4px}}
 .main{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;padding:0 18px 14px}}
 .card{{background:#161e27;border:1px solid #223;border-radius:8px;padding:10px}}
 .card h2{{font-size:14px;margin:0 0 8px;color:#b9c7d8;font-weight:600}}
 canvas{{width:100%;display:block;border-radius:6px;background:#0b0f14}}
 .foot{{padding:8px 18px;color:#678;font-size:12px}}
 .legend{{display:flex;gap:16px;font-size:12px;color:#8a9bad;margin-top:6px}}
 .sw{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:middle}}
</style></head><body>
<div class="hdr">
  <h1>FlyAim · ANN 训练仪表盘</h1>
  <span class="badge" id="conn">连接中…</span>
  <span class="note" id="connnote">MaleCNS v1.0 · 166,700 神经元</span>
</div>
<div class="kpis">
  <div class="kpi"><div class="v" id="k_frame">—</div><div class="l">累计帧数</div></div>
  <div class="kpi"><div class="v" id="k_loss">—</div><div class="l">克隆 loss</div></div>
  <div class="kpi"><div class="v" id="k_fps">—<span class="u">f/s</span></div><div class="l">训练速度</div></div>
  <div class="kpi"><div class="v" id="k_dist">—<span class="u">px</span></div><div class="l">当前靶距</div></div>
</div>
<div class="main">
  <div class="card">
    <h2>训练曲线</h2>
    <canvas id="curves" height="360"></canvas>
    <div class="legend"><span><span class="sw" style="background:#4090ff"></span>克隆 loss</span>
      <span><span class="sw" style="background:#ffb040"></span>评估靶距(右轴)</span>
      <span><span class="sw" style="background:#20c060"></span>随机基准 278px</span></div>
  </div>
  <div class="card">
    <h2>训练画面 <span class="note">(靶场实帧 · 准星/靶/命中判定)</span></h2>
    <canvas id="scene" height="360" style="background:#0b0f14"></canvas>
    <div class="legend"><span>白点=准星</span><span>红圈=靶</span>
      <span class="note">准星进入靶半径即命中</span></div>
  </div>
  <div class="card">
    <h2>神经活动 <span class="note" id="actnote">(解剖分组簇布局 · 非真实脑坐标)</span></h2>
    <canvas id="brain" height="360"></canvas>
    <div class="legend"><span><span class="sw" style="background:#3a6ea8"></span>下降</span>
      <span><span class="sw" style="background:#556"></span>平稳</span>
      <span><span class="sw" style="background:#e08a3c"></span>上升</span>
      <span class="note">点越大越亮 = 活动越强;显示抽样子集</span></div>
  </div>
</div>
<div class="foot" id="foot">等待训练状态…</div>
<script>
const LAYOUT = {payload};
let S = null;
const cvs = document.getElementById('brain'), ctx = cvs.getContext('2d');
const ccv = document.getElementById('curves'), cctx = ccv.getContext('2d');
const scv = document.getElementById('scene'), sctx = scv.getContext('2d');
const dpr = window.devicePixelRatio || 1;
const sceneImg = new Image();
let sceneReady = false;
sceneImg.onload = () => {{ sceneReady = true; }};
function drawScene(){{
  const w = scv.width/dpr, h = scv.height/dpr;
  sctx.clearRect(0,0,w,h);
  if(!sceneReady) return;
  // 保持靶场 4:3 纵横比,居中 letterbox
  const iw = sceneImg.naturalWidth, ih = sceneImg.naturalHeight;
  const s = Math.min(w/iw, h/ih), dw = iw*s, dh = ih*s;
  sctx.drawImage(sceneImg, (w-dw)/2, (h-dh)/2, dw, dh);
}}
function fit(c, ctx){{
  const r = c.getBoundingClientRect();
  c.width = r.width * dpr; c.height = c.height / (c.height/r.height) * 0; // reset below
}}
function resize(){{
  for (const [c, x] of [[cvs, ctx], [ccv, cctx], [scv, sctx]]) {{
    const r = c.getBoundingClientRect();
    c.width = Math.max(200, r.width * dpr);
    c.height = Math.max(150, (parseFloat(c.getAttribute('height')) || 360) * dpr);
    x.setTransform(dpr,0,0,dpr,0,0);
  }}
}}
window.addEventListener('resize', resize); resize();

function drawBrain(){{
  const w = cvs.width/dpr, h = cvs.height/dpr;
  ctx.clearRect(0,0,w,h);
  if(!S || !S.activity) return;
  // 等比缩放:布局是 0..1 方形坐标,画布为整行宽矩形;若按 x*w/y*h 独立
  // 缩放会把脑形横向拉伸。取短边做边长、水平居中,保持形状不变。
  const side = Math.min(w, h);
  const ox = (w - side)/2, oy = (h - side)/2;
  const act = S.activity;
  const n = act.length;
  // 相对水平:z = (v - mean) / std → 蓝(低于均值)/橙(高于均值)
  let mean=0; for(let i=0;i<n;i++) mean += act[i]; mean /= Math.max(1,n);
  let varr=0; for(let i=0;i<n;i++) varr += (act[i]-mean)**2; varr /= Math.max(1,n);
  const sd = Math.sqrt(varr) || 1e-9;
  const hi = mean + 3*sd;
  for(let i=0;i<n;i++){{
    const z = (act[i]-mean)/sd;
    const u = Math.max(-1, Math.min(1, z/1.2));   // 放宽色阶:更多点进入色区
    let col;
    if (u > 0.06) col = `rgba(236,150,60,${{0.45+0.55*u}})`;
    else if (u < -0.06) col = `rgba(70,140,225,${{0.45+0.55*(-u)}})`;
    else col = 'rgba(140,150,165,0.6)';
    const a = Math.max(0, Math.min(1, act[i]/(hi+1e-9)));
    const rad = 0.8 + 2.4*a;
    ctx.fillStyle = col;
    ctx.beginPath();
    ctx.arc(ox + LAYOUT.x[i]*side, oy + LAYOUT.y[i]*side, rad, 0, 6.283);
    ctx.fill();
  }}
}}
function drawCurves(){{
  const w = ccv.width/dpr, h = ccv.height/dpr;
  cctx.clearRect(0,0,w,h);
  if(!S) return;
  const loss = S.loss_hist || [], ev = S.eval_hist || [];
  const all = loss.concat(ev.map(e=>e[1]));
  const ymax = Math.max(0.05, ...all)*1.1;
  // grid
  cctx.strokeStyle='#1d2733'; cctx.lineWidth=1;
  for(let k=0;k<=4;k++){{const y=h*k/4;cctx.beginPath();cctx.moveTo(0,y);cctx.lineTo(w,y);cctx.stroke();}}
  // random baseline (right axis uses same scale for simplicity: 278px mapped)
  const distMax = Math.max(300, ...ev.map(e=>e[1]))*1.15;
  const ry = h - (278/distMax)*h;
  cctx.strokeStyle='#20c060'; cctx.setLineDash([5,4]);
  cctx.beginPath(); cctx.moveTo(0,ry); cctx.lineTo(w,ry); cctx.stroke();
  cctx.setLineDash([]);
  const xN = Math.max(2, loss.length);
  cctx.strokeStyle='#4090ff'; cctx.lineWidth=2; cctx.beginPath();
  loss.forEach((v,i)=>{{const x=i/(xN-1)*w, y=h-(v/ymax)*h; i?cctx.lineTo(x,y):cctx.moveTo(x,y);}});
  cctx.stroke();
  cctx.fillStyle='#ffb040';
  ev.forEach((e,i)=>{{ if(e[1]<0) return;
    const x=e[0]/(xN-1)*w, y=h-(e[1]/distMax)*h;
    cctx.beginPath(); cctx.arc(x,y,4,0,6.283); cctx.fill(); }});
  cctx.fillStyle='#678'; cctx.font='12px sans-serif';
  cctx.fillText('loss 0 — ' + ymax.toFixed(2), 6, 14);
  cctx.fillStyle='#ffb040';
  cctx.fillText('靶距 0 — ' + distMax.toFixed(0) + 'px', 6, 30);
  cctx.fillStyle='#20c060';
  cctx.fillText('random 278px', w-110, ry-6);
}}
let _lastFrame = -1, _lastChange = Date.now();
async function tick(){{
  try{{
    const r = await fetch('/state', {{cache:'no-store'}});
    const j = await r.json();
    if(j && j.ok){{ S = j; }}
    const f = S.frame||0;
    // 陈旧检测:训练进程退出后状态会静默冻结,看起来像"坏了"。
    // 若累计帧数超过 8s 不变,标为"已结束",避免误判。
    if(f !== _lastFrame){{ _lastFrame = f; _lastChange = Date.now(); }}
    const staleS = Math.round((Date.now()-_lastChange)/1000);
    const badge = document.getElementById('conn');
    if(staleS >= 8 && f > 0){{
      badge.textContent = '训练已结束 · ' + staleS + 's 无新数据';
      badge.style.color = '#e0a03c'; badge.style.borderColor = '#a76';
    }} else {{
      badge.textContent = '已连接';
      badge.style.color = ''; badge.style.borderColor = '';
    }}
    document.getElementById('k_frame').textContent = f.toLocaleString();
    document.getElementById('k_loss').textContent = (S.loss!=null? S.loss.toFixed(4):'—');
    document.getElementById('k_fps').innerHTML = (S.fps!=null? S.fps.toFixed(1):'—')+'<span class="u">f/s</span>';
    document.getElementById('k_dist').innerHTML = (S.target_dist!=null? S.target_dist.toFixed(0):'—')+'<span class="u">px</span>';
    document.getElementById('foot').textContent = S.status || '';
    if(S && S.frame_jpg && S.frame_jpg !== sceneImg._src){{
      sceneImg._src = S.frame_jpg;
      sceneImg.src = 'data:image/jpeg;base64,' + S.frame_jpg;
    }}
    drawScene(); drawBrain(); drawCurves();
  }}catch(e){{ document.getElementById('conn').textContent = '连接中断'; }}
  setTimeout(tick, 150);
}}
tick();
</script></body></html>"""


# ================================================================ 服务器


def _read_shared(path: Path) -> bytes:
    """Windows 共享读:不阻塞训练侧的原子替换(D16 文件锁陷阱)。"""
    import os
    for _ in range(5):
        try:
            fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_BINARY", 0))
            try:
                data = os.read(fd, 1 << 22)
            finally:
                os.close(fd)
            return data if data.strip() else b'{"ok":false}'
        except (PermissionError, FileNotFoundError):
            import time

            time.sleep(0.05)
    return b'{"ok":false}'


def serve(state_path: Path, layout: dict, port: int, title: str, open_browser: bool):
    import http.server

    html = _make_html(layout, title).encode("utf-8")

    class _H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/state"):
                body = _read_shared(state_path)
                self._send(body, "application/json")
            else:
                self._send(html, "text/html; charset=utf-8")

        def _send(self, body: bytes, ctype: str):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), _H)
    url = f"http://127.0.0.1:{port}/"
    print(f"仪表盘: {url}")
    if open_browser:
        try:
            import webbrowser

            webbrowser.open(url)
        except Exception:
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def _encode_jpeg_b64(frame: np.ndarray, quality: int = 75,
                     downsample: int = 2) -> str:
    """靶场帧 -> base64 JPEG(仪表盘"训练画面"面板用)。

    实测(2026-10-04):640×480 JPEG 2.32ms、320×240 0.55ms。面板显示宽度仅
    ~570px,2× 降采样视觉无差别却省 ~75% 编码时间与一半 payload。
    """
    import base64
    import io

    from PIL import Image

    arr = np.ascontiguousarray(frame)
    if downsample > 1:
        arr = np.ascontiguousarray(arr[::downsample, ::downsample])
    im = Image.fromarray(arr)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class StatePublisher:
    """训练进程侧:把状态写入 JSON(仪表盘读)。

    逐帧发布时,**编码/JSON/写盘全部放到后台 worker 线程**,训练主循环只做
    一次入队(微秒级)。这些工作虽然单项只有 ~1ms,但都在训练线程里持 GIL
    串行执行,叠加后实测把拍频从 27.6 压到 19.2 f/s —— 故必须移出主循环。
    队列满时丢最旧:可视化丢帧无害,训练绝不阻塞。
    """

    def __init__(self, path: str | Path, sample_idx: np.ndarray | None = None,
                 total_frames: int | None = None, frame_every: int = 1,
                 queue_size: int = 8):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sample_idx = None if sample_idx is None else np.asarray(sample_idx, np.int64)
        self.total = total_frames
        self.frame_every = int(frame_every)   # 画面发布粒度(1=每帧);<=0 关闭
        self._frame_n = 0
        self.loss_hist: list[float] = []
        self.eval_hist: list[list] = []   # [frame_or_progress, mean_dist]
        self.frame = 0
        self._t0 = None
        self._last_activity = None
        self._last_frame = None
        self._last_rollout_n = None
        self._q: "queue.Queue" = queue.Queue(maxsize=int(queue_size))
        threading.Thread(target=self._run, daemon=True, name="dash-pub").start()

    def __call__(self, stage: str, **kv):
        import time

        if self._t0 is None:
            self._t0 = time.perf_counter()
        if stage == "rollout":
            n = int(kv.get("frames", 0))
            prev = self._last_rollout_n or 0
            self.frame += (n - prev) if n >= prev else n   # 回绕(新轮)时直接加
            self._last_rollout_n = n
            self._enqueue({
                "kind": "rollout", "frame": self.frame, "n": n,
                "loss": self.loss_hist[-1] if self.loss_hist else None,
                "fps": n / max(time.perf_counter() - self._t0, 1e-9),
                "target_dist": kv.get("target_dist"), "hit": kv.get("hit"),
                "policy": kv.get("policy"), "activity": kv.get("activity"),
                "arena_frame": kv.get("frame"),
                "loss_hist": list(self.loss_hist), "eval_hist": list(self.eval_hist),
            })
        elif stage == "round_done":
            self.loss_hist.append(float(kv.get("loss", 0.0)))
            self._last_rollout_n = None
            self._t0 = None
            self._enqueue({"kind": "round_done", "frame": self.frame,
                           "round": kv.get("round"), "loss": self.loss_hist[-1],
                           "loss_hist": list(self.loss_hist),
                           "eval_hist": list(self.eval_hist)})
        elif stage == "eval":
            vals = kv.get("values", [])
            for i, v in enumerate(vals):
                self.eval_hist.append([i + 1, float(v)])
            self._enqueue({"kind": "eval", "frame": self.frame,
                           "loss": self.loss_hist[-1] if self.loss_hist else None,
                           "arm": kv.get("arm"), "has_vals": bool(vals),
                           "vals_mean": float(np.mean(vals)) if vals else None,
                           "loss_hist": list(self.loss_hist),
                           "eval_hist": list(self.eval_hist)})

    def _enqueue(self, spec: dict):
        """非阻塞入队;满则丢最旧一帧(只影响可视化的流畅度)。"""
        try:
            self._q.put_nowait(spec)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(spec)
            except queue.Full:
                pass

    def _run(self):
        while True:
            spec = self._q.get()
            if spec is None:
                return
            try:
                self._build_and_write(spec)
            except Exception:
                pass   # 可视化失败绝不拖累训练

    def _build_and_write(self, spec: dict):
        kind = spec["kind"]
        if kind == "rollout":
            payload = {
                "ok": True, "stage": "rollout", "frame": spec["frame"],
                "loss": spec["loss"], "fps": spec["fps"],
                "target_dist": spec["target_dist"], "hit": spec["hit"],
                "status": f"采集中({'策略驱动' if spec['policy'] else '教师驱动'}) "
                          f"— 本轮 {spec['n']} 帧",
                "loss_hist": spec["loss_hist"], "eval_hist": spec["eval_hist"],
            }
            act = spec["activity"]
            if act is not None and self.sample_idx is not None:
                arr = np.asarray(act)
                # 训练侧可能已在 GPU 上按 sample_idx 切片(逐帧刷新的省流路径);
                # 长度已等于采样数时不再二次索引。
                if arr.size != self.sample_idx.size:
                    arr = arr[self.sample_idx]
                payload["activity"] = np.round(arr.astype(float), 3).tolist()
                self._last_activity = payload["activity"]
            fr = spec["arena_frame"]
            if fr is not None and self.frame_every > 0:
                self._frame_n += 1
                if self._frame_n % self.frame_every == 0:
                    payload["frame_jpg"] = _encode_jpeg_b64(fr)
                    self._last_frame = payload["frame_jpg"]
        elif kind == "round_done":
            payload = {"ok": True, "stage": "round_done", "frame": spec["frame"],
                       "loss": spec["loss"],
                       "status": f"轮 {spec['round']} 训练完成",
                       "loss_hist": spec["loss_hist"], "eval_hist": spec["eval_hist"]}
        else:  # eval
            payload = {"ok": True, "stage": "eval", "frame": spec["frame"],
                       "loss": spec["loss"],
                       "status": (f"评估 {spec['arm']}: 均值 {spec['vals_mean']:.0f}px"
                                  if spec["has_vals"] else "评估中"),
                       "loss_hist": spec["loss_hist"], "eval_hist": spec["eval_hist"]}
        self._write(payload)

    def _write(self, payload: dict):
        """原地写状态(D16 教训:Windows 上 os.replace 到被读端打开的文件
        会 WinError 5;且读端加共享读仍可能被其他句柄挡住)。可视化失败
        绝不拖累训练——最终兜底完全静默。"""
        if "activity" not in payload and self._last_activity is not None:
            payload["activity"] = self._last_activity
        if "frame_jpg" not in payload and self._last_frame is not None:
            payload["frame_jpg"] = self._last_frame
        import time

        body = json.dumps(payload, ensure_ascii=False)
        # 逐帧(30Hz)发布时,重试退避必须很短,否则会反过来拖住训练循环。
        for attempt in range(3):
            try:
                with open(self.path, "w", encoding="utf-8") as fh:
                    fh.write(body)
                return
            except (PermissionError, OSError):
                time.sleep(0.004 * (attempt + 1))
        # 仍失败则放弃这一次(只影响可视化的一个刷新点,训练继续)

    def set_frame(self, frame: int):
        self.frame = int(frame)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=".cache/ann2_state.json")
    ap.add_argument("--port", type=int, default=8764)
    ap.add_argument("--sample", type=int, default=4096)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    import pandas as pd

    idx = pd.read_parquet(ROOT / "flyaim/data/build/neuron_index.parquet")
    cache = ROOT / ".cache/ann_layout.json"
    if cache.exists():
        layout = json.loads(cache.read_text(encoding="utf-8"))
    else:
        print("构建神经元布局(一次,缓存到 .cache/ann_layout.json)…")
        layout = build_layout(len(idx), idx["superclass"].astype(str).to_numpy(),
                              sample=args.sample)
        cache.write_text(json.dumps(layout), encoding="utf-8")
    print(f"布局 OK:{len(layout['x'])} 个抽样神经元点")
    serve(Path(ROOT / args.state), layout, args.port, "FlyAim · ANN 训练", not args.no_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
