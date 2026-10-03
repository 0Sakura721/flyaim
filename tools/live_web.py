"""浏览器 Canvas 实时渲染 —— 替代 matplotlib/TkAgg。

为什么换掉 TkAgg
----------------
1. **窗口"未响应"**:脑仿真一帧 ~240 ms,主线程被 numpy 占住时,
   Tk 的消息循环得不到调度 → Windows 判定窗口无响应(标题栏挂"未响应")。
2. **抢焦点**:Tk 的 `lift()` / `topmost` 会把窗口顶到最前。
3. **卡**:matplotlib 的 Agg 渲染是纯 CPU 的,和仿真抢同一个核(实测单核满载)。
4. **不能自由调整大小**:figsize 固定,拖拽不重排。

浏览器方案
----------
- Python 侧**只写 JSON**(原子替换),开销 < 2 ms,不阻塞仿真;
- 一个 daemon 线程跑 http.server,服务静态文件;
- 浏览器用 `requestAnimationFrame` + Canvas 画热图(硬件加速,60 fps);
- 窗口是浏览器窗口:可拖拽缩放、不抢焦点、关闭不影响训练;
- 零新依赖(Windows 必有 Edge/Chrome)。

接口与 `live_render.LiveRenderer` 一致(`update` / `close` / `is_live`),
可直接替换。

用法::

    from tools.live_web import make_viewer

    live = make_viewer(roles, idx, out_dir="flyaim/runs/live",
                       title="FlyAim 训练")
    for i in range(n_frames):
        ...
        live.update(frame=frame, spikes=brain.spikes,
                    rates=brain.rates, meta={"target_dist": d, "R2": r2})
    live.close()
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

_LAYER_PRIORITY = ("ol_sensory", "ol_intrinsic", "visual_projection", "visual_centrifugal",
                   "descending_neuron", "vnc_motor")


def _say(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _u8(arr: np.ndarray) -> np.ndarray:
    """把浮点数组归一化到 uint8(0-255),供浏览器直接当灰度索引用。"""
    a = np.asarray(arr, np.float32)
    lo = float(a.min()) if a.size else 0.0
    hi = float(a.max()) if a.size else 1.0
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((a - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


class _NullViewer:
    is_live = False

    def update(self, **kw):
        return None

    def close(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class LiveViewer:
    """写数据 + 起本地 HTTP + 开浏览器窗口。"""

    is_live = True

    def __init__(self, roles, neuron_index=None, *, out_dir,
                 sample_shape=(96, 96), dn_rows=128, history=256,
                 preview_size=(320, 240), title: str = "FlyAim",
                 every: int = 1, preview_every: int = 4,
                 open_browser: bool = True, port: int = 0):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.sample_shape = tuple(int(x) for x in sample_shape)
        self.dn_rows = int(dn_rows)
        self.history = int(history)
        self.preview_size = tuple(int(x) for x in preview_size)
        self.every = max(1, int(every))
        # JPEG 编码是渲染侧最贵的一步(约 5~10 ms),单独降频
        self.preview_every = max(1, int(preview_every))
        self.title = str(title)
        self.n_frames = 0
        self.t0 = time.perf_counter()
        self._rast_u8 = None
        self._dn_u8 = None
        self._preview_b64 = None
        self._stats = {}
        self.url = ""

        # --- 下采样方案 ---
        df = (neuron_index.df if hasattr(neuron_index, "df") else neuron_index)
        n = int(df.shape[0]) if df is not None else 0
        k = int(np.prod(self.sample_shape))
        rng = np.random.default_rng(20261003)
        self.sample_idx = (np.arange(n, dtype=np.int64) if n <= k else
                           np.sort(rng.choice(n, size=k, replace=False).astype(np.int64)))
        self.dn_idx = np.asarray(roles.descending, np.int64)

        sup = None
        if df is not None and "superclass" in getattr(df, "columns", []):
            sup = df["superclass"].astype(str).to_numpy()[self.sample_idx]
        self.order, self.bounds, self.names = self._build_order(self.sample_idx, sup)

        if self.dn_idx.size > self.dn_rows:
            sel = (np.arange(self.dn_rows) * (self.dn_idx.size // self.dn_rows)).astype(np.int64)
            self.dn_idx = self.dn_idx[np.sort(sel)]

        self._dn_buf = None
        self._xs: list[int] = []
        self._dist: list[float] = []
        self._sr: list[float] = []
        self._rx: list[int] = []
        self._ry: list[float] = []

        # --- 写 HTML + 起服务 + 开浏览器 ---
        # HTML 只落一份到磁盘作为备份(浏览器实际从内存服务拿,不读这个文件)
        (self.out / "index.html").write_text(_HTML, encoding="utf-8")
        self._start_server(port)
        self._publish()  # 先发布一份,浏览器进来就有东西可画
        if open_browser:
            self._open_browser()

    # ---------------------------------------------------------------- 排序

    @staticmethod
    def _build_order(sample_idx, labels):
        if labels is None:
            return np.arange(len(sample_idx)), [], []
        lab = np.asarray([str(x) for x in labels])
        present = [c for c in _LAYER_PRIORITY if np.any(lab == c)]
        present += sorted({*set(lab)} - set(present))
        parts, bounds, cur = [], [], 0
        for c in present:
            sel = np.flatnonzero(lab == c)
            if sel.size == 0:
                continue
            parts.append(sel)
            cur += sel.size
            bounds.append(cur)
        order = np.concatenate(parts) if parts else np.arange(lab.size)
        return order, bounds, present

    # ---------------------------------------------------------------- HTTP

    def _start_server(self, port: int) -> None:
        """起一个**内存服务**的 HTTP 服务。

        为什么不在磁盘上写 data.json:
            浏览器每 ~16ms 轮询一次,HTTP 服务线程会一直持有 `data.json` 的读句柄,
            此时 `os.replace(tmp, data.json)` 在 Windows 上必然抛
            `PermissionError [WinError 5]`(目标文件被打开时无法替换)。
            改成内存服务后:无临时文件、无替换、无锁,而且更快(省掉磁盘 I/O)。
        """
        import http.server
        import socketserver

        html_bytes = _HTML.encode("utf-8")
        shared: dict[str, bytes] = {"data": b"{}"}
        self._shared = shared

        class _H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):
                path = self.path.split("?")[0]
                if path in ("/", "/index.html"):
                    body, ctype = html_bytes, "text/html; charset=utf-8"
                elif path == "/data.json":
                    body, ctype = shared["data"], "application/json; charset=utf-8"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *a):
                pass  # 不刷屏

        self.httpd = socketserver.ThreadingTCPServer(("127.0.0.1", int(port)), _H)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def _open_browser(self) -> None:
        import webbrowser

        self.url = f"http://127.0.0.1:{self.port}/index.html"
        try:
            webbrowser.open(self.url)
        except Exception:
            pass
        _say(f"[live] 浏览器窗口: {self.url}")
        _say(f"[live]   关掉浏览器窗口不影响训练;想再打开就访问上面的地址")

    # ---------------------------------------------------------------- 数据

    def _encode_preview(self, frame) -> str | None:
        if frame is None:
            return None
        try:
            from PIL import Image
        except ImportError:
            return None
        im = Image.fromarray(np.asarray(frame).astype(np.uint8))
        if im.size != self.preview_size:
            im = im.resize(self.preview_size, Image.BILINEAR)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=62, optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _publish(self) -> None:
        """把当前状态序列化到内存,供 HTTP 服务直接返回(不落盘)。"""
        payload = {
            "title": self.title,
            "frame": int(self.n_frames),
            "wall_s": round(time.perf_counter() - self.t0, 2),
            "sample_shape": list(self.sample_shape),
            "layer_bounds": [int(x) for x in self.bounds],
            "layer_names": [str(x) for x in self.names],
            "dn_rows": int(self.dn_idx.size),
            "xs": self._xs[-self.history:],
            "dist": self._dist[-self.history:],
            "sr": self._sr[-self.history:],
            "rx": self._rx[-self.history:],
            "ry": self._ry[-self.history:],
        }
        if getattr(self, "_rast_u8", None) is not None:
            payload["rast"] = base64.b64encode(self._rast_u8.tobytes()).decode("ascii")
            payload["rast_max"] = float(getattr(self, "_rast_max", 1.0))
        if getattr(self, "_dn_u8", None) is not None:
            payload["dn"] = base64.b64encode(self._dn_u8.tobytes()).decode("ascii")
            payload["dn_w"] = int(self._dn_u8.shape[1])
            payload["dn_max"] = float(getattr(self, "_dn_max", 1.0))
        if getattr(self, "_preview_b64", None):
            payload["preview"] = self._preview_b64
        payload.update(getattr(self, "_stats", {}))

        # 原子地换掉内存里的字节串:HTTP 线程要么拿到旧的、要么拿到新的,
        # 不会读到半截 JSON(磁盘方案的 WinError 5 就出在这一步)。
        self._shared["data"] = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    # ---------------------------------------------------------------- 每帧

    def update(self, *, frame=None, spikes=None, rates=None, meta=None) -> None:
        try:
            self._update(frame, spikes, rates, meta or {})
        except Exception:
            import traceback

            traceback.print_exc(file=sys.stderr)

    def _update(self, frame, spikes, rates, meta) -> None:
        self.n_frames += 1
        i = self.n_frames
        rt = None if rates is None else np.asarray(rates).ravel()
        sp = None if spikes is None else np.asarray(spikes).ravel()

        # 曲线数据(每次都收)
        self._xs.append(i)
        self._dist.append(float(meta.get("target_dist", float("nan"))))
        self._sr.append(float(sp.mean()) if sp is not None else float("nan"))
        if meta.get("r2") is not None and np.isfinite(meta["r2"]):
            self._rx.append(i)
            self._ry.append(float(meta["r2"]))

        stats = {"spike_rate": float(sp.mean()) if sp is not None else 0.0,
                 "active": int((rt > 0).sum()) if rt is not None else 0}
        for k in ("target_dist", "hit_rate", "hits", "n_samples", "r2"):
            if k in meta and isinstance(meta[k], (int, float)) and np.isfinite(meta[k]):
                stats[k] = (float(meta[k]) if isinstance(meta[k], float) else int(meta[k]))
        self._stats = stats

        if i % self.every != 0:
            return

        # 热图
        if rt is not None and rt.size > self.sample_idx.size:
            full = rt[self.order].reshape(self.sample_shape)
            self._rast_max = float(max(full.max(), 1e-6))
            self._rast_u8 = _u8(full)
        # DN 时序
        if rt is not None and self.dn_idx.size and rt.size > int(self.dn_idx.max()):
            row = rt[self.dn_idx][None, :]
            self._dn_buf = row if self._dn_buf is None else \
                np.concatenate([self._dn_buf, row], axis=0)[-self.history:]
            self._dn_max = float(max(self._dn_buf.max(), 1e-6))
            self._dn_u8 = _u8(self._dn_buf)
        # 预览帧(JPEG 编码最贵,单独降频;沿用上一帧图像填充)
        if frame is not None and (i % self.preview_every == 0 or self._preview_b64 is None):
            self._preview_b64 = self._encode_preview(frame)

        self._publish()

    # ---------------------------------------------------------------- 收尾

    def close(self) -> None:
        try:
            self.httpd.shutdown()
            self.httpd.server_close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# ================================================================ HTML

_HTML = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>FlyAim live</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; background:#141418; color:#ddd;
         font:13px/1.45 "Segoe UI","Microsoft YaHei",system-ui,sans-serif; }
  header { padding:8px 14px; background:#1d1d22; border-bottom:1px solid #333;
           display:flex; gap:16px; align-items:baseline; flex-wrap:wrap; }
  h1 { font-size:15px; margin:0; font-weight:600; color:#fff; }
  #stats span { margin-right:14px; opacity:.9; }
  #stats b { color:#7fd3ff; font-weight:600; }
  #grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; padding:10px; }
  .card { background:#1b1b21; border:1px solid #2c2c34; border-radius:6px; padding:8px; }
  .card h2 { font-size:12px; margin:0 0 6px; font-weight:600; color:#9aa; }
  canvas, img { width:100%; display:block; border-radius:4px; background:#000; }
  .mono { font-family:Consolas,monospace; }
  #msg { padding:10px 14px; color:#f6b26b; display:none; }
</style></head>
<body>
<header>
  <h1 id="title">FlyAim live</h1>
  <div id="stats" class="mono"></div>
</header>
<div id="msg">正在等待数据…</div>
<div id="grid">
  <div class="card"><h2>靶场画面 / Arena</h2><img id="prev" alt=""></div>
  <div class="card"><h2>全网活动热图(按解剖层次分块)</h2><canvas id="rast"></canvas></div>
  <div class="card"><h2>下行神经元放电率(Hz) — 滚动时序</h2><canvas id="dn"></canvas></div>
  <div class="card"><h2>指标曲线</h2><canvas id="met"></canvas></div>
</div>
<script>
const $ = id => document.getElementById(id);

// ---- 伪彩色(inferno / magma 近似) ----
const INFERNO = [[0,0,4],[12,8,44],[36,12,82],[66,10,104],[93,21,110],[120,32,112],
  [147,44,109],[173,57,101],[198,73,88],[221,92,70],[239,116,49],[250,145,26],
  [250,179,12],[240,213,45],[252,255,164]];
const MAGMA  = [[0,0,4],[18,10,54],[51,13,92],[84,20,110],[116,31,120],[148,44,122],
  [181,59,115],[210,78,98],[234,102,72],[250,132,43],[253,170,25],[250,210,55],
  [252,253,191]];
function lerp(t, stops) {
  const x = t * (stops.length - 1), i = Math.min(Math.floor(x), stops.length - 2), f = x - i;
  const a = stops[i], b = stops[i+1];
  return [a[0]+(b[0]-a[0])*f, a[1]+(b[1]-a[1])*f, a[2]+(b[2]-a[2])*f];
}
const cache = {};
function palette(name) {
  if (!cache[name]) {
    const stops = name === 'magma' ? MAGMA : INFERNO;
    const t = new Uint8Array(256*3);
    for (let i = 0; i < 256; i++) { const c = lerp(i/255, stops);
      t[i*3]=c[0]; t[i*3+1]=c[1]; t[i*3+2]=c[2]; }
    cache[name] = t;
  }
  return cache[name];
}

function drawHeat(cv, b64, w, h, name, labels, bounds) {
  const bin = atob(b64);
  const off = document.createElement('canvas'); off.width=w; off.height=h;
  const octx = off.getContext('2d');
  const img = octx.createImageData(w,h), t = palette(name);
  for (let i=0;i<w*h;i++) {
    const v = bin.charCodeAt(i);
    img.data[i*4]=t[v*3]; img.data[i*4+1]=t[v*3+1]; img.data[i*4+2]=t[v*3+2]; img.data[i*4+3]=255;
  }
  octx.putImageData(img,0,0);
  const ctx = cv.getContext('2d');
  cv.width = 900; cv.height = Math.round(900*h/w) || 300;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off,0,0,cv.width,cv.height);
  if (bounds && bounds.length) {
    ctx.strokeStyle='rgba(255,255,255,.5)'; ctx.lineWidth=1.5;
    ctx.font='12px sans-serif'; ctx.fillStyle='rgba(255,255,255,.85)';
    let prev=0;
    for (let i=0;i<bounds.length;i++) {
      const y = bounds[i]/h*cv.height;
      ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(cv.width,y); ctx.stroke();
      if (labels && labels[i]) {
        ctx.fillText(labels[i], 6, (prev+y)/2+4);
      }
      prev=y;
    }
  }
}

function drawLine(cv, xs, series, colors, labels) {
  const W=900, H=Math.round(900*0.52);
  cv.width=W; cv.height=H;
  const ctx=cv.getContext('2d');
  ctx.clearRect(0,0,W,H);
  ctx.strokeStyle='#2a2a32'; ctx.lineWidth=1;
  for (let g=0; g<=4; g++) { const y=g/4*H;
    ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke(); }
  if (!xs || !xs.length) return;
  const n=xs.length, x0=xs[0], x1=xs[n-1]||1;
  series.forEach((s,si)=>{
    if (!s || !s.length) return;
    let lo=Infinity, hi=-Infinity;
    for (const v of s) if (isFinite(v)) { if(v<lo)lo=v; if(v>hi)hi=v; }
    if (!isFinite(lo)) return;
    if (hi-lo < 1e-9) { hi=lo+1; }
    ctx.strokeStyle=colors[si]; ctx.lineWidth=1.8; ctx.beginPath();
    for (let i=0;i<n;i++) {
      const v=s[i]; if (!isFinite(v)) continue;
      const x=(xs[i]-x0)/Math.max(1,(x1-x0))*W;
      const y=H-((v-lo)/(hi-lo))*(H-8)-4;
      i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    }
    ctx.stroke();
  });
  ctx.font='12px sans-serif';
  labels.forEach((L,i)=>{ ctx.fillStyle=colors[i]; ctx.fillText(L, 10, 20+i*16); });
}

async function tick() {
  try {
    const r = await fetch('data.json?_=' + Date.now(), {cache:'no-store'});
    if (!r.ok) return;
    const d = await r.json();
    $('msg').style.display = d.frame ? 'none' : 'block';
    $('title').textContent = d.title || 'FlyAim live';

    const s = d.stats || {};
    $('stats').innerHTML =
      `<span>帧 <b>${d.frame}</b></span>` +
      (s.target_dist!=null ? `<span>靶距 <b>${s.target_dist.toFixed(1)}px</b></span>`:'') +
      (s.spike_rate!=null ? `<span>spike <b>${s.spike_rate.toFixed(4)}</b></span>`:'') +
      (s.active!=null ? `<span>活跃 <b>${s.active}</b></span>`:'') +
      (s.r2!=null ? `<span>R² <b>${s.r2.toFixed(4)}</b></span>`:'') +
      (s.n_samples!=null ? `<span>n <b>${s.n_samples}</b></span>`:'') +
      (s.hit_rate!=null ? `<span>命中 <b>${(s.hit_rate*100).toFixed(2)}%</b></span>`:'') +
      `<span style="opacity:.6">${d.wall_s}s</span>`;

    if (d.preview) $('prev').src = 'data:image/jpeg;base64,' + d.preview;

    const [h,w] = d.sample_shape;
    if (d.rast) drawHeat($('rast'), d.rast, w, h, 'inferno', d.layer_names, d.layer_bounds);
    if (d.dn)   drawHeat($('dn'), d.dn, d.dn_w, d.dn_rows, 'magma', null, null);
    drawLine($('met'), d.xs,
             [d.dist, (d.sr||[]).map(v=>v*1000), d.ry],
             ['#4a9eff','#8a8a92','#ff5252'],
             ['平均靶距 (px)','spike_rate x1000','读出层 R²']);
  } catch (e) { /* 服务还没起/文件还没写 */ }
  // 数据每帧才更新一次(训练一帧 ~250ms),60fps 轮询纯属浪费
  setTimeout(tick, 150);
}
tick();
</script>
</body></html>
"""


def make_viewer(roles, neuron_index=None, **kw):
    """构造 LiveViewer(浏览器版);不可用时返回 no-op。"""
    try:
        v = LiveViewer(roles, neuron_index, **kw)
    except Exception:
        import traceback

        _say("[live] 浏览器渲染不可用,降级为无输出模式")
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        return _NullViewer()
    _say(f"[live] OK  内存服务已就绪  -> {v.url}")
    _say(f"[live]    数据经 HTTP 内存直出,不写磁盘(避免 Windows 文件锁 WinError 5)")
    return v
