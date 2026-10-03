"""报告生成:把实验结果渲染成 Markdown。

关键约束(CONTRACT.md 第 3、5 节):
    - 零模型数据**必须**出现在报告里,无论结果是否有利
    - 结论必须由 `flyaim.stats.compare_all` 的预注册规则给出,不得手写
    - `visual_input_fallback` 必须显式声明(说明"果蝇的眼睛"是否为真实生物接线)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from flyaim.stats import compare_all, summarize_arm

ARM_LABEL = {
    "fly": "果蝇连接组 + 读出",
    "shuffle": "打乱接线(零模型)",
    "pid": "PID(偷看靶位,性能上限参照)",
    "random": "均匀随机(下界)",
}


def _fmt(v, nd=4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if np.isnan(v):
            return "—"
        return f"{v:.{nd}f}"
    return str(v)


def build_report(
    payload: dict,
    stats: dict,
    manifest: dict | None = None,
    extra_notes: list[str] | None = None,
) -> str:
    cfg = payload.get("config", {})
    by_arm: dict[str, list[dict]] = payload.get("results", {})
    timings = payload.get("timings", {})

    lines: list[str] = []
    A = lines.append

    A("# FlyAim 实验结果报告")
    A("")
    A("> 本报告由 `flyaim/report.py` 自动生成。结论来自预先注册的判定规则,非人工撰写。")
    A("")

    # ---------------------------------------------------------- 数据溯源
    A("## 1. 数据溯源")
    A("")
    if manifest:
        A(f"- 数据集:**{manifest.get('dataset', '未知')}** ({manifest.get('license', '?')})")
        A(f"- 神经元总数:**{manifest.get('n_neurons', '?'):,}**")
        A(f"- 连接边数:**{manifest.get('n_edges', '?'):,}**")
        roles = manifest.get("roles", {})
        if roles:
            A(f"- 下行神经元(DN):**{roles.get('n_descending', '?')}** 个 —— 唯一合法控制读出群")
            A(f"- 运动神经元(MN):{roles.get('n_motor', '?')} 个")
            A(f"- 抑制性神经元:{roles.get('n_inhibitory', '?')} 个")
        A("")
        fb = manifest.get("visual_input_fallback", True)
        A("### 视觉输入端的真实性声明")
        A("")
        if fb:
            A("**`visual_input_fallback = True`**:未能在数据集中定位到真实感光细胞群,"
              "本项目的「复眼」是**人工设计的编码器**,**不是**果蝇真实的生物接线。")
            A("")
            A("这意味着:**系统的视觉能力上限由编码器设计决定,而非由果蝇神经回路决定**。"
              "任何「果蝇在看画面」的表述都必须附带这一限制。")
        else:
            A(f"**`visual_input_fallback = False`**,策略:`{manifest.get('visual_input_strategy')}`。")
            A("")
            A("视觉输入驱动的是数据集内**真实存在的感光细胞群**(`superclass == 'ol_sensory'`)。"
              "但这仍不等于完整的生物视觉系统,须注意以下边界:")
            A("")
            A("- 编码器决定「像素 → 小眼照度」的映射方式,**这一步是人为设计的**;")
            A("- 数据集未提供感光细胞的光谱敏感曲线与增益,色通道映射为近似;")
            A("- 下游的 lamina/medulla 处理虽由真实接线图承载,但**神经元模型是 LIF 简化模型**,"
              "不含真实电导、递质动力学与神经调质细节。")
        A("")
        notes = manifest.get("notes") or []
        if notes:
            A("数据侧备注:")
            for n in notes:
                A(f"- {n}")
            A("")
        sf = manifest.get("source_files") or {}
        if sf:
            A("| 源文件 | 字节 | sha256(前 16) |")
            A("|---|---:|---|")
            for name, meta in sf.items():
                A(f"| {name} | {meta.get('bytes', 0):,} | `{str(meta.get('sha256', ''))[:16]}` |")
            A("")

    # ---------------------------------------------------------- 实验设定
    A("## 2. 实验设定")
    A("")
    A(f"- 种子数:**{len(cfg.get('seeds', []))}** (要求 ≥ 10)")
    A(f"- 每 episode 帧数:**{cfg.get('frames_per_episode', '?')}**")
    A(f"- 对照臂:{', '.join(cfg.get('arms', []))}")
    arena = cfg.get("arena", {})
    if arena:
        A(f"- 靶场:{arena.get('width')}×{arena.get('height')},靶半径 {arena.get('target_radius')}px,"
          f"准星速度 {arena.get('speed_px_per_action')}px/帧")
    brain = cfg.get("brain", {})
    if brain:
        sub = brain.get("subnet_size")
        A(f"- 仿真:dt={brain.get('dt_ms')}ms,每帧 {brain.get('steps_per_frame')} 步,"
          f"子网={'全量' if sub is None else f'{sub:,} 神经元'}")
    A("")

    # ---------------------------------------------------------- 主结果表
    A("## 3. 主结果")
    A("")
    A("| 控制臂 | 命中率 mean±std | 中位数 | 平均靶距(px) | 首中耗时(ms) | 闭环 FPS | 耗时(s) |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for arm in cfg.get("arms", []):
        rs = by_arm.get(arm)
        if not rs:
            continue
        hr = summarize_arm(rs, "hit_rate")
        md = summarize_arm(rs, "mean_target_dist_px")
        tf = summarize_arm(rs, "time_to_first_hit_ms")
        fps = summarize_arm(rs, "fps_loop")
        label = ARM_LABEL.get(arm, arm)
        A(
            f"| {label} | {_fmt(hr.get('mean'),4)} ± {_fmt(hr.get('std'),4)} "
            f"| {_fmt(hr.get('median'),4)} | {_fmt(md.get('mean'),1)} "
            f"| {_fmt(tf.get('mean'),1)} | {_fmt(fps.get('mean'),1)} "
            f"| {_fmt(timings.get(arm),1)} |"
        )
    A("")

    # ---------------------------------------------------------- 预注册判定
    A("## 4. 预注册判定(核心结论)")
    A("")
    primary = stats.get("primary")
    if primary:
        A(f"**{primary['verdict']}**")
        A("")
        A(f"- 配对样本数:{primary['n_pairs']}")
        A(f"- 配对差值均值:{_fmt(primary['mean_diff'],4)} "
          f"(95% CI [{_fmt(primary['ci95'][0],4)}, {_fmt(primary['ci95'][1],4)}])")
        A(f"- Cohen's dz:{_fmt(primary['cohens_dz'],3)}")
        A(f"- p 值:{primary['p_value']:.4g}(α = {primary['alpha']})")
    else:
        A("**未产出主判定**:缺少 `fly` 或 `shuffle` 臂的配对数据。")
    A("")
    A("判定规则在实验前注册于 `CONTRACT.md` 第 3 节:**若 `fly` 相对零模型 `shuffle` 无显著优势,"
      "则结论为「该接线图对本任务无因果贡献」。**")
    A("")

    # ---------------------------------------------------------- 全部比较
    A("## 5. 全部配对比较")
    A("")
    A(f"指标:`{stats.get('metric', 'hit_rate')}`")
    A("")
    tests = stats.get("tests", {})
    if tests:
        A("| 比较 | 差值均值 | 95% CI | dz | p | 显著 |")
        A("|---|---:|---|---:|---:|:--:|")
        for key, t in tests.items():
            ci = t["ci95"]
            A(
                f"| {t['arm_a']} vs {t['arm_b']} | {_fmt(t['mean_diff'],4)} "
                f"| [{_fmt(ci[0],4)}, {_fmt(ci[1],4)}] | {_fmt(t['cohens_dz'],2)} "
                f"| {t['p_value']:.4g} | {'✅' if t['significant'] else '❌'} |"
            )
        A("")
        A("> `pid` 臂直接读取靶位坐标,是**性能上限参照系**,不参与「果蝇是否有效」的判定。")
    else:
        A("无可用配对比较。")
    A("")

    # ---------------------------------------------------------- 学习曲线
    A("## 6. 学习曲线(是否越玩越准)")
    A("")
    A("`acq_curve` 把一个 episode 分成若干箱,给出每箱命中率。若曲线平坦,说明**没有学习发生**"
      "(这与「连接组是固定接线图、本身不含可塑性」的预期一致)。")
    A("")
    for arm in cfg.get("arms", []):
        rs = by_arm.get(arm)
        if not rs:
            continue
        curves = [r["summary"].get("acq_curve") for r in rs]
        curves = [c for c in curves if c]
        if not curves:
            continue
        minlen = min(len(c) for c in curves)
        if minlen == 0:
            continue
        arr = np.asarray([c[:minlen] for c in curves], dtype=np.float64)
        mean = arr.mean(axis=0)
        A(f"- **{ARM_LABEL.get(arm, arm)}**: " + " → ".join(f"{v:.2f}" for v in mean))
    A("")

    # ---------------------------------------------------------- 限制声明
    A("## 7. 限制与方法学声明")
    A("")
    A("1. **视觉前端的第一步是人为设计的**:即使驱动的是真实感光细胞群,"
      "「像素 → 小眼照度/色通道」的映射方式仍由编码器决定,不含果蝇真实的光谱敏感曲线。")
    A("2. **神经元模型是 LIF 简化模型**:连接组只给出「谁连谁、连多强」,"
      "不含真实膜电导、递质释放动力学与神经调质扩散。")
    A("3. **连接组权重未被修改**;任何学习只发生于读出层(若 `readout.mode == \"trained\"`)。")
    A("4. **动作语义是人为定义的**:DN 活动到「左右/上下」的映射由开发者设定,"
      "不是果蝇真实的运动映射。真实 DN 与转向的对应关系需要行为学实验才能确定。")
    A("5. 本实验是**离线自建靶场**,不涉及任何真实游戏客户端或在线对战。")
    A("6. 报告中的 p 值为配对 t 检验;若 scipy 缺失则退化为正态近似(小样本下偏乐观)。")
    A("")
    if extra_notes:
        A("### 附加说明")
        A("")
        for n in extra_notes:
            A(f"- {n}")
        A("")

    return "\n".join(lines)


def write_report(path, payload: dict, stats: dict, manifest: dict | None = None,
                 extra_notes: list[str] | None = None) -> str:
    text = build_report(payload, stats, manifest, extra_notes)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return text


def compute_stats(payload: dict, metric: str = "hit_rate", alpha: float = 0.05) -> dict:
    """从 raw_results 计算全部统计量(报告与 CLI 共用)。"""
    return compare_all(
        payload.get("results", {}),
        reference="shuffle",
        test_arm="fly",
        metric=metric,
        alpha=alpha,
    )
