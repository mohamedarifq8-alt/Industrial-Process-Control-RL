"""لوحة SCADA/HMI لخزان كيميائي متحكم به عبر DQN."""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import torch

from dqn_model import DQN
from tank_env import ACTION_LABELS, TARGET_LEVEL, TankEnv, heuristic_action


BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_NAME = "tank_dqn_weights.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_state() -> None:
    if "env" not in st.session_state:
        st.session_state.env = TankEnv(seed=7)
        st.session_state.observation, _ = st.session_state.env.reset(seed=7)
    if "running" not in st.session_state:
        st.session_state.running = False
    if "history" not in st.session_state:
        st.session_state.history = {
            "المستوى": [],
            "الهدف": [],
            "السرعة": [],
        }
    if "model" not in st.session_state:
        st.session_state.model = None
        st.session_state.model_signature = None
        st.session_state.model_status = "لم يتم تحميل ملف أوزان؛ تعمل السياسة الاحتياطية."
        st.session_state.model_report = None


def reset_simulation() -> None:
    env: TankEnv = st.session_state.env
    st.session_state.observation, _ = env.reset(seed=7)
    st.session_state.running = False
    for series in st.session_state.history.values():
        series.clear()


def weight_source(uploaded: Any) -> tuple[Any, str]:
    if uploaded is not None:
        payload = io.BytesIO(uploaded.getvalue())
        return payload, f"upload:{uploaded.name}:{uploaded.size}"

    path = BASE_DIR / WEIGHTS_NAME
    if path.exists():
        return path, f"file:{path.stat().st_mtime_ns}:{path.stat().st_size}"
    return None, "fallback"


def load_model_if_needed(uploaded: Any) -> None:
    source, signature = weight_source(uploaded)
    if signature == st.session_state.model_signature:
        return

    st.session_state.model_signature = signature
    st.session_state.model = None
    st.session_state.model_report = None
    if source is None:
        st.session_state.model_status = (
            "لم يُعثر على tank_dqn_weights.pth؛ السياسة الاحتياطية فعالة."
        )
        return

    try:
        model = DQN().to(DEVICE)
        report = model.load_checkpoint(source, device=DEVICE)
        st.session_state.model = model
        st.session_state.model_report = report
        if report["missing"] or report["unexpected"]:
            st.session_state.model_status = (
                "تم تحميل الملف جزئياً؛ راجع مفاتيح الأوزان غير المتطابقة."
            )
        else:
            st.session_state.model_status = (
                f"تم تحميل أوزان DQN بنجاح على {DEVICE.type.upper()}."
            )
    except (OSError, RuntimeError, ValueError, EOFError) as exc:
        st.session_state.model_status = f"تعذر تحميل الأوزان: {exc}"


def choose_action(observation: np.ndarray) -> tuple[int, str]:
    model: DQN | None = st.session_state.model
    if model is None:
        action = heuristic_action(observation)
        return action, "سياسة احتياطية"
    action = model.act(observation, device=DEVICE)
    return action, "DQN"


def append_history(env: TankEnv) -> None:
    history = st.session_state.history
    history["المستوى"].append(env.level)
    history["الهدف"].append(TARGET_LEVEL)
    history["السرعة"].append(env.velocity)
    max_points = 120
    for key in history:
        history[key] = history[key][-max_points:]


def render_tank(env: TankEnv) -> None:
    fill = int(np.clip(env.level / 10.0, 0.03, 1.0) * 230)
    fill_y = 300 - fill
    level_color = "#22d3a7" if abs(env.level - TARGET_LEVEL) < 0.8 else "#f0a35e"
    status = "مستقر" if abs(env.level - TARGET_LEVEL) < 0.2 and abs(env.velocity) < 0.3 else "تصحيح آلي"
    svg = f"""
    <div class="tank-panel">
      <div class="tank-title"><span>مقطع الخزان T-101</span>
        <span class="status-pill">{status}</span></div>
      <svg viewBox="0 0 520 350" role="img" aria-label="مجسم الخزان">
        <defs>
          <linearGradient id="shell" x1="0" x2="1">
            <stop offset="0" stop-color="#1e3440"/><stop offset="0.5" stop-color="#294854"/>
            <stop offset="1" stop-color="#172933"/>
          </linearGradient>
          <linearGradient id="liquid" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0" stop-color="{level_color}" stop-opacity=".88"/>
            <stop offset="1" stop-color="#087f87" stop-opacity=".94"/>
          </linearGradient>
        </defs>
        <path d="M135 45 Q260 18 385 45 L385 292 Q260 327 135 292 Z"
          fill="url(#shell)" stroke="#6d8b94" stroke-width="3"/>
        <path d="M136 {fill_y} Q260 {fill_y-16} 384 {fill_y} L384 292
          Q260 327 136 292 Z" fill="url(#liquid)" opacity=".94"/>
        <ellipse cx="260" cy="45" rx="125" ry="25" fill="#18303a" stroke="#7899a2" stroke-width="3"/>
        <ellipse cx="260" cy="45" rx="78" ry="12" fill="#10232c" stroke="#4a6b75"/>
        <line x1="260" y1="45" x2="260" y2="176" stroke="#b1cbd0" stroke-width="4"/>
        <circle cx="260" cy="184" r="13" fill="{level_color}" stroke="#d5f7ef" stroke-width="3"/>
        <path d="M135 118 L75 118 L75 146 L28 146" fill="none" stroke="#71939c" stroke-width="9"/>
        <path d="M385 250 L445 250 L445 276 L492 276" fill="none" stroke="#71939c" stroke-width="9"/>
        <polygon points="54,138 54,154 72,146" fill="#22d3a7"/>
        <polygon points="465,268 465,284 447,276" fill="#f0a35e"/>
        <text x="28" y="105" fill="#a8c2c9" font-size="14">FEED</text>
        <text x="410" y="307" fill="#a8c2c9" font-size="14">OUTLET</text>
        <text x="260" y="333" text-anchor="middle" fill="#d9eef0" font-size="18"
          font-weight="700">{env.level:.2f} / 10.00  |  v={env.velocity:+.2f}</text>
      </svg>
    </div>
    """
    st.markdown(svg, unsafe_allow_html=True)


def render_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&display=swap');
        :root { --bg:#071116; --panel:#0d1f27; --line:#21404a; --text:#d9eef0;
                --muted:#86a5ad; --teal:#22d3a7; --amber:#f0a35e; }
        .stApp { background: radial-gradient(circle at 80% 0%, #12353e 0%, var(--bg) 42%);
                 color:var(--text); font-family:'IBM Plex Sans Arabic', sans-serif; }
        [data-testid="stHeader"] { background:transparent; }
        .block-container { max-width:1400px; padding-top:2rem; }
        h1, h2, h3, p, label { font-family:'IBM Plex Sans Arabic', sans-serif; }
        h1 { letter-spacing:-.03em; }
        [data-testid="stMetric"] { background:rgba(13,31,39,.84); border:1px solid var(--line);
          border-radius:12px; padding:12px 16px; }
        [data-testid="stMetricLabel"] { color:var(--muted); }
        [data-testid="stMetricValue"] { color:var(--text); }
        .hero { border:1px solid var(--line); border-radius:16px; padding:22px 26px;
          background:linear-gradient(100deg,rgba(17,50,59,.94),rgba(8,22,28,.7));
          margin-bottom:18px; }
        .eyebrow { color:var(--teal); text-transform:uppercase; letter-spacing:.16em;
          font-size:.72rem; font-weight:700; }
        .hero h1 { margin:.2rem 0 .2rem; font-size:2rem; }
        .hero p { margin:0; color:var(--muted); }
        .tank-panel, .section-card { border:1px solid var(--line); border-radius:14px;
          background:rgba(13,31,39,.7); padding:16px; }
        .tank-title { display:flex; justify-content:space-between; align-items:center;
          color:#cde4e7; font-weight:600; margin-bottom:4px; }
        .status-pill { border:1px solid rgba(34,211,167,.45); color:var(--teal);
          border-radius:999px; padding:3px 9px; font-size:.72rem; }
        .signal { display:flex; align-items:center; gap:8px; color:var(--muted);
          font-size:.83rem; margin-top:10px; }
        .signal-dot { width:8px; height:8px; border-radius:50%; background:var(--teal);
          box-shadow:0 0 12px var(--teal); }
        .signal-warn .signal-dot { background:var(--amber); box-shadow:0 0 12px var(--amber); }
        .small-note { color:var(--muted); font-size:.8rem; line-height:1.6; }
        div.stButton > button { border:1px solid var(--line); background:#102f38;
          color:var(--text); border-radius:8px; font-weight:600; }
        div.stButton > button:hover { border-color:var(--teal); color:var(--teal); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="T-101 | Industrial Control",
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    render_styles()
    init_state()

    st.markdown(
        """
        <div class="hero" dir="rtl">
          <div class="eyebrow">SCADA / HMI · T-101</div>
          <h1>وحدة التحكم الذكي في الخزان الكيميائي</h1>
          <p>مراقبة مستمرة للمستوى والسرعة مع قرار تحكم صادر من شبكة DQN.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    env: TankEnv = st.session_state.env
    with st.sidebar:
        st.markdown("### إعدادات العملية")
        uploaded = st.file_uploader(
            "ملف أوزان DQN",
            type=["pth", "pt"],
            help="يمكن رفع tank_dqn_weights.pth هنا أو وضعه بجانب main.py.",
        )
        load_model_if_needed(uploaded)
        st.caption(st.session_state.model_status)
        if st.session_state.model_report and (
            st.session_state.model_report["missing"]
            or st.session_state.model_report["unexpected"]
        ):
            st.warning(
                f"مفقود: {len(st.session_state.model_report['missing'])} · "
                f"غير متوقع: {len(st.session_state.model_report['unexpected'])}"
            )

        st.divider()
        start, stop = st.columns(2)
        with start:
            if st.button("بدء", width="stretch", type="primary"):
                st.session_state.running = True
        with stop:
            if st.button("إيقاف", width="stretch"):
                st.session_state.running = False
        if st.button("إعادة ضبط", width="stretch"):
            reset_simulation()
            st.rerun()
        st.markdown(
            '<div class="small-note">تعمل المحاكاة بخطوة زمنية 0.2 ثانية. '
            "القرار الاحتياطي لا يُستخدم إلا عند غياب أوزان متوافقة.</div>",
            unsafe_allow_html=True,
        )

    if st.session_state.running:
        action, controller = choose_action(st.session_state.observation)
        observation, reward, terminated, truncated, info = env.step(action)
        st.session_state.observation = observation
        append_history(env)
        st.session_state.last_action = action
        st.session_state.controller = controller
        if terminated or truncated:
            st.session_state.running = False

    info = env._get_info()
    st.markdown("#### القياسات اللحظية", unsafe_allow_html=True)
    metric_cols = st.columns(5)
    metric_cols[0].metric("المستوى", f"{env.level:.3f} / 10.0", f"{info['level_error']:+.3f}")
    metric_cols[1].metric("السرعة", f"{env.velocity:+.3f}", "قصور ذاتي")
    metric_cols[2].metric("خطأ المستوى", f"{abs(info['level_error']):.3f}")
    metric_cols[3].metric("المكافأة", f"{env.last_reward:.3f}")
    metric_cols[4].metric("الدورة", f"{env.step_count} / {env.max_steps}")

    left, right = st.columns([1.08, 1.35])
    with left:
        render_tank(env)
        controller = st.session_state.get("controller", "جاهز")
        action_index = st.session_state.get("last_action", env.last_action)
        st.markdown(
            f'<div class="signal"><span class="signal-dot"></span>'
            f'المتحكم: <b>{controller}</b> · الأمر: <b>{ACTION_LABELS[action_index]}</b></div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("#### الاتجاهات الحية")
        if len(st.session_state.history["المستوى"]) >= 2:
            chart_data = {
                key: np.asarray(values, dtype=np.float64)
                for key, values in st.session_state.history.items()
            }
            st.line_chart(
                chart_data,
                y=["المستوى", "الهدف", "السرعة"],
                height=285,
                width="stretch",
            )
        else:
            st.info("اضغط «بدء» لجمع نقطتين على الأقل وعرض الاتجاه الحي.")

    st.markdown("#### حالة النظام")
    status_cols = st.columns(3)
    status_cols[0].markdown(
        '<div class="section-card"><b>PLC / Runtime</b><br>'
        '<span class="signal"><span class="signal-dot"></span>متصل · دورة التحكم 200 ms</span></div>',
        unsafe_allow_html=True,
    )
    status_cols[1].markdown(
        f'<div class="section-card"><b>وضع التشغيل</b><br>'
        f'<span class="signal"><span class="signal-dot"></span>'
        f'{"تشغيل آلي" if st.session_state.running else "استعداد"}</span></div>',
        unsafe_allow_html=True,
    )
    status_cols[2].markdown(
        f'<div class="section-card"><b>سلامة العملية</b><br>'
        f'<span class="signal"><span class="signal-dot"></span>'
        f'{"تسريب مرصود" if env.leak_active else "ضمن الحدود"}</span></div>',
        unsafe_allow_html=True,
    )

    if st.session_state.running:
        time.sleep(0.15)
        st.rerun()


if __name__ == "__main__":
    main()