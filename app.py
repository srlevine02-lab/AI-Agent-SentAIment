"""
AccessIQ Review Operations — web UI for the multi-agent review app.

    pip install -r requirements.txt
    streamlit run app.py

On first load the app asks for an OpenAI API key. The key is kept only in this
browser session's memory and is never written to disk.
"""
from __future__ import annotations

import asyncio
import html
import io
import json
import os
import threading
from datetime import datetime

import pandas as pd
import streamlit as st

import evaluate as ev
import review_agents as ra

BASE = os.path.dirname(os.path.abspath(__file__))
ra.DATA_PATH = os.path.join(BASE, "accessiq_reviews_unlabeled.csv")
ev.LABELED_PATH = os.path.join(BASE, "accessiq_reviews_labeled.csv")

MODELS = ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4o"]
DEPT_LABEL = {"legal": "Legal & Product Safety", "shipping": "Shipping & Fulfillment",
              "app_digital": "App Management / Digital", "manufacturing": "Manufacturing & Hardware"}
DEPT_ICON = {"legal": "gavel", "shipping": "local_shipping", "app_digital": "smartphone", "manufacturing": "precision_manufacturing"}
DEPT_SHORT = {"legal": "Legal & Safety", "shipping": "Shipping", "app_digital": "App / Digital", "manufacturing": "Manufacturing"}
OFFER_LABEL = {"return_or_exchange": "Return or exchange", "exchange": "Exchange", "none": "No return offer"}
TOOL_LABEL = {"lookup_review": ("Looked up review", "search"),
              "analyze_sentiment": ("Sentiment Analyst", "psychology"),
              "route_to_department": ("Route to department", "alt_route"),
              "check_return_policy": ("Checked return policy", "policy"),
              "draft_customer_reply": ("Reply Writer drafted reply", "edit_note"),
              "post_customer_reply": ("Post customer reply", "send")}

st.set_page_config(page_title="AccessIQ Review Ops", page_icon=":material/shield_lock:", layout="wide")

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton button, input, textarea { font-family: 'Inter', sans-serif; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 4.2rem; max-width: 1280px; }
.aq-head { display:flex; justify-content:space-between; align-items:flex-end; gap:1rem; flex-wrap:wrap;
           border-bottom:1px solid #E5E9F2; padding-bottom:.9rem; margin-bottom:1.1rem; }
.aq-head h1 { font-size:1.55rem; font-weight:700; color:#0F172A; margin:0; letter-spacing:-.01em; }
.aq-head p { color:#5B6474; margin:.2rem 0 0; font-size:.93rem; }
.aq-chips { display:flex; gap:.4rem; flex-wrap:wrap; }
.aq-chip { font-size:.76rem; font-weight:600; padding:.28rem .62rem; border-radius:999px; background:#F1F4F9;
           color:#334155; border:1px solid #E2E8F0; white-space:nowrap; }
.aq-chip.ok { background:#E8F7EF; color:#146C43; border-color:#C5EBD5; }
.aq-chip.warn { background:#FFF4E5; color:#9A5B00; border-color:#FBDDB0; }
.aq-badge { display:inline-block; font-size:.72rem; font-weight:600; padding:.18rem .55rem; border-radius:6px;
            text-transform:uppercase; letter-spacing:.03em; }
.b-positive{background:#E8F7EF;color:#146C43}.b-negative{background:#FDECEC;color:#B42318}
.b-mixed{background:#FFF4E5;color:#9A5B00}.b-neutral{background:#EEF2F7;color:#475467}
.b-urgent{background:#B42318;color:#fff}.b-high{background:#FDECEC;color:#B42318}
.b-normal{background:#EEF2F7;color:#344054}.b-low{background:#F4F6FA;color:#667085}
.b-route{background:#EAF0FF;color:#2443B3}.b-reply{background:#F3EBFF;color:#6B2FBF}
.b-offer{background:#E8F7EF;color:#146C43}.b-nooffer{background:#EEF2F7;color:#475467}
.aq-review { background:#F7F9FC; border:1px solid #E5E9F2; border-radius:10px; padding:.85rem 1rem; }
.aq-review .t { font-weight:600; color:#0F172A; margin-bottom:.25rem; }
.aq-review .m { color:#667085; font-size:.8rem; margin-bottom:.45rem; }
.aq-review .c { color:#1F2937; font-size:.92rem; line-height:1.5; }
.aq-stars { color:#F5A524; letter-spacing:1px; } .aq-stars span { color:#D0D5DD; }
.aq-reply { border-left:3px solid #7C4DDB; background:#FBF9FF; padding:.85rem 1rem; border-radius:0 10px 10px 0;
            color:#1F2937; font-size:.93rem; line-height:1.55; white-space:pre-wrap; }
.aq-dept { font-size:1.15rem; font-weight:700; color:#0F172A; margin:.1rem 0 .15rem; }
.aq-muted { color:#667085; font-size:.84rem; }
.aq-kv { font-size:.88rem; color:#344054; margin:.15rem 0; } .aq-kv b { color:#0F172A; font-weight:600; }
.aq-step { display:flex; gap:.6rem; align-items:flex-start; padding:.38rem 0; border-bottom:1px dashed #E5E9F2; font-size:.88rem; }
.aq-step:last-child { border-bottom:none; }
.aq-step .n { min-width:1.5rem; height:1.5rem; border-radius:50%; background:#EAF0FF; color:#2443B3; font-weight:700;
              font-size:.75rem; display:flex; align-items:center; justify-content:center; }
.aq-step .d { color:#667085; font-size:.82rem; }
.aq-res { font-size:.72rem; font-weight:600; padding:.08rem .45rem; border-radius:5px; margin-left:.35rem; }
.r-ok{background:#E8F7EF;color:#146C43}.r-bad{background:#FDECEC;color:#B42318}.r-wait{background:#FFF4E5;color:#9A5B00}
.aq-empty { text-align:center; padding:3rem 1rem; color:#667085; border:1px dashed #D0D5DD; border-radius:12px; }
.aq-empty b { display:block; color:#0F172A; font-size:1.05rem; margin-bottom:.3rem; }
.aq-hero { padding:3.5rem 2rem; border-radius:16px; background:linear-gradient(135deg,#0F2A5C 0%,#2F5BEA 100%); color:#fff; }
.aq-hero h1 { color:#fff; font-size:2rem; margin:0 0 .4rem; } .aq-hero p { color:#DCE5FF; max-width:640px; }
div[data-testid="stMetric"] { background:#fff; border:1px solid #E5E9F2; border-radius:12px; padding:.75rem 1rem; }
div[data-testid="stMetricLabel"] p { color:#5B6474; font-weight:500; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Background event loop: agent runs keep going between page refreshes
# ---------------------------------------------------------------------------
@st.cache_resource
def _loop() -> asyncio.AbstractEventLoop:
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True, name="agents-loop").start()
    return loop


@st.cache_resource
def _limiter() -> asyncio.Semaphore:
    return asyncio.Semaphore(3)


def submit(coro_fn, bump: dict) -> None:
    async def runner():
        try:
            async with _limiter():
                await coro_fn()
        finally:
            bump["v"] += 1
    asyncio.run_coroutine_threadsafe(runner(), _loop())


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("api_key", None)
ss.setdefault("model", ra.MODEL)
ss.setdefault("jobs", [])
ss.setdefault("bump", {"v": 0})
ss.setdefault("seen_v", 0)
ss.setdefault("decisions", {})
ss.setdefault("eval", {"running": False, "done": 0, "total": 0, "result": None, "error": None, "recent": []})
if "ctx" not in ss:
    ss.ctx = ra.AppContext(reviews=ra.load_reviews(ra.DATA_PATH))
if ss.api_key:  # re-apply on every rerun (cheap) so the agents always use this session's settings
    ra.configure(api_key=ss.api_key, model=ss.model)


def esc(x) -> str:
    return html.escape(str(x or ""))


def stars(n: int) -> str:
    n = int(n)
    return f'<span class="aq-stars">{"★" * n}<span>{"★" * (5 - n)}</span></span>'


def review(rid: str) -> dict:
    return ss.ctx.reviews.get(rid, {})


def review_html(rid: str, full: bool = True) -> str:
    r = review(rid)
    if not r:
        return ""
    content = r["Review Content"] if full else r["Review Content"][:220] + ("…" if len(r["Review Content"]) > 220 else "")
    verified = "Verified purchase" if str(r["Verified Purchase"]).lower() == "true" else "Unverified"
    return (f'<div class="aq-review"><div class="t">{esc(r["Review Title"])}</div>'
            f'<div class="m">{stars(r["Review Rating"])} &nbsp;·&nbsp; {esc(rid)} &nbsp;·&nbsp; {esc(r["Review Date"])}'
            f' &nbsp;·&nbsp; {verified}</div><div class="c">{esc(content)}</div></div>')


def jobs_by(status: str) -> list:
    return [j for j in ss.jobs if j.status == status]


def pending_count() -> int:
    return sum(len(j.pending()) for j in jobs_by("awaiting_approval"))


def active() -> bool:
    return any(j.status in ("queued", "running") for j in ss.jobs) or ss.eval["running"]


def processed_ids() -> set:
    out = set()
    for j in ss.jobs:
        for t in j.trace:
            if t["tool"] == "lookup_review":
                out.add(t["args"].get("review_id"))
        if j.label.startswith("AIQ-"):
            out.add(j.label)
    return out


def start_jobs(ids: list[str]) -> None:
    for rid in ids:
        job = ra.ApprovalSession(f"Process review {rid}.", ss.ctx, label=rid)
        ss.jobs.insert(0, job)
        submit(job.start, ss.bump)


def start_instruction(text: str) -> None:
    job = ra.ApprovalSession(text, ss.ctx, label=text[:80])
    ss.jobs.insert(0, job)
    submit(job.start, ss.bump)


def decide(job, index: int, decision: str, feedback: str | None) -> None:
    key = id(job)
    d = ss.decisions.setdefault(key, {})
    d[index] = (decision, feedback)
    if len(d) >= len(job.pending()):
        decisions = ss.decisions.pop(key)
        job.status = "running"
        submit(lambda: job.resume(decisions), ss.bump)


# ---------------------------------------------------------------------------
# API key prompt (shown on open)
# ---------------------------------------------------------------------------
def validate_key(key: str) -> tuple[bool, str]:
    try:
        from openai import APIConnectionError, AuthenticationError, OpenAI
        OpenAI(api_key=key, timeout=12).models.list()
        return True, "ok"
    except AuthenticationError:
        return False, "OpenAI rejected this key. Check that it is complete and active."
    except APIConnectionError:
        return False, "Couldn't reach OpenAI to verify the key. Check your internet connection."
    except Exception as e:  # permission/quota errors still mean the key is real
        return True, f"Connected with a warning: {type(e).__name__}"


@st.dialog("Connect your OpenAI account", width="medium", dismissible=False, icon=":material/key:")
def key_dialog():
    st.markdown("AccessIQ Review Ops runs its agents on OpenAI models. Paste an API key to begin. "
                "The key stays in this browser session's memory only and is never saved to disk.")
    key = st.text_input("OpenAI API key", type="password", placeholder="sk-...", key="key_input")
    model = st.selectbox("Model", MODELS, index=MODELS.index(ss.model) if ss.model in MODELS else 0,
                         accept_new_options=True, help="All three agents use this model. You can change it later in the sidebar.")
    env_key = os.getenv("OPENAI_API_KEY")
    use_env = env_key and st.checkbox("Use the key from my OPENAI_API_KEY environment variable")
    skip = st.checkbox("Skip verification", help="Connect without a test call to OpenAI (for offline setups).")
    if st.button("Connect", type="primary", width="stretch", icon=":material/link:"):
        k = (env_key if use_env else key or "").strip()
        if not k:
            st.error("Paste an API key first.")
            return
        ok, msg = (True, "ok") if skip else validate_key(k)
        if not ok:
            st.error(msg)
            return
        ss.api_key, ss.model = k, model
        ra.configure(api_key=k, model=model)
        if msg != "ok":
            st.toast(msg, icon=":material/warning:")
        st.rerun()
    st.caption("Need a key? Create one at platform.openai.com/api-keys.")


# ---------------------------------------------------------------------------
# Shared chrome
# ---------------------------------------------------------------------------
def header(title: str, subtitle: str) -> None:
    n = pending_count()
    chips = [f'<span class="aq-chip ok">● Connected</span>' if ss.api_key else '<span class="aq-chip warn">Not connected</span>',
             f'<span class="aq-chip">{esc(ss.model)}</span>']
    if n:
        chips.append(f'<span class="aq-chip warn">{n} awaiting approval</span>')
    running = len([j for j in ss.jobs if j.status in ("queued", "running")])
    if running:
        chips.append(f'<span class="aq-chip">{running} agent run(s) in progress</span>')
    st.markdown(f'<div class="aq-head"><div><h1>{esc(title)}</h1><p>{esc(subtitle)}</p></div>'
                f'<div class="aq-chips">{"".join(chips)}</div></div>', unsafe_allow_html=True)


def sidebar() -> None:
    with st.sidebar:
        st.markdown("### :material/shield_lock: AccessIQ Review Ops")
        st.caption("Multi-agent review triage for the AccessIQ K1 smart lock")
        st.divider()
        st.markdown("**Connection**")
        st.markdown(f":green-badge[:material/check_circle: Key ends …{esc(ss.api_key[-4:])}]")
        new_model = st.selectbox("Model", MODELS, index=MODELS.index(ss.model) if ss.model in MODELS else 0,
                                 accept_new_options=True, key="model_select")
        if new_model != ss.model:
            ss.model = new_model
            ra.configure(model=new_model)
            st.toast(f"Agents now use {new_model}")
        if st.button("Change API key", icon=":material/key:", width="stretch"):
            ss.api_key = None
            st.rerun()
        st.divider()
        st.markdown("**Agents**")
        st.markdown("- **Review Operations Manager** — plans and calls tools\n"
                    "- **Sentiment Analyst** — sentiment, issues, evidence\n"
                    "- **Reply Writer** — sympathetic replies")
        st.markdown("**Human approval required for**")
        st.markdown("- Routing to a department\n- Posting a customer reply")
        st.divider()
        if st.button("Clear session activity", icon=":material/restart_alt:", width="stretch"):
            ss.jobs, ss.decisions = [], {}
            ss.ctx = ra.AppContext(reviews=ra.load_reviews(ra.DATA_PATH))
            st.rerun()


@st.fragment(run_every=1.5)
def live_refresh() -> None:
    """While agents run in the background, refresh the page when something changes."""
    if ss.bump["v"] != ss.seen_v:
        ss.seen_v = ss.bump["v"]
        st.rerun(scope="app")


def empty(title: str, body: str) -> None:
    st.markdown(f'<div class="aq-empty"><b>{esc(title)}</b>{esc(body)}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_queue() -> None:
    header("Review queue", "Select reviews and let the agents analyze, route and draft replies. You approve every action.")
    reviews = list(ss.ctx.reviews.values())
    done_ids = processed_ids()
    c = st.columns(5)
    c[0].metric("Reviews in dataset", f"{len(reviews):,}")
    c[1].metric("Rated 3★ or less", f"{sum(int(r['Review Rating']) <= 3 for r in reviews):,}")
    c[2].metric("Processed this session", len(done_ids))
    c[3].metric("Awaiting your approval", pending_count())
    c[4].metric("Replies posted", len(ss.ctx.replies))

    f1, f2, f3, f4 = st.columns([1.3, 1, 2, 0.9])
    ratings = f1.multiselect("Star rating", [1, 2, 3, 4, 5], default=[1, 2, 3])
    status = f2.selectbox("Status", ["All", "Not processed", "Processed"])
    q = f3.text_input("Search", placeholder="Search titles and review text…")
    verified_only = f4.toggle("Verified only")

    df = pd.DataFrame([{
        "Review ID": r["Review Id"], "Date": r["Review Date"], "Rating": int(r["Review Rating"]),
        "Status": "Processed" if r["Review Id"] in done_ids else "New",
        "Verified": str(r["Verified Purchase"]).lower() == "true",
        "Title": r["Review Title"], "Review": r["Review Content"]} for r in reviews])
    if ratings:
        df = df[df.Rating.isin(ratings)]
    if status != "All":
        df = df[df.Status == ("Processed" if status == "Processed" else "New")]
    if q:
        df = df[df.Title.str.contains(q, case=False) | df.Review.str.contains(q, case=False)]
    if verified_only:
        df = df[df.Verified]
    df = df.sort_values("Date", ascending=False).reset_index(drop=True)

    sel = st.dataframe(
        df, hide_index=True, height=420, on_select="rerun", selection_mode="multi-row", key="queue_table",
        column_config={
            "Rating": st.column_config.NumberColumn("Rating", format="%d ★", width="small"),
            "Review": st.column_config.TextColumn("Review", width="large"),
            "Title": st.column_config.TextColumn("Title", width="medium"),
            "Verified": st.column_config.CheckboxColumn("Verified", width="small"),
            "Status": st.column_config.TextColumn("Status", width="small"),
        })
    chosen = df.iloc[sel.selection.rows]["Review ID"].tolist() if sel and sel.selection.rows else []

    b1, b2, b3 = st.columns([1.4, 1.6, 3])
    if b1.button(f"Run agents on {len(chosen)} selected", type="primary", disabled=not chosen,
                 icon=":material/play_arrow:", width="stretch"):
        start_jobs(chosen)
        st.toast(f"Started {len(chosen)} agent run(s). Approvals will appear as they're needed.")
        st.rerun()
    new_neg = [r for r in df[(df.Rating <= 3) & (df.Status == "New")]["Review ID"].tolist()][:3]
    if b2.button("Quick start: 3 newest negatives", disabled=not new_neg, icon=":material/bolt:", width="stretch"):
        start_jobs(new_neg)
        st.rerun()
    b3.caption(f"Showing {len(df):,} reviews. Up to 3 agent runs execute at once; the rest wait in line.")

    with st.expander("Give the manager agent an instruction", icon=":material/chat:"):
        text = st.text_area("Instruction", placeholder="e.g. Process reviews AIQ-0007 and AIQ-0019, then summarize the main issues.",
                            label_visibility="collapsed")
        if st.button("Send to manager", disabled=not text.strip(), icon=":material/send:"):
            start_instruction(text.strip())
            st.toast("Instruction sent.")
            st.rerun()


def page_approvals() -> None:
    header("Approvals", "The agents pause here. Approve, reject, or send a proposal back with feedback.")
    waiting = jobs_by("awaiting_approval")
    if not waiting:
        if any(j.status in ("queued", "running") for j in ss.jobs):
            empty("Agents are working…", "New approval requests will appear here automatically.")
        else:
            empty("Nothing to approve", "Run the agents on some reviews from the Review queue.")
        return
    for job in waiting:
        for p in job.pending():
            if p["index"] in ss.decisions.get(id(job), {}):
                continue
            a, rid = p["args"], p["args"].get("review_id", job.label)
            with st.container(border=True):
                is_route = p["tool"] == "route_to_department"
                kind = ('<span class="aq-badge b-route">Department routing</span>' if is_route
                        else '<span class="aq-badge b-reply">Customer reply</span>')
                st.markdown(f'{kind} &nbsp;<span class="aq-muted">Review {esc(rid)} · requested by the Review Operations Manager</span>',
                            unsafe_allow_html=True)
                left, right = st.columns([1, 1.15], gap="large")
                with left:
                    st.markdown('<div class="aq-muted" style="margin:.4rem 0 .35rem">Customer review</div>', unsafe_allow_html=True)
                    st.markdown(review_html(rid), unsafe_allow_html=True)
                with right:
                    st.markdown('<div class="aq-muted" style="margin:.4rem 0 .35rem">Proposed action</div>', unsafe_allow_html=True)
                    if is_route:
                        d = a.get("department", "")
                        pr = a.get("priority", "normal")
                        st.markdown(f'<div class="aq-dept">{esc(DEPT_LABEL.get(d, d))}</div>'
                                    f'<div class="aq-muted">{esc(ra.DEPARTMENTS.get(d, ""))}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="aq-kv"><b>Priority</b> &nbsp;<span class="aq-badge b-{esc(pr)}">{esc(pr)}</span></div>'
                                    f'<div class="aq-kv"><b>Issue</b> &nbsp;{esc(a.get("issue_category", "").replace("_", " "))}</div>'
                                    f'<div class="aq-kv"><b>Summary</b> &nbsp;{esc(a.get("summary"))}</div>', unsafe_allow_html=True)
                    else:
                        off = a.get("offer", "none")
                        st.markdown(f'<span class="aq-badge {"b-nooffer" if off == "none" else "b-offer"}">'
                                    f'{esc(OFFER_LABEL.get(off, off))}</span>', unsafe_allow_html=True)
                        st.markdown(f'<div class="aq-reply" style="margin-top:.5rem">{esc(a.get("reply_text"))}</div>',
                                    unsafe_allow_html=True)
                        pol = ss.ctx.policy_checks.get(rid)
                        if pol:
                            st.caption(f"Policy check: {pol['reason']}")
                k = f"{id(job)}_{p['index']}_{len(job.approvals)}"
                fb = st.text_input("Feedback for the agent (optional)", key=f"fb_{k}",
                                   placeholder="e.g. Mark this urgent, or keep the reply under 70 words")
                c1, c2, c3, _ = st.columns([1, 1.25, 1, 2.2])
                if c1.button("Approve", type="primary", key=f"ap_{k}", icon=":material/check:", width="stretch"):
                    decide(job, p["index"], "approve", None)
                    st.rerun()
                if c2.button("Send back with feedback", key=f"fbk_{k}", icon=":material/undo:", width="stretch",
                             disabled=not fb.strip()):
                    decide(job, p["index"], "reject", fb.strip())
                    st.rerun()
                if c3.button("Reject", key=f"rj_{k}", icon=":material/close:", width="stretch"):
                    decide(job, p["index"], "reject", None)
                    st.rerun()


def result_tag(t: dict) -> str:
    out = t.get("output")
    if out is None:
        return '<span class="aq-res r-wait">pending</span>' if t["tool"] in ("route_to_department", "post_customer_reply") else ""
    if "BLOCKED" in out:
        return '<span class="aq-res r-bad">blocked by guardrail</span>'
    if '"routed"' in out or '"posted"' in out:
        return '<span class="aq-res r-ok">approved & done</span>'
    if t["tool"] in ("route_to_department", "post_customer_reply"):
        return '<span class="aq-res r-bad">rejected</span>'
    return '<span class="aq-res r-ok">done</span>'


def step_detail(t: dict) -> str:
    a = t["args"]
    if t["tool"] == "lookup_review":
        return a.get("review_id", "")
    if t["tool"] == "route_to_department":
        return f'{DEPT_LABEL.get(a.get("department"), a.get("department"))} · {a.get("priority")} · {a.get("issue_category", "").replace("_", " ")}'
    if t["tool"] == "check_return_policy":
        try:
            return f'{a.get("issue_category", "").replace("_", " ")} → {OFFER_LABEL.get(json.loads(t["output"])["offer"])}'
        except Exception:
            return a.get("issue_category", "")
    if t["tool"] == "analyze_sentiment" and t.get("output"):
        try:
            o = json.loads(t["output"])
            issues = ", ".join(c["issue_category"].replace("_", " ") for c in o["complaints"]) or "no complaints"
            return f'{o["overall_sentiment"]} · {issues}'
        except Exception:
            return ""
    if t["tool"] == "post_customer_reply":
        return OFFER_LABEL.get(a.get("offer"), a.get("offer", ""))
    return ""


STATUS_BADGE = {"queued": ":gray-badge[Queued]", "running": ":blue-badge[:material/progress_activity: Running]",
                "awaiting_approval": ":orange-badge[:material/pending_actions: Awaiting approval]",
                "done": ":green-badge[:material/check_circle: Completed]", "error": ":red-badge[:material/error: Error]"}


def page_runs() -> None:
    header("Agent runs", "Every step each agent took, in order: the trajectory behind each decision.")
    if not ss.jobs:
        empty("No runs yet", "Start from the Review queue.")
        return
    for job in ss.jobs:
        out = job.final_output if isinstance(job.final_output, ra.ReviewReport) else None
        title = f"{job.label}"
        if out:
            title += f" — {out.overall_sentiment}"
        with st.expander(f"**{title}**", expanded=job.status in ("awaiting_approval", "running", "error") or job is ss.jobs[0]):
            st.markdown(f"{STATUS_BADGE[job.status]} &nbsp; started {job.created_at:%H:%M:%S} · {len(job.trace)} tool call(s)")
            left, right = st.columns([1.4, 1], gap="large")
            with left:
                if not job.trace:
                    st.caption("Waiting for the manager's first step…")
                rows = []
                for i, t in enumerate(job.trace, 1):
                    label, icon = TOOL_LABEL.get(t["tool"], (t["tool"], "build"))
                    rows.append(f'<div class="aq-step"><div class="n">{i}</div><div><b>{esc(label)}</b>{result_tag(t)}'
                                f'<div class="d">{esc(step_detail(t))}</div></div></div>')
                st.markdown("".join(rows), unsafe_allow_html=True)
                if job.error:
                    st.error(job.error)
            with right:
                if job.label.startswith("AIQ-"):
                    st.markdown(review_html(job.label, full=False), unsafe_allow_html=True)
                if out:
                    st.markdown(f'<div style="margin-top:.7rem"><span class="aq-badge b-{out.overall_sentiment}">{out.overall_sentiment}</span></div>'
                                f'<div class="aq-kv" style="margin-top:.45rem"><b>Issues</b> {esc(", ".join(i.replace("_", " ") for i in out.issues) or "none")}</div>'
                                f'<div class="aq-kv"><b>Routed to</b> {esc(", ".join(DEPT_LABEL[d] for d in out.departments_routed) or "not routed")}</div>'
                                f'<div class="aq-kv"><b>Reply</b> {"posted" if out.reply_posted else "none"} · {esc(OFFER_LABEL.get(out.offer))}</div>'
                                f'<div class="aq-muted" style="margin-top:.35rem">{esc(out.notes)}</div>', unsafe_allow_html=True)
                elif job.status == "done" and job.final_output:
                    st.write(job.final_output)
            if job.approvals:
                st.caption("Human decisions: " + " · ".join(
                    f'{TOOL_LABEL.get(x["tool"], (x["tool"],))[0]}: {x["decision"]}' + (f' ("{x["feedback"]}")' if x["feedback"] else "")
                    for x in job.approvals))


def page_outcomes() -> None:
    header("Outcomes", "What the agents actually did after approval: tickets, replies and sentiment.")
    ctx = ss.ctx
    reports = [j.final_output for j in ss.jobs if isinstance(j.final_output, ra.ReviewReport)]
    c = st.columns(5)
    c[0].metric("Reviews analyzed", len(reports))
    c[1].metric("Negative or mixed", sum(r.overall_sentiment in ("negative", "mixed") for r in reports))
    c[2].metric("Tickets routed", len(ctx.tickets))
    c[3].metric("Replies posted", len(ctx.replies))
    c[4].metric("Guardrail blocks", len(ctx.guardrail_blocks), help="Replies stopped because they broke the return policy")

    st.markdown("#### Tickets by department")
    dc = st.columns(4)
    for col, d in zip(dc, DEPT_LABEL):
        with col.container(border=True):
            st.markdown(f":material/{DEPT_ICON[d]}: **{DEPT_SHORT[d]}**")
            st.markdown(f"<div style='font-size:1.8rem;font-weight:700;color:#0F172A'>{sum(t['department'] == d for t in ctx.tickets)}</div>",
                        unsafe_allow_html=True)

    left, right = st.columns([1, 1.4], gap="large")
    with left:
        st.markdown("#### Sentiment of analyzed reviews")
        if reports:
            s = pd.Series([r.overall_sentiment for r in reports]).value_counts().reindex(
                ["positive", "mixed", "neutral", "negative"], fill_value=0)
            st.bar_chart(s.rename("Reviews"), color="#2F5BEA", height=260, horizontal=True, sort=False)
        else:
            empty("No data yet", "Analyzed reviews will appear here.")
    with right:
        st.markdown("#### Routed tickets")
        if ctx.tickets:
            tdf = pd.DataFrame(ctx.tickets)[["ticket_id", "review_id", "department", "priority", "issue_category", "summary"]]
            tdf["department"] = tdf["department"].map(DEPT_LABEL)
            tdf["issue_category"] = tdf["issue_category"].str.replace("_", " ")
            tdf.columns = ["Ticket", "Review", "Department", "Priority", "Issue", "Summary"]
            st.dataframe(tdf, hide_index=True, height=260)
        else:
            empty("No tickets yet", "Approved routings appear here.")

    st.markdown("#### Posted replies")
    if ctx.replies:
        for r in reversed(ctx.replies):
            with st.expander(f"{r['review_id']} · {OFFER_LABEL.get(r['offer'])} · {r['posted_at'][11:]}"):
                a, b = st.columns(2, gap="large")
                a.markdown(review_html(r["review_id"]), unsafe_allow_html=True)
                b.markdown(f'<div class="aq-reply">{esc(r["reply_text"])}</div>', unsafe_allow_html=True)
    else:
        empty("No replies yet", "Approved replies appear here.")

    if ctx.tickets or ctx.replies:
        d1, d2, _ = st.columns([1, 1, 3])
        d1.download_button("Download tickets (CSV)", pd.DataFrame(ctx.tickets).to_csv(index=False), "tickets.csv",
                           icon=":material/download:", disabled=not ctx.tickets, width="stretch")
        d2.download_button("Download replies (CSV)", pd.DataFrame(ctx.replies).to_csv(index=False), "replies.csv",
                           icon=":material/download:", disabled=not ctx.replies, width="stretch")


def start_eval(n: int, repeats: int, judge: bool) -> None:
    state, bump = ss.eval, ss.bump
    state.update(running=True, done=0, total=n * repeats, result=None, error=None, recent=[])

    def progress(done, total, row):
        state["done"], state["total"] = done, total
        state["recent"] = ([{k: row.get(k) for k in ("review_id", "type", "case_pass", "tool_sequence")}] + state["recent"])[:8]
        bump["v"] += 1

    async def go():
        try:
            state["result"] = await ev.run_evaluation(n, repeats, judge, 4, progress, out_dir=BASE)
        except Exception as e:
            state["error"] = f"{type(e).__name__}: {e}"
        finally:
            state["running"] = False

    asyncio.run_coroutine_threadsafe(_wrap(go, bump), _loop())


async def _wrap(fn, bump):
    try:
        await fn()
    finally:
        bump["v"] += 1


def page_eval() -> None:
    header("Evaluation", "Score the application at three scopes on held-out labeled reviews (lecture slides 34-37).")
    e = ss.eval
    with st.container(border=True):
        a, b, c, d = st.columns([1.6, 1, 1.2, 1.3])
        n = a.slider("Test cases", 30, 100, 40, help="Drawn from the held-out test split; negative and rare cases first.")
        repeats = b.number_input("Runs per case", 1, 3, 1, help="More than 1 also reports pass^k.")
        judge = c.toggle("LLM judge for replies", value=True)
        d.write("")
        if d.button("Run evaluation", type="primary", icon=":material/science:", width="stretch", disabled=e["running"]):
            start_eval(n, int(repeats), judge)
            st.rerun()
        st.caption("Uses about 6-8 model calls per case. Approvals are simulated during evaluation: the simulated "
                   "reviewer approves everything but sends every 4th reply back with feedback, to test revision.")

    if e["running"]:
        st.progress(e["done"] / max(1, e["total"]), text=f"Running… {e['done']} of {e['total']} runs scored")
        if e["recent"]:
            st.dataframe(pd.DataFrame(e["recent"]).rename(columns={"review_id": "Review", "type": "Type",
                         "case_pass": "Passed", "tool_sequence": "Tool sequence"}), hide_index=True)
        return
    if e["error"]:
        st.error(e["error"])
    res = e["result"]
    if not res:
        st.markdown("#### What gets measured")
        st.dataframe(pd.DataFrame([{"Scope": s, "Metric": k, "What it checks": d} for s, k, d in ev.METRICS]),
                     hide_index=True)
        return

    st.markdown(f"#### Results · {res['n_cases']} cases × {res['repeats']} run(s) · `{res['model']}`")
    k = st.columns(4)
    k[0].metric("Case pass rate", f"{res['case_pass']:.0%}", help="E2, T1 and T3 all pass")
    k[1].metric("Sentiment macro F1", f"{res['sent_f1']:.2f}")
    k[2].metric("Outcome correct (E2)", f"{res['summary']['E2_outcome_correct']:.0%}")
    k[3].metric("pass^k" if res["pass_k"] is not None else "Tool order correct (T1)",
                f"{(res['pass_k'] if res['pass_k'] is not None else res['summary']['T1_tool_order_correct']):.0%}")

    mdf = pd.DataFrame([{"Scope": s, "Metric": key, "What it checks": d, "Average": res["summary"][key]}
                        for s, key, d in ev.METRICS])
    st.dataframe(mdf, hide_index=True, column_config={
        "Average": st.column_config.ProgressColumn("Average", min_value=0, max_value=1, format="%.3f")})
    st.markdown("#### Interpretation")
    st.markdown("\n".join(f"- {line}" for line in res["interpretation"]))
    with st.expander("Per-case results"):
        rdf = pd.DataFrame(res["rows"])
        cols = ["review_id", "type", "rating", "expected_sentiment", "pred_sentiment", "expected_departments",
                "case_pass", "E2_outcome_correct", "T1_tool_order_correct", "T2_step_efficiency", "tool_sequence", "error"]
        st.dataframe(rdf[[c for c in cols if c in rdf.columns]], hide_index=True)
    x1, x2, _ = st.columns([1, 1, 3])
    buf = io.StringIO()
    pd.DataFrame(res["rows"]).to_csv(buf, index=False)
    x1.download_button("Per-case results (CSV)", buf.getvalue(), "eval_results.csv", icon=":material/download:", width="stretch")
    x2.download_button("Summary (Markdown)", res["markdown"], "eval_summary.md", icon=":material/download:", width="stretch")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if not ss.api_key:
    st.markdown('<div class="aq-hero"><h1>AccessIQ Review Operations</h1><p>Agents read every smart-lock review, '
                'measure sentiment, route complaints to Legal, Shipping, App/Digital or Manufacturing, and draft '
                'customer replies — with a person approving every action.</p></div>', unsafe_allow_html=True)
    key_dialog()
    st.stop()

sidebar()
n_pending = pending_count()
pages = [
    st.Page(page_queue, title="Review queue", icon=":material/inbox:", url_path="queue", default=True),
    st.Page(page_approvals, title=f"Approvals ({n_pending})" if n_pending else "Approvals",
            icon=":material/fact_check:", url_path="approvals"),
    st.Page(page_runs, title="Agent runs", icon=":material/account_tree:", url_path="runs"),
    st.Page(page_outcomes, title="Outcomes", icon=":material/monitoring:", url_path="outcomes"),
    st.Page(page_eval, title="Evaluation", icon=":material/science:", url_path="evaluation"),
]
st.navigation(pages, position="top").run()
if active():
    live_refresh()
