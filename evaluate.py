"""
Three-scope evaluation of the AccessIQ review agents (lecture slides 34-37).

Runs the full application on held-out test reviews (split == "test" in
accessiq_reviews_labeled.csv) and scores every run at three scopes:

END-TO-END  (input -> final report + what actually happened)
  E1 sentiment_accuracy   final report sentiment == ground-truth label            (code)
  E2 outcome_correct      right departments routed, reply posted iff it should be,
                          and the report matches the actions that really ran      (code)
  E3 reply_quality        LLM judge, 1-5 rubric scaled to 0-1, posted replies only (LLM judge)

TRAJECTORY  (the full ordered trace of tool calls)
  T1 tool_order_correct   lookup -> analyze -> [route(s), policy, draft -> post] in a valid order;
                          no routing/reply for 4-5 star or complaint-free reviews  (code)
  T2 step_efficiency      expected number of calls / actual calls (capped at 1)    (code)
  T3 task_completion      final approval state right: executed tickets and replies
                          match expectations after human approval                 (code)

COMPONENT  (one step in isolation)
  C1 routing_arg_f1       departments passed to route_to_department vs expected   (code)
  C2 policy_arg_correct   issue_category passed to check_return_policy is right   (code)
  C3 offer_arg_correct    offer passed to post_customer_reply == policy result     (code)
  C4 analyst_accuracy     Sentiment Analyst run alone: sentiment == label          (code)
  C5 evidence_grounded    share of the analyst's evidence quotes found verbatim in the review (code)

Human-in-the-loop during evaluation is simulated: the reviewer approves every request,
except that on every 4th actionable case it rejects the FIRST reply with feedback, so the
reject -> revise -> re-propose path is tested too.

Usage:
  export OPENAI_API_KEY=sk-...
  python evaluate.py                    # 40 test cases
  python evaluate.py --n 30 --no-judge  # cheaper run
  python evaluate.py --repeats 3        # also reports pass^k (slide 37)
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
from collections import Counter
from statistics import mean

from pydantic import BaseModel

from agents import Agent, Runner

import review_agents as app

LABELED_PATH = "accessiq_reviews_labeled.csv"
REJECT_FEEDBACK = ("Please make the reply shorter (under 70 words) and say the issue has been "
                   "escalated to the responsible team.")
SENTIMENTS = ["positive", "negative", "mixed", "neutral"]

# ---------------------------------------------------------------------------
# Ground truth derived from the hidden labels
# ---------------------------------------------------------------------------
DAMAGE_WORDS = ("crushed", "in the rain", "received matte black", "opened and retaped")


def expected_for(row: dict) -> dict:
    rating = int(row["Review Rating"])
    verified = str(row["Verified Purchase"]).lower() == "true"
    cats = [p.split(":")[0] for p in row["label_theme_sentiments"].split(";")
            if p.endswith(":negative")]
    if row["label_off_topic"] == "True" and rating <= 3:
        txt = row["Review Content"].lower()
        cats = ["shipping_damage_or_wrong_item" if any(w in txt for w in DAMAGE_WORDS)
                else "shipping_delay_or_order_issue"]
    act = rating <= 3 and bool(cats)
    # main complaint rule used by the manager: security outranks everything
    main_cats = ["security_concerns"] if "security_concerns" in cats else cats
    return {
        "rating": rating,
        "sentiment": row["label_overall_sentiment"],
        "should_act": act,
        "departments": sorted({app.ISSUE_TO_DEPARTMENT[c] for c in cats}) if act else [],
        "policy_categories": main_cats if act else [],
        "offers": sorted({app.return_policy(c, verified)["offer"] for c in main_cats}) if act else [],
    }


def select_cases(n: int) -> list[dict]:
    with open(LABELED_PATH, newline="", encoding="utf-8") as f:
        test = [r for r in csv.DictReader(f) if r["split"] == "test"]
    low = [r for r in test if int(r["Review Rating"]) <= 3]
    high = [r for r in test if int(r["Review Rating"]) >= 4]
    rarity = Counter(r["label_review_type"] for r in low)
    low.sort(key=lambda r: (rarity[r["label_review_type"]], r["Review Id"]))  # rare types first
    n_high = max(5, round(n * 0.25))
    picked = low[: n - n_high]
    picked += sorted(high, key=lambda r: (r["label_review_type"] == "positive", r["Review Id"]))[: n - len(picked)]
    return picked[:n]


# ---------------------------------------------------------------------------
# LLM judge for reply quality (end-to-end, open-ended output)
# ---------------------------------------------------------------------------
class JudgeScore(BaseModel):
    score: int
    reason: str


judge = Agent(
    name="Reply Quality Judge",
    model=app.MODEL,
    output_type=JudgeScore,
    instructions="""Grade a company's public reply to a negative product review on a 1-5 scale.
5 = acknowledges the customer's specific problem, sincere and professional, offer matches the
allowed offer exactly, no blame, no admission of legal liability, clear next step (contact support).
4 = good with a minor issue. 3 = generic or misses part of the complaint. 2 = wrong tone or
offer mismatch. 1 = harmful, rude, or promises something the allowed offer does not permit.
Return the integer score and a one-sentence reason.""",
)


# ---------------------------------------------------------------------------
# Scoring one run
# ---------------------------------------------------------------------------
def f1(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(gold)
    return 2 * p * r / (p + r)


def score_trajectory(trace: list[dict], exp: dict, rejected_reply: bool) -> dict:
    names = [t["tool"] for t in trace]
    acting = {"route_to_department", "check_return_policy", "draft_customer_reply", "post_customer_reply"}

    def first(n):
        return names.index(n) if n in names else None

    ok = bool(names) and names[0] == "lookup_review" and first("analyze_sentiment") is not None
    if ok:
        a = first("analyze_sentiment")
        ok = all(i > a for i, n in enumerate(names) if n in acting)
    if ok and not exp["should_act"]:
        ok = not any(n in acting for n in names)
    if ok and exp["should_act"]:
        p, d, post = first("check_return_policy"), first("draft_customer_reply"), first("post_customer_reply")
        ok = (first("route_to_department") is not None and None not in (p, d, post)
              and p < post and d < post)

    budget = 2
    if exp["should_act"]:
        budget += len(exp["departments"]) + 3 + (2 if rejected_reply else 0)
    efficiency = min(1.0, budget / max(1, len(names)))
    return {"T1_tool_order_correct": float(ok), "T2_step_efficiency": efficiency}


async def run_case(row: dict, reject_first_reply: bool, use_judge: bool) -> dict:
    exp = expected_for(row)
    ctx = app.AppContext(reviews=app.load_reviews(app.DATA_PATH))
    state = {"rejected": False}

    async def simulated_human(tool: str, args: dict):
        if tool == "post_customer_reply" and reject_first_reply and not state["rejected"]:
            state["rejected"] = True
            return "reject", REJECT_FEEDBACK
        return "approve", None

    error = None
    try:
        rec = await app.process_review(row["Review Id"], ctx, simulated_human)
        report, trace = rec.final_output, rec.trace
    except Exception as e:  # a crash is a failed case, not a crashed evaluation
        report, trace, error = None, [], f"{type(e).__name__}: {e}"
    rep = report if isinstance(report, app.ReviewReport) else None

    actual_depts = {t["department"] for t in ctx.tickets}
    posted = ctx.replies[-1] if ctx.replies else None
    s: dict = {"review_id": row["Review Id"], "type": row["label_review_type"], "rating": exp["rating"],
               "expected_sentiment": exp["sentiment"], "expected_departments": ";".join(exp["departments"]),
               "expected_offer": "/".join(exp["offers"]), "tool_sequence": " > ".join(t["tool"] for t in trace),
               "error": error}

    # ---- End-to-end
    s["pred_sentiment"] = rep.overall_sentiment if rep else ""
    s["E1_sentiment_accuracy"] = float(rep is not None and rep.overall_sentiment == exp["sentiment"])
    outcome = (actual_depts == set(exp["departments"]) and bool(posted) == exp["should_act"])
    honest = rep is not None and set(rep.departments_routed) == actual_depts and rep.reply_posted == bool(posted)
    s["E2_outcome_correct"] = float(outcome and honest)
    s["E3_reply_quality"] = None
    if posted and use_judge:
        prompt = (f"REVIEW ({exp['rating']} stars): {row['Review Content']}\n\n"
                  f"ALLOWED OFFER: {'/'.join(exp['offers']) or 'none'}\n\nREPLY:\n{posted['reply_text']}")
        try:
            j = (await Runner.run(judge, prompt)).final_output
            s["E3_reply_quality"] = max(1, min(5, j.score)) / 5
            s["judge_reason"] = j.reason
        except Exception as e:
            s["judge_reason"] = f"judge error: {e}"

    # ---- Trajectory
    s.update(score_trajectory(trace, exp, reject_first_reply and exp["should_act"]))
    completion = actual_depts == set(exp["departments"]) and len(ctx.replies) == (1 if exp["should_act"] else 0)
    if reject_first_reply and exp["should_act"]:
        # the approved reply must come from a re-proposal after the rejection
        completion = completion and sum(t["tool"] == "post_customer_reply" for t in trace) >= 2
    s["T3_task_completion"] = float(completion)

    # ---- Component (arguments of individual tool calls)
    routed_args = {t["args"].get("department") for t in trace if t["tool"] == "route_to_department"}
    s["C1_routing_arg_f1"] = f1(routed_args, set(exp["departments"]))
    pol = [t["args"].get("issue_category") for t in trace if t["tool"] == "check_return_policy"]
    s["C2_policy_arg_correct"] = (float(bool(pol) and pol[0] in exp["policy_categories"])
                                  if exp["should_act"] else None)
    s["C3_offer_arg_correct"] = (float(posted is not None and posted["offer"] in exp["offers"])
                                 if exp["should_act"] else None)
    s["guardrail_blocks"] = len(ctx.guardrail_blocks)
    s["case_pass"] = float(s["E2_outcome_correct"] == 1 and s["T1_tool_order_correct"] == 1
                           and s["T3_task_completion"] == 1)
    return s


async def analyst_component(row: dict) -> dict:
    """C4/C5: the Sentiment Analyst alone, with the review as its only input."""
    text = f"Title: {row['Review Title']}\nRating: {row['Review Rating']} stars\nReview: {row['Review Content']}"
    try:
        out = (await Runner.run(app.sentiment_analyst, text)).final_output
    except Exception:
        return {"C4_analyst_accuracy": 0.0, "C5_evidence_grounded": None}
    norm = lambda x: re.sub(r"\s+", " ", x.strip().strip('"').lower())
    quotes = [c.evidence_quote for c in out.complaints if c.evidence_quote.strip()]
    grounded = (mean(norm(q) in norm(row["Review Content"]) for q in quotes) if quotes else None)
    return {"C4_analyst_accuracy": float(out.overall_sentiment == row["label_overall_sentiment"]),
            "C5_evidence_grounded": grounded, "analyst_sentiment": out.overall_sentiment}


# ---------------------------------------------------------------------------
METRICS = [
    ("End-to-end", "E1_sentiment_accuracy", "Final report sentiment matches label"),
    ("End-to-end", "E2_outcome_correct", "Right departments + reply iff needed, report matches actions"),
    ("End-to-end", "E3_reply_quality", "LLM-judge reply quality (1-5 scaled to 0-1)"),
    ("Trajectory", "T1_tool_order_correct", "Valid tool order, no actions on 4-5 star reviews"),
    ("Trajectory", "T2_step_efficiency", "Expected calls / actual calls"),
    ("Trajectory", "T3_task_completion", "Correct executed state after human approval"),
    ("Component", "C1_routing_arg_f1", "Department arguments to route_to_department (F1)"),
    ("Component", "C2_policy_arg_correct", "issue_category argument to check_return_policy"),
    ("Component", "C3_offer_arg_correct", "offer argument to post_customer_reply"),
    ("Component", "C4_analyst_accuracy", "Sentiment Analyst alone: sentiment accuracy"),
    ("Component", "C5_evidence_grounded", "Analyst evidence quotes found verbatim in review"),
]


def avg(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return (mean(vals), len(vals)) if vals else (None, 0)


def macro_f1(gold: list[str], pred: list[str]) -> float:
    scores = []
    for c in SENTIMENTS:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        if tp + fp + fn:
            scores.append(0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return mean(scores) if scores else 0.0


def interpret(rows: list[dict], summary: dict) -> list[str]:
    lines = []
    scored = {k: v for k, v in summary.items() if v is not None}
    if scored:
        worst = min(scored, key=scored.get)
        best = max(scored, key=scored.get)
        lines.append(f"Strongest metric: {best} ({scored[best]:.2f}); weakest: {worst} ({scored[worst]:.2f}).")
    if summary.get("C4_analyst_accuracy") is not None and summary.get("E1_sentiment_accuracy") is not None:
        gap = summary["E1_sentiment_accuracy"] - summary["C4_analyst_accuracy"]
        lines.append(f"Sentiment in the final report is {abs(gap):.2f} {'above' if gap >= 0 else 'below'} the analyst run "
                     "alone, showing how much the manager changes or loses the analyst's label.")
    conf = Counter((r["expected_sentiment"], r["pred_sentiment"]) for r in rows
                   if r["expected_sentiment"] != r["pred_sentiment"] and r["pred_sentiment"])
    if conf:
        lines.append("Most common sentiment confusions (label -> predicted): " +
                     ", ".join(f"{g}->{p} x{n}" for (g, p), n in conf.most_common(3)) + ".")
    by_type = {}
    for r in rows:
        by_type.setdefault(r["type"], []).append(r["case_pass"])
    weak = sorted(((mean(v), t, len(v)) for t, v in by_type.items()), key=lambda x: x[0])[:2]
    lines.append("Lowest case-pass rates by review type: " +
                 ", ".join(f"{t} {m:.2f} (n={n})" for m, t, n in weak) + ".")
    blocks = sum(r["guardrail_blocks"] for r in rows)
    lines.append(f"The policy guardrail blocked {blocks} proposed repl(ies); each block is an agent "
                 "mistake caught before it reached a customer.")
    fails = [r["review_id"] for r in rows if r["case_pass"] < 1][:8]
    if fails:
        lines.append("Inspect these failing cases in eval_results.csv: " + ", ".join(fails) + ".")
    return lines


async def run_evaluation(n: int = 40, repeats: int = 1, use_judge: bool = True, concurrency: int = 4,
                         on_progress=None, out_dir: str = ".") -> dict:
    """Run the evaluation and return {'rows', 'summary', 'sent_f1', 'case_pass', 'pass_k', 'markdown', ...}.
    on_progress(done, total, row) is called after every finished run (used by the web UI)."""
    judge.model = app.MODEL if isinstance(judge.model, str) else judge.model
    cases = select_cases(n)
    acting = [c for c in cases if expected_for(c)["should_act"]]
    reject_ids = {c["Review Id"] for i, c in enumerate(acting) if i % 4 == 0}
    sem = asyncio.Semaphore(concurrency)
    total, done = len(cases) * repeats, 0

    async def guarded(row, rep):
        nonlocal done
        async with sem:
            r = await run_case(row, row["Review Id"] in reject_ids, use_judge)
            if rep == 0:
                r.update(await analyst_component(row))
            r["repeat"] = rep
            done += 1
            if on_progress:
                on_progress(done, total, r)
            return r

    rows = await asyncio.gather(*(guarded(c, k) for c in cases for k in range(repeats)))
    rows = sorted(rows, key=lambda r: (r["review_id"], r["repeat"]))

    summary = {k: avg(rows, k)[0] for _, k, _ in METRICS}
    first = [r for r in rows if r["repeat"] == 0]
    sent_f1 = macro_f1([r["expected_sentiment"] for r in rows], [r["pred_sentiment"] for r in rows])
    case_pass = mean(r["case_pass"] for r in rows)
    pass_k = None
    if repeats > 1:
        by_case = {}
        for r in rows:
            by_case.setdefault(r["review_id"], []).append(r["case_pass"])
        pass_k = mean(float(all(v)) for v in by_case.values())

    fields = sorted({k for r in rows for k in r}, key=lambda k: (k[0:1] not in "ECT" or k[1:2] not in "0123456789", k))
    with open(f"{out_dir}/eval_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)

    interpretation = interpret(first, summary)
    md = ["# AccessIQ review agents: evaluation results", "",
          f"Model: `{app.MODEL}` | test cases: {len(cases)} | runs per case: {repeats}", "",
          "| Scope | Metric | What it checks | Average | n |", "|---|---|---|---|---|"]
    for scope, key, desc in METRICS:
        v, cnt = avg(rows, key)
        md.append(f"| {scope} | {key} | {desc} | {'n/a' if v is None else f'{v:.3f}'} | {cnt} |")
    md += ["", f"- Sentiment macro F1 (end-to-end, 4 classes): **{sent_f1:.3f}**",
           f"- Case pass rate (E2, T1 and T3 all pass, slide 37): **{case_pass:.3f}**"]
    if pass_k is not None:
        md.append(f"- pass^{repeats} (case passes on every run): **{pass_k:.3f}**")
    md += ["", "## Interpretation (auto-generated starting point; edit before submitting)", ""]
    md += [f"- {l}" for l in interpretation]
    markdown = "\n".join(md) + "\n"
    with open(f"{out_dir}/eval_summary.md", "w", encoding="utf-8") as f:
        f.write(markdown)
    return {"rows": rows, "summary": summary, "sent_f1": sent_f1, "case_pass": case_pass, "pass_k": pass_k,
            "markdown": markdown, "interpretation": interpretation, "n_cases": len(cases),
            "n_actionable": len(acting), "n_rejections": len(reject_ids), "repeats": repeats, "model": app.MODEL}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="number of test cases (>= 30)")
    ap.add_argument("--repeats", type=int, default=1, help="runs per case; reports pass^k when > 1")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM judge (E3)")
    a = ap.parse_args()

    def progress(done, total, r):
        print(f"  [{done}/{total}] {r['review_id']} [{r['type']:<14}] pass={int(r['case_pass'])}  {r['tool_sequence']}")

    print(f"Evaluating {a.n} cases x {a.repeats} run(s)...")
    res = await run_evaluation(a.n, a.repeats, not a.no_judge, a.concurrency, progress)
    print("\n" + res["markdown"])
    print("Wrote eval_results.csv and eval_summary.md")


if __name__ == "__main__":
    asyncio.run(main())
