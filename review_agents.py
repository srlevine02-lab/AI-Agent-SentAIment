"""
AccessIQ Review Operations — multi-agent app built with the OpenAI Agents SDK.

Architecture (centralized / "manager with specialists as tools", lecture slide 22):

    Review Operations Manager  (decides which tool to call, in what order)
      ├── lookup_review            python tool   - reads the review record
      ├── analyze_sentiment        AGENT as tool - Sentiment Analyst (structured output)
      ├── route_to_department      python tool   - HUMAN APPROVAL required
      ├── check_return_policy      python tool   - deterministic return/exchange policy
      ├── draft_customer_reply     AGENT as tool - Reply Writer
      └── post_customer_reply      python tool   - HUMAN APPROVAL required + guardrail

Business rules the manager follows:
  * Every review gets a sentiment analysis.
  * Reviews rated 3 stars or less that contain a complaint are routed to the
    department(s) the complaint is aimed at: legal, shipping, app_digital,
    manufacturing. A person approves each routing before it happens.
  * Those negative reviews then get a sympathetic public reply. A return or
    exchange is offered only when the policy tool says the case qualifies.
    A person approves the reply before it is posted.
  * A tool guardrail blocks any reply whose offer exceeds the policy, or any reply
    to a 4-5 star review. Human approval cannot override the guardrail (slide 33).

Usage
  export OPENAI_API_KEY=sk-...
  python review_agents.py --review AIQ-0042             # one review, you approve in the terminal
  python review_agents.py --batch 5 --max-rating 3      # first 5 reviews rated <= 3
  python review_agents.py --instruction "Process review AIQ-0007 and AIQ-0019"
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Awaitable, Callable, Literal

from pydantic import BaseModel, Field

from agents import (
    Agent,
    RunContextWrapper,
    Runner,
    ToolGuardrailFunctionOutput,
    ToolInputGuardrailData,
    function_tool,
    set_default_openai_key,
    tool_input_guardrail,
)
from agents.items import ToolCallItem, ToolCallOutputItem

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
DATA_PATH = os.getenv("REVIEWS_CSV", "accessiq_reviews_unlabeled.csv")

# ---------------------------------------------------------------------------
# Business configuration
# ---------------------------------------------------------------------------
Department = Literal["legal", "shipping", "app_digital", "manufacturing"]
IssueCategory = Literal[
    "installation_difficulty", "lock_reliability", "battery_life", "keypad_usability",
    "build_quality", "app_connectivity", "security_concerns",
    "shipping_damage_or_wrong_item", "shipping_delay_or_order_issue", "other",
]
Offer = Literal["return_or_exchange", "exchange", "none"]
Sentiment = Literal["positive", "negative", "mixed", "neutral"]

DEPARTMENTS = {
    "legal": "Legal & Product Safety: security failures, unauthorized entry, safety risk, liability.",
    "shipping": "Shipping & Fulfillment: late/damaged/wrong deliveries, packaging, order, price and return-logistics issues.",
    "app_digital": "App Management / Digital: mobile app, Bluetooth, Wi-Fi bridge, firmware updates, notifications, activity log.",
    "manufacturing": "Manufacturing & Hardware Quality: lock mechanism, keypad hardware, batteries/power, build/finish, fit and installation hardware.",
}
ISSUE_TO_DEPARTMENT = {
    "installation_difficulty": "manufacturing", "lock_reliability": "manufacturing",
    "battery_life": "manufacturing", "keypad_usability": "manufacturing",
    "build_quality": "manufacturing", "app_connectivity": "app_digital",
    "security_concerns": "legal", "shipping_damage_or_wrong_item": "shipping",
    "shipping_delay_or_order_issue": "shipping", "other": "shipping",
}
# Return / exchange policy (the deterministic "rule" the agents must respect)
DEFECT_ISSUES = {"installation_difficulty", "lock_reliability", "battery_life",
                 "keypad_usability", "build_quality", "security_concerns"}


def return_policy(issue_category: str, verified_purchase: bool) -> dict:
    """Single source of truth for return/exchange eligibility."""
    if not verified_purchase:
        return {"offer": "none", "reason": "Purchase is not verified; invite the customer to contact support with their order number."}
    if issue_category in DEFECT_ISSUES:
        return {"offer": "return_or_exchange", "reason": "Hardware/product defect on a verified purchase qualifies for a return or exchange."}
    if issue_category == "shipping_damage_or_wrong_item":
        return {"offer": "exchange", "reason": "Damaged in transit or wrong item shipped: replace it with an exchange."}
    if issue_category == "app_connectivity":
        return {"offer": "none", "reason": "Software/app issue: offer technical support and firmware help, not a return."}
    return {"offer": "none", "reason": "Delivery timing, pricing or other non-product issue: apologize, no return or exchange."}


OFFER_RANK = {"none": 0, "exchange": 1, "return_or_exchange": 2}


# ---------------------------------------------------------------------------
# Run context: the review data plus the "business systems" the tools write to
# ---------------------------------------------------------------------------
@dataclass
class AppContext:
    reviews: dict[str, dict]
    tickets: list[dict] = field(default_factory=list)          # executed routings
    replies: list[dict] = field(default_factory=list)          # executed (posted) replies
    policy_checks: dict[str, dict] = field(default_factory=dict)
    guardrail_blocks: list[dict] = field(default_factory=list)


def load_reviews(path: str = DATA_PATH) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return {r["Review Id"]: r for r in csv.DictReader(f)}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@function_tool
def lookup_review(ctx: RunContextWrapper[AppContext], review_id: str) -> str:
    """Look up a customer review by its ID (e.g. AIQ-0042). Returns title, text, star rating,
    review date and whether it is a verified purchase."""
    r = ctx.context.reviews.get(review_id.strip())
    if r is None:
        return json.dumps({"error": f"Review {review_id} not found."})
    return json.dumps({
        "review_id": r["Review Id"], "title": r["Review Title"], "content": r["Review Content"],
        "rating": int(r["Review Rating"]), "date": r["Review Date"],
        "verified_purchase": str(r["Verified Purchase"]).lower() == "true",
        "product": r["Product Name"],
    })


@function_tool
def check_return_policy(ctx: RunContextWrapper[AppContext], review_id: str,
                        issue_category: IssueCategory) -> str:
    """Check whether the customer qualifies for a return or exchange under company policy.
    Call this BEFORE drafting a reply to a negative review. issue_category is the main
    complaint in the review."""
    r = ctx.context.reviews.get(review_id)
    if r is None:
        return json.dumps({"error": f"Review {review_id} not found."})
    verified = str(r["Verified Purchase"]).lower() == "true"
    decision = {"review_id": review_id, "issue_category": issue_category,
                "verified_purchase": verified, **return_policy(issue_category, verified)}
    ctx.context.policy_checks[review_id] = decision
    return json.dumps(decision)


@function_tool(needs_approval=True)
def route_to_department(ctx: RunContextWrapper[AppContext], review_id: str,
                        department: Department, issue_category: IssueCategory,
                        priority: Literal["low", "normal", "high", "urgent"],
                        summary: str) -> str:
    """Send a review rated 3 stars or less to the department responsible for the complaint.
    Requires human approval. Call once per distinct department.
    Departments: legal (security/safety), shipping (delivery, packaging, order issues),
    app_digital (app, Bluetooth, Wi-Fi, firmware), manufacturing (hardware, batteries,
    keypad, build quality, installation hardware)."""
    ticket = {"ticket_id": f"T-{len(ctx.context.tickets) + 1001}", "review_id": review_id,
              "department": department, "issue_category": issue_category,
              "priority": priority, "summary": summary,
              "created_at": datetime.now().isoformat(timespec="seconds")}
    ctx.context.tickets.append(ticket)
    return json.dumps({"status": "routed", **ticket})


@tool_input_guardrail
def reply_policy_guardrail(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
    """Hard limits on public replies. Runs before post_customer_reply executes."""
    app: AppContext = data.context.context
    args = json.loads(data.context.tool_arguments or "{}")
    rid, offer = args.get("review_id"), args.get("offer", "none")
    review = app.reviews.get(rid)

    def block(msg: str) -> ToolGuardrailFunctionOutput:
        app.guardrail_blocks.append({"review_id": rid, "reason": msg})
        return ToolGuardrailFunctionOutput.reject_content(message=f"BLOCKED by policy guardrail: {msg}")

    if review is None:
        return block(f"unknown review {rid}.")
    if int(review["Review Rating"]) > 3:
        return block("replies are only posted to reviews rated 3 stars or less.")
    policy = app.policy_checks.get(rid)
    if policy is None:
        return block("call check_return_policy for this review before posting a reply.")
    if OFFER_RANK[offer] > OFFER_RANK[policy["offer"]]:
        return block(f"offer '{offer}' exceeds what policy allows ('{policy['offer']}'). Revise the reply.")
    text = args.get("reply_text", "").lower()
    if policy["offer"] == "none" and any(w in text for w in ("refund", "return it", "exchange it", "send you a new", "replacement")):
        return block("reply text promises a return/refund/replacement that policy does not allow.")
    return ToolGuardrailFunctionOutput.allow()


@function_tool(needs_approval=True, tool_input_guardrails=[reply_policy_guardrail])
def post_customer_reply(ctx: RunContextWrapper[AppContext], review_id: str,
                        reply_text: str, offer: Offer) -> str:
    """Post the company's public reply to a negative review. Requires human approval.
    offer must match the result of check_return_policy."""
    rec = {"review_id": review_id, "offer": offer, "reply_text": reply_text,
           "posted_at": datetime.now().isoformat(timespec="seconds")}
    ctx.context.replies.append(rec)
    return json.dumps({"status": "posted", "review_id": review_id, "offer": offer})


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
class Complaint(BaseModel):
    issue_category: IssueCategory
    department: Department
    severity: Literal["low", "medium", "high"]
    evidence_quote: str = Field(description="Exact words copied from the review that show this complaint.")


class SentimentResult(BaseModel):
    overall_sentiment: Sentiment
    complaints: list[Complaint]
    praised_features: list[str]
    feature_requests: list[str]
    summary: str


SENTIMENT_INSTRUCTIONS = f"""You analyze customer reviews of the AccessIQ K1 keyless keypad smart doorknob lock.

Return:
- overall_sentiment: positive (satisfied), negative (dissatisfied), mixed (clear praise AND a clear
  complaint about the product), or neutral (lukewarm, no real complaint or praise).
- complaints: one entry per DISTINCT problem. Group different wordings of the same problem
  ("keypad stops responding", "sometimes won't unlock") into one complaint. Categories:
  installation_difficulty, lock_reliability (fails to lock/unlock, unresponsive, jams, freezes),
  battery_life, keypad_usability (readability, button feel, code setup confusion),
  build_quality, app_connectivity (app, Bluetooth, Wi-Fi bridge, firmware, notifications),
  security_concerns (door unlocked unexpectedly, deleted/unknown codes work, forced open,
  any risk of unauthorized entry), shipping_damage_or_wrong_item, shipping_delay_or_order_issue, other.
  Shipping, packaging, price or seller problems are NOT product defects.
- department for each complaint: {json.dumps(ISSUE_TO_DEPARTMENT)}
- severity: high if it can lock someone out or create a security/safety risk.
- evidence_quote: copy the exact words from the review. Never paraphrase or invent a quote.
- feature_requests: things the customer wishes the product had.
"""

sentiment_analyst = Agent(
    name="Sentiment Analyst",
    instructions=SENTIMENT_INSTRUCTIONS,
    model=MODEL,
    output_type=SentimentResult,
)

reply_writer = Agent(
    name="Reply Writer",
    instructions="""You write public replies from AccessIQ to negative customer reviews.
- Open by thanking the customer and sincerely acknowledging the specific problem they described.
- Express sympathy without blaming the customer and without admitting legal liability.
  For security or safety complaints, say the report has been escalated to our product safety team
  and ask them to contact support; do not speculate about causes.
- If the offer is return_or_exchange, offer a return or exchange. If it is exchange, offer an
  exchange/replacement. If it is none, do NOT mention returns, refunds, exchanges or replacements;
  offer help from support instead (and troubleshooting/firmware help for app issues).
- Invite them to contact support@accessiq.example with their order number.
- 60-120 words, warm and professional, no emojis. Sign off: "- The AccessIQ Care Team".
Return only the reply text.""",
    model=MODEL,
)


async def _sentiment_json(result) -> str:
    out = result.final_output
    return out.model_dump_json() if isinstance(out, BaseModel) else str(out)


class ReviewReport(BaseModel):
    review_id: str
    overall_sentiment: Sentiment
    issues: list[IssueCategory]
    departments_routed: list[Department]
    reply_posted: bool
    offer: Offer
    notes: str


MANAGER_INSTRUCTIONS = """You are the Review Operations Manager for AccessIQ smart locks.
You decide which tools to use and in what order to handle the user's request. For each review:

1. Call lookup_review to get the review. Never guess its content or rating.
2. Call analyze_sentiment with the review's title, rating and full text.
3. If the rating is 4 or 5: do NOT route it and do NOT reply. Just report the sentiment.
4. If the rating is 3 or less and the analysis found at least one complaint:
   a. Call route_to_department once for EACH distinct department in the complaints
      (legal, shipping, app_digital, manufacturing). Use the complaint's issue_category,
      a priority (urgent for security/safety, high for lockouts, otherwise normal or low)
      and a one-sentence summary. A person must approve each routing.
   b. Call check_return_policy with the main complaint's issue_category
      (security_concerns outranks everything; otherwise the most severe complaint).
   c. Call draft_customer_reply with the review text, the main issue and the policy offer.
   d. Call post_customer_reply with the draft and EXACTLY the offer returned by check_return_policy.
      A person must approve the reply.
5. If the rating is 3 or less but there is no complaint (lukewarm/neutral), do not route or reply.
6. If a person rejects a tool call with feedback, revise using the feedback and propose it again once.
   If a call is rejected without feedback or blocked by a guardrail you cannot satisfy, do not retry it.
7. Don't call any tool more often than needed. Never claim an action happened unless its tool
   returned a success status.
Finish with the report for the (last) review processed. departments_routed and reply_posted
must reflect only actions that actually succeeded."""

manager = Agent(
    name="Review Operations Manager",
    instructions=MANAGER_INSTRUCTIONS,
    model=MODEL,
    tools=[
        lookup_review,
        sentiment_analyst.as_tool(
            tool_name="analyze_sentiment",
            tool_description="Analyze a review's sentiment and extract each complaint with its category, "
                             "responsible department, severity and an exact supporting quote.",
            custom_output_extractor=_sentiment_json,
        ),
        route_to_department,
        check_return_policy,
        reply_writer.as_tool(
            tool_name="draft_customer_reply",
            tool_description="Draft a sympathetic public reply to a negative review. Input must include the "
                             "review text, the main issue and the allowed offer (return_or_exchange, exchange or none).",
        ),
        post_customer_reply,
    ],
    output_type=ReviewReport,
)

AGENTS = [manager, sentiment_analyst, reply_writer]

# ---------------------------------------------------------------------------
# Human-in-the-loop runner (slides 26-27): pause -> approve/reject -> resume
# ---------------------------------------------------------------------------
Approver = Callable[[str, dict], Awaitable[tuple[str, str | None]]]


async def terminal_approver(tool_name: str, args: dict) -> tuple[str, str | None]:
    """Ask the person at the keyboard. Returns ('approve', None) or ('reject', feedback)."""
    print("\n" + "=" * 70)
    if tool_name == "route_to_department":
        print(f"APPROVAL NEEDED - route review {args.get('review_id')} to "
              f"{args.get('department', '').upper()}  [{args.get('priority')}]")
        print(f"  issue:   {args.get('issue_category')}\n  summary: {args.get('summary')}")
    elif tool_name == "post_customer_reply":
        print(f"APPROVAL NEEDED - post public reply to {args.get('review_id')} (offer: {args.get('offer')})")
        print("-" * 70 + f"\n{args.get('reply_text')}\n" + "-" * 70)
    else:
        print(f"APPROVAL NEEDED - {tool_name}({json.dumps(args)})")
    while True:
        ans = (await asyncio.to_thread(input, "Approve? [y]es / [n]o / [f]eedback: ")).strip().lower()
        if ans in ("y", "yes"):
            return "approve", None
        if ans in ("n", "no"):
            return "reject", None
        if ans in ("f", "feedback"):
            fb = await asyncio.to_thread(input, "Feedback for the agent: ")
            return "reject", fb.strip() or None


@dataclass
class RunRecord:
    final_output: ReviewReport | str | None
    trace: list[dict]            # ordered tool calls proposed by the manager
    approvals: list[dict]        # human decisions
    context: AppContext


class ApprovalSession:
    """One manager run that can pause for human approval and be resumed later.

    Used by the terminal flow, the evaluation and the web UI:
        s = ApprovalSession("Process review AIQ-0042.", ctx)
        await s.start()                       # runs until done or a tool needs approval
        s.pending()                           # [{'index', 'tool', 'args'}, ...]
        await s.resume({0: ("approve", None)})  # or ("reject", "feedback text")
    status: queued | running | awaiting_approval | done | error
    """

    def __init__(self, user_input: str, ctx: AppContext, label: str | None = None, max_turns: int = 25):
        self.user_input, self.ctx, self.label, self.max_turns = user_input, ctx, label or user_input, max_turns
        self.trace: list[dict] = []      # tool calls in order, with their outputs once available
        self.approvals: list[dict] = []
        self.status = "queued"
        self.error: str | None = None
        self.final_output = None
        self.created_at = datetime.now()
        self._result = None
        self._by_call: dict[str, dict] = {}

    def _collect(self, result) -> None:
        for item in result.new_items:
            raw = item.raw_item
            if isinstance(item, ToolCallItem):
                cid = getattr(raw, "call_id", None) or getattr(raw, "id", None)
                if cid in self._by_call:
                    continue
                try:
                    args = json.loads(getattr(raw, "arguments", "") or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": getattr(raw, "arguments", "")}
                entry = {"tool": getattr(raw, "name", "?"), "args": args, "call_id": cid, "output": None}
                self._by_call[cid] = entry
                self.trace.append(entry)
            elif isinstance(item, ToolCallOutputItem):
                cid = raw.get("call_id") if isinstance(raw, dict) else getattr(raw, "call_id", None)
                if cid in self._by_call:
                    self._by_call[cid]["output"] = str(item.output)

    def _after(self, result) -> None:
        self._result = result
        self._collect(result)
        if result.interruptions:
            self.status = "awaiting_approval"
        else:
            self.status, self.final_output = "done", result.final_output

    async def _run(self, run_input) -> None:
        self.status = "running"
        try:
            self._after(await Runner.run(manager, run_input, context=self.ctx, max_turns=self.max_turns))
        except Exception as e:
            self.status, self.error = "error", f"{type(e).__name__}: {e}"

    async def start(self) -> None:
        await self._run(self.user_input)

    def pending(self) -> list[dict]:
        return self._interruptions() if self.status == "awaiting_approval" else []

    def _interruptions(self) -> list[dict]:
        if self._result is None:
            return []
        out = []
        for i, req in enumerate(self._result.interruptions):
            raw = req.raw_item
            out.append({"index": i, "tool": getattr(req, "tool_name", None) or getattr(raw, "name", "?"),
                        "args": json.loads(getattr(raw, "arguments", "") or "{}")})
        return out

    async def resume(self, decisions: dict[int, tuple[str, str | None]]) -> None:
        state = self._result.to_state()
        for p, req in zip(self._interruptions(), self._result.interruptions):
            decision, feedback = decisions.get(p["index"], ("reject", None))
            self.approvals.append({**p, "decision": decision, "feedback": feedback,
                                   "at": datetime.now().isoformat(timespec="seconds")})
            if decision == "approve":
                state.approve(req)
            else:
                state.reject(req, rejection_message=feedback or "Rejected by the human reviewer. Do not retry this call.")
        await self._run(state)

    def record(self) -> RunRecord:
        return RunRecord(self.final_output, self.trace, self.approvals, self.ctx)


async def run_with_approvals(user_input: str, ctx: AppContext, approver: Approver,
                             max_rounds: int = 10, max_turns: int = 25) -> RunRecord:
    s = ApprovalSession(user_input, ctx, max_turns=max_turns)
    await s.start()
    rounds = 0
    while s.status == "awaiting_approval" and rounds < max_rounds:
        rounds += 1
        decisions = {}
        for p in s.pending():
            decisions[p["index"]] = await approver(p["tool"], p["args"])
        await s.resume(decisions)
    if s.status == "error":
        raise RuntimeError(s.error)
    return s.record()


def configure(api_key: str | None = None, model: str | None = None) -> None:
    """Set the OpenAI key and/or the model used by all three agents (used by the web UI)."""
    global MODEL
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        set_default_openai_key(api_key)
    if model:
        MODEL = model
        for agent in AGENTS:
            agent.model = model


async def process_review(review_id: str, ctx: AppContext, approver: Approver = terminal_approver) -> RunRecord:
    return await run_with_approvals(f"Process review {review_id}.", ctx, approver)


def save_outbox(ctx: AppContext, folder: str = "outbox") -> None:
    os.makedirs(folder, exist_ok=True)
    for name, rows in (("tickets", ctx.tickets), ("replies", ctx.replies)):
        if rows:
            with open(os.path.join(folder, f"{name}.jsonl"), "a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")


def print_report(rec: RunRecord) -> None:
    out = rec.final_output
    print("\n" + "#" * 70)
    if isinstance(out, ReviewReport):
        print(f"Review {out.review_id}: sentiment={out.overall_sentiment} issues={out.issues}")
        print(f"  routed to: {out.departments_routed or 'none'} | reply posted: {out.reply_posted} | offer: {out.offer}")
        print(f"  notes: {out.notes}")
    else:
        print(out)
    print("  tool sequence: " + " -> ".join(t["tool"] for t in rec.trace))


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--review", help="Review ID to process, e.g. AIQ-0042")
    ap.add_argument("--batch", type=int, help="Process the first N reviews (filtered by --max-rating)")
    ap.add_argument("--max-rating", type=int, default=5)
    ap.add_argument("--instruction", help="Free-text instruction for the manager agent")
    ap.add_argument("--data", default=DATA_PATH)
    a = ap.parse_args()

    ctx = AppContext(reviews=load_reviews(a.data))
    if a.instruction:
        rec = await run_with_approvals(a.instruction, ctx, terminal_approver)
        print_report(rec)
    elif a.review:
        print_report(await process_review(a.review, ctx))
    elif a.batch:
        ids = [rid for rid, r in ctx.reviews.items() if int(r["Review Rating"]) <= a.max_rating][: a.batch]
        for rid in ids:
            print_report(await process_review(rid, ctx))
    else:
        ap.print_help()
        return
    save_outbox(ctx)
    print(f"\nExecuted: {len(ctx.tickets)} ticket(s), {len(ctx.replies)} repl(ies). Saved to ./outbox/")


if __name__ == "__main__":
    asyncio.run(main())
