# AccessIQ Review Agents (ISM6422, Team SentAIment)

A multi-agent app built with the **OpenAI Agents SDK**. It runs sentiment analysis on smart-lock reviews, routes reviews rated 3 stars or less to the responsible department, and drafts sympathetic replies to negative reviews, offering a return or exchange when policy allows. A person approves every routing and every reply.

## Setup

```bash
pip install -r requirements.txt
```

Keep every file from this folder together, including the hidden `.streamlit/` folder (it holds the UI theme).

## Run the web app (recommended)

```bash
streamlit run app.py
```

Your browser opens at http://localhost:8501. A dialog asks you to paste your OpenAI API key and choose a model. The key stays in that browser session's memory and is never written to disk. The app has five pages:

| Page | What happens there |
|---|---|
| **Review queue** | Filter and search the 500 reviews, select some, and start the agents. Or use Quick start, or give the manager a free-text instruction. |
| **Approvals** | Every routing and every reply waits here. Each card shows the customer review beside the proposed action. Approve, reject, or send it back with feedback. |
| **Agent runs** | The trajectory: each agent's tool calls in order, whether each was approved, rejected or blocked, plus the final report. |
| **Outcomes** | Tickets by department, sentiment mix, posted replies, guardrail blocks, and CSV downloads. |
| **Evaluation** | Run the three-scope evaluation with live progress, then see metric averages, an interpretation, per-case results and downloads. |

Agent runs continue in the background while you move between pages, and the screen updates as approvals arrive. Up to 3 runs execute at once.

## Run from the terminal (alternative)

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4.1-mini     # optional, this is the default
python review_agents.py --review AIQ-0018                 # one review, approve in the terminal
python review_agents.py --batch 5 --max-rating 3          # the first 5 reviews rated 3 or less
python review_agents.py --instruction "Process reviews AIQ-0007 and AIQ-0019"
python evaluate.py                                        # 40 held-out test cases, three-scope metrics
```

At each approval prompt, type `y` to approve, `n` to reject, or `f` to reject with feedback. With feedback, the agent revises its proposal and asks again. Executed tickets and replies are saved to `outbox/`.

## How it meets the requirements

| Requirement | Where |
|---|---|
| At least 2 agents, centralized or decentralized | 3 agents in a **centralized** design: the **Review Operations Manager** calls the **Sentiment Analyst** and the **Reply Writer** as tools (`agent.as_tool`, slide 22). |
| At least 2 tools, at least 1 with human-in-the-loop | 6 tools. `route_to_department` and `post_customer_reply` use `needs_approval=True`. The run pauses, a person approves or rejects with feedback, and the run resumes from `state` (slides 26-27). |
| Agents decide which tool to use, in what order, then consolidate the results | The manager picks the tool sequence for each review. It skips routing and replies for 4-5 star reviews, routes once per department, checks policy before replying, and ends with a structured `ReviewReport`. |
| OpenAI Agents SDK | `Agent`, `Runner`, `function_tool`, `as_tool`, `RunState.approve/reject`, `tool_input_guardrail` |
| Evaluation at three scopes, at least 1 metric each, at least 30 cases | `evaluate.py`: 11 metrics over 40 held-out test reviews (see below). |

**Tools:** `lookup_review`, `analyze_sentiment` (agent), `route_to_department` (approval), `check_return_policy`, `draft_customer_reply` (agent), `post_customer_reply` (approval + guardrail).

**Departments:** legal (security and safety), shipping (delivery, packaging, order issues), app_digital (app, Bluetooth, Wi-Fi, firmware), manufacturing (lock mechanism, batteries, keypad, build quality, installation hardware).

**Return / exchange policy** (`return_policy()` in `review_agents.py`):

| Situation | Offer |
|---|---|
| Hardware defect or security issue on a verified purchase | Return or exchange |
| Damaged in transit or wrong item shipped | Exchange |
| App issue | Tech support only |
| Late delivery, pricing, or unverified purchase | No offer |

A guardrail blocks any reply that offers more than the policy allows, or that replies to a 4-5 star review, even after a person approved it (slide 33).

## Evaluation metrics

| Scope | Metric | Scored by |
|---|---|---|
| End-to-end | E1 sentiment accuracy (+ macro F1), E2 outcome correct (right departments, reply if and only if needed, report matches what actually happened), E3 reply quality | code, code, LLM judge |
| Trajectory | T1 valid tool order, T2 step efficiency, T3 task completion after approval | code |
| Component | C1 department arguments (F1), C2 policy `issue_category` argument, C3 `offer` argument, C4 Sentiment Analyst alone, C5 evidence quotes found verbatim in the review | code |

The simulated reviewer approves everything, except that it rejects the first reply on every 4th actionable case with feedback. This tests the reject, revise and re-propose path. `--repeats 3` also reports pass^k (slide 37).

The script writes `eval_results.csv` (one row per case, including the tool sequence) and `eval_summary.md` (the averages table plus an auto-generated interpretation to edit).
