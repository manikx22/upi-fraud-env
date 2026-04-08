"""
inference.py — UPI Fraud Prevention OpenEnv
Official inference script for hackathon submission.

Environment variables (required):
    API_BASE_URL   The API endpoint for the LLM (e.g. https://api-inference.huggingface.co/v1/)
    MODEL_NAME     The model identifier (e.g. meta-llama/Meta-Llama-3.1-70B-Instruct)
    HF_TOKEN       Your Hugging Face / API key

Stdout log format (strict — must not deviate):
    [START] {"task_id": ..., "model": ..., "timestamp": ...}
    [STEP]  {"task_id": ..., "step": ..., "action": ..., "reward": ..., "done": ...}
    [END]   {"task_id": ..., "score": ..., "passed": ..., "steps": ...}

Usage:
    API_BASE_URL=https://api-inference.huggingface.co/v1/ \
    MODEL_NAME=meta-llama/Meta-Llama-3.1-70B-Instruct \
    HF_TOKEN=hf_xxx \
    python inference.py
"""

import os
import sys
import json
import time
from datetime import datetime, timezone
from openai import OpenAI

from environment import UPIFraudEnv, Action, TurnObservation
from tasks import TASKS, grade_task
from scenarios import ActionType

# ─── Environment variable configuration ──────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "").rstrip("/") + "/"
MODEL_NAME   = os.environ.get("MODEL_NAME",   "meta-llama/Meta-Llama-3.1-70B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")

if not API_BASE_URL.strip("/"):
    print("[ERROR] API_BASE_URL environment variable is not set", flush=True)
    sys.exit(1)
if not HF_TOKEN:
    print("[ERROR] HF_TOKEN environment variable is not set", flush=True)
    sys.exit(1)

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# ─── Agent system prompt ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a UPI fraud prevention advisor helping Indian mobile users navigate scam situations in real time.

You will be shown a fraud scenario with structured signals. Choose ONE action from the available list.

STRUCTURED SIGNALS — use these to make your decision:
- vpa_verified: false = the UPI ID is fake/unverified. Never pay.
- is_official_channel: false = caller/sender is NOT from an official channel.
- pressure_tactic: urgency/threat/authority/fomo = scammer is using psychological pressure. Resist.
- call_spoofed: true = phone number is spoofed to look official.
- amount_anomaly: true = payment amount differs from what is expected.
- re_victimisation_attempt: true = this is a SECOND scam targeting the same victim.

GOLDEN RULES — never violate:
1. NEVER share OTP or UPI PIN with anyone — not even bank or police
2. NEVER pay a scammer or scan their QR code to "receive" money
3. NEVER install remote access apps (AnyDesk, QuickSupport, TeamViewer)
4. Always verify via official channels independently

RECOVERY SEQUENCE when money is already lost:
  take_screenshot → call_bank_helpline → raise_upi_dispute → file_cybercrime

AVAILABLE ACTIONS:
  Safe: decline_request, hang_up, block_contact, ignore, call_bank_helpline,
        raise_upi_dispute, report_to_npci, file_cybercrime, verify_upi_id,
        check_qr_code, call_official_number, share_with_family, take_screenshot, do_nothing
  DANGEROUS (never choose): share_otp, share_pin, pay_scammer, scan_qr

Respond ONLY with a JSON object, no markdown, no explanation outside JSON:
{
  "action_type": "<action from available list>",
  "reasoning": "<one sentence why>"
}"""


# ─── Logging helpers (strict format) ─────────────────────────────────────────

def log_start(task_id: str) -> None:
    payload = {
        "task_id":   task_id,
        "model":     MODEL_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    print(f"[START] {json.dumps(payload)}", flush=True)


def log_step(task_id: str, step: int, action: str, reward: float, done: bool) -> None:
    payload = {
        "task_id": task_id,
        "step":    step,
        "action":  action,
        "reward":  round(reward, 4),
        "done":    done,
    }
    print(f"[STEP]  {json.dumps(payload)}", flush=True)


def log_end(task_id: str, score: float, passed: bool, steps: int) -> None:
    payload = {
        "task_id": task_id,
        "score":   round(score, 4),
        "passed":  passed,
        "steps":   steps,
    }
    print(f"[END]   {json.dumps(payload)}", flush=True)


# ─── Observation formatter ────────────────────────────────────────────────────

def format_observation(obs: TurnObservation) -> str:
    lines = [
        f"Scenario: {obs.scenario_id} | Scam type: {obs.scam_type} | Difficulty: {obs.difficulty}",
        f"User profile: {obs.user_profile}",
        f"Amount at risk: ₹{obs.amount_at_risk:,} | Already lost: ₹{obs.amount_lost:,}",
        f"Turn {obs.turn_number} of {obs.total_turns}",
        "",
        "STRUCTURED SIGNALS:",
        f"  vpa_verified: {obs.vpa_verified}",
        f"  is_official_channel: {obs.is_official_channel}",
        f"  pressure_tactic: {obs.pressure_tactic}",
        f"  call_spoofed: {obs.call_spoofed}",
        f"  amount_anomaly: {obs.amount_anomaly}",
        f"  re_victimisation_attempt: {obs.re_victimisation_attempt}",
        "",
        f"SITUATION:\n{obs.context}",
    ]
    if obs.scammer_message:
        lines.append(f'\nSCAMMER SAYS:\n"{obs.scammer_message}"')
    if obs.last_action_result:
        lines.append(f"\nLAST ACTION RESULT: {obs.last_action_result}")
    return "\n".join(lines)


# ─── Agent action getter ──────────────────────────────────────────────────────

def get_agent_action(obs: TurnObservation, task_id: str, step: int) -> Action:
    prompt = f"Current fraud situation:\n\n{format_observation(obs)}\n\nWhat should the user do now?"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=200,
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        data = json.loads(raw)
        action_str = data.get("action_type", "hang_up").strip().lower()

        try:
            action_type = ActionType(action_str)
        except ValueError:
            # Fallback: map common model outputs
            fallback_map = {
                "decline": ActionType.DECLINE_REQUEST,
                "hang":    ActionType.HANG_UP,
                "block":   ActionType.BLOCK_CONTACT,
                "report":  ActionType.REPORT_TO_NPCI,
            }
            action_type = next(
                (v for k, v in fallback_map.items() if k in action_str),
                ActionType.HANG_UP,
            )

        return Action(
            action_type=action_type,
            reasoning=data.get("reasoning", ""),
        )

    except json.JSONDecodeError:
        return Action(action_type=ActionType.HANG_UP, reasoning="JSON parse error — defaulting to hang_up")
    except Exception as e:
        return Action(action_type=ActionType.HANG_UP, reasoning=f"API error: {str(e)[:60]}")


# ─── Task runner ──────────────────────────────────────────────────────────────

def run_task(task_id: str, seed: int) -> dict:
    log_start(task_id)

    env   = UPIFraudEnv(task_id=task_id, seed=seed)
    obs   = env.reset()
    done  = False
    step  = 0
    actions_taken: list[Action] = []

    while not done and step < 50:
        action = get_agent_action(obs, task_id, step)
        actions_taken.append(action)

        obs, reward_obj, done, info = env.step(action)
        step += 1

        log_step(
            task_id=task_id,
            step=step,
            action=action.action_type.value,
            reward=reward_obj.value,
            done=done,
        )

        # Infra restriction: stay well under 20 min total
        time.sleep(0.1)

    result = grade_task(task_id, actions_taken)

    log_end(
        task_id=task_id,
        score=result["score"],
        passed=result["passed"],
        steps=step,
    )

    return result


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> float:
    task_filter = os.environ.get("TASK", None)
    results: dict[str, dict] = {}

    for task in TASKS:
        if task_filter and task.id != task_filter:
            continue
        result = run_task(task.id, task.seed)
        results[task.id] = result

    avg = sum(r["score"] for r in results.values()) / max(1, len(results))

    # Final summary to stdout (parseable)
    summary = {
        "model":        MODEL_NAME,
        "tasks":        {tid: {"score": r["score"], "passed": r["passed"]} for tid, r in results.items()},
        "average_score": round(avg, 4),
    }
    print(f"[SUMMARY] {json.dumps(summary)}", flush=True)

    return avg


if __name__ == "__main__":
    score = main()
    sys.exit(0 if score >= 0.0 else 1)
