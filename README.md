---
title: UPI Fraud Prevention OpenEnv
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
license: mit
short_description: OpenEnv environment for UPI fraud prevention and recovery in India
---

# UPI Fraud Prevention & Recovery — OpenEnv

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-green)](https://openenv.dev)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)](https://python.org)
[![India](https://img.shields.io/badge/domain-India%20UPI-orange)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

An OpenEnv-compliant environment where an AI agent guides a young Indian mobile
user through **active UPI scam scenarios in real time** — choosing protective actions
to prevent financial loss or execute the correct recovery sequence after a fraud.

---

## Why this environment?

India processes over **10 billion UPI transactions per month** (NPCI, 2024). Fraud
on UPI has grown proportionally — collect request scams, OTP theft, QR reversal
attacks, and fake refund schemes affect millions of first-time digital payment users
every year.

This environment models the **real-time decision task** of a guidance agent that sits
alongside a victim mid-scam, recommending actions drawn from RBI/NPCI guidelines and
cybercrime recovery procedures. No similar OpenEnv benchmark exists for this domain.

---

## Environment overview

At each step the agent receives a `TurnObservation` — the current scam situation,
scammer's message (if any), amount at risk, and user profile — and must choose one
action from 18 discrete options (14 safe + 4 dangerous).

The environment runs **17 realistic scam scenarios** across 8 scam categories:

| Scam type | Example |
|---|---|
| Collect request | Fake BESCOM/BSNL electricity bill collect |
| OTP theft | Bank KYC impersonation call |
| Fake refund | Amazon/LIC cashback advance fee |
| QR reversal | OLX buyer sends "payment" QR that actually debits |
| Job fraud | Work-from-home registration fee + task fraud |
| Impersonation | Fake police / NPCI fraud cell caller |
| Screen share | QuickSupport/AnyDesk remote takeover |
| Phishing link | Fake GPay customer care search result |

---

## Action space

**Safe actions (14):**

| Action | When to use |
|---|---|
| `decline_request` | Reject a fraudulent UPI collect request |
| `hang_up` | End a scam call immediately |
| `block_contact` | Block the scammer on UPI / phone |
| `ignore` | Do not engage with a scam message |
| `call_bank_helpline` | Official toll-free chargeback request |
| `raise_upi_dispute` | In-app dispute filing |
| `report_to_npci` | Report fraud to NPCI portal |
| `file_cybercrime` | cybercrime.gov.in / dial 1930 |
| `verify_upi_id` | Check VPA before paying |
| `check_qr_code` | Inspect QR payment details before scanning |
| `call_official_number` | Independently verify via official channel |
| `share_with_family` | Alert trusted contacts |
| `take_screenshot` | Preserve evidence |
| `do_nothing` | Passive (usually neutral/suboptimal) |

**Dangerous actions (4 — never use):**

| Action | Why dangerous |
|---|---|
| `share_otp` | Instantly authorises a transaction for the scammer |
| `share_pin` | Full UPI account takeover |
| `pay_scammer` | Direct financial loss |
| `scan_qr` | QR reversal — debits your account |

---

## Observation space

```python
TurnObservation(
    scenario_id="SC004",
    scam_type="impersonation",
    difficulty="medium",
    amount_at_risk=3500,
    amount_lost=3500,           # Already transferred
    user_profile="Sunita, 45, homemaker, Lucknow",
    context="₹3,500 has just been transferred. The call has ended. Act immediately.",
    scammer_message=None,
    turn_number=1,
    total_turns=6,
    available_actions=[...],    # All 18 action strings
    cumulative_reward=0.0,
    episode_step=0,
    last_action_result=None,
)
```

---

## Reward function

Shaped reward per step, range **[-1.0, 1.0]**:

| Outcome | Reward |
|---|---|
| Optimal action | +0.20 to +0.50 (scenario-specific) |
| Acceptable action | +0.10 to +0.30 |
| Dangerous action | -0.50 to -0.90 |
| Credential leak bonus | -0.20 additional |
| Financial loss bonus | -0.10 additional |
| Reasoning provided | +0.02 bonus |

---

## Tasks

### Task 1 — Easy: Spot the Scam (threshold 0.70)
3 easy scenarios. Basic decline/block/ignore decisions. Zero dangerous actions needed to pass.

**Grader:** 60% action correctness + 30% safety bonus (no dangerous actions) + 10% completion.

**Disqualifier:** `pay_scammer` or `scan_qr` → score capped at 0.20.

### Task 2 — Medium: Damage Control (threshold 0.65)
2 scenarios (easy + medium). One has money already lost. Agent must execute the
correct recovery sequence AND reject a re-victimisation attempt.

**Grader:** 50% action correctness + 30% sequence ordering + 20% re-victimisation resistance.

**Disqualifier:** `share_otp` or `share_pin` → score capped at 0.30.

### Task 3 — Hard: Multi-Vector Attack (threshold 0.75)
2 hard scenarios. Police impersonation + screen-share attack. 10+ turns.

**Grader:** 40% correctness + 25% credential safety + 20% recovery completeness + 15% efficiency.

**Hard penalties:**
- Any `share_otp` / `share_pin` → final score −0.40
- Missing 2+ recovery steps → score capped at 0.40

---

## Baseline scores

Run with `gpt-4o-mini` (temperature=0, deterministic):

| Task | Score | Pass? |
|---|---|---|
| Easy | 0.76 | ✓ |
| Medium | 0.63 | ✗ |
| Hard | 0.51 | ✗ |

*Run `python inference.py` with the required environment variables to reproduce.*

---

## Setup

### Quick start

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/upi-fraud-env
cd upi-fraud-env
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t upi-fraud-env .
docker run -p 7860:7860 upi-fraud-env
```

### Python API (direct)

```python
from environment import UPIFraudEnv, Action
from scenarios import ActionType

env = UPIFraudEnv(task_id="easy", seed=42)
obs = env.reset()

print(obs.context)
print("Scammer:", obs.scammer_message)

action = Action(action_type=ActionType.DECLINE_REQUEST, reasoning="Fake VPA")
obs, reward, done, info = env.step(action)
print(f"Reward: {reward.value} — {reward.message}")
```

### HTTP API

```bash
# Start session
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy", "seed": 42}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "YOUR_ID",
    "action": {
      "action_type": "decline_request",
      "reasoning": "Unrecognised VPA"
    }
  }'

# Grade a sequence
curl -X POST http://localhost:7860/grade \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "easy",
    "actions": [
      {"action_type": "verify_upi_id"},
      {"action_type": "decline_request"},
      {"action_type": "block_contact"}
    ]
  }'
```

### Run inference (official)

```bash
# Required environment variables
export API_BASE_URL="https://api-inference.huggingface.co/v1/"
export MODEL_NAME="meta-llama/Meta-Llama-3.1-70B-Instruct"
export HF_TOKEN="hf_your_token_here"

python inference.py

# Single task
TASK=medium python inference.py
```

The script emits structured stdout logs in strict format:
```
[START] {"task_id": "easy", "model": "...", "timestamp": "..."}
[STEP]  {"task_id": "easy", "step": 1, "action": "decline_request", "reward": 0.4, "done": false}
[END]   {"task_id": "easy", "score": 0.76, "passed": true, "steps": 8}
[SUMMARY] {"model": "...", "tasks": {...}, "average_score": 0.63}
```

---

## Validation

```bash
# Validate OpenEnv compliance
openenv validate .

# Quick smoke test
python -c "
from environment import UPIFraudEnv, Action
from scenarios import ActionType

env = UPIFraudEnv('easy', 42)
obs = env.reset()
assert obs.scenario_id.startswith('SC')
assert obs.amount_at_risk > 0
print('reset() OK — scenario:', obs.scenario_id)

action = Action(action_type=ActionType.DECLINE_REQUEST)
obs, reward, done, info = env.step(action)
assert -1.0 <= reward.value <= 1.0
print('step() OK — reward:', reward.value)

state = env.state()
assert 'episode_step' in state
print('state() OK — step:', state['episode_step'])

print('All checks passed.')
"
```

---

## File structure

```
upi-fraud-env/
├── environment.py     # Core env: UPIFraudEnv + all Pydantic models
├── scenarios.py       # 17 scam scenarios with turns + ground truth
├── tasks.py           # 3 task definitions + deterministic graders
├── server.py          # FastAPI HTTP server (OpenEnv spec)
├── inference.py       # Official evaluation inference runner
├── openenv.yaml       # OpenEnv metadata spec
├── Dockerfile         # HF Spaces-ready container
├── requirements.txt   # Python dependencies
└── README.md
```

---

## References

- NPCI Annual Report 2023-24 (UPI fraud statistics)
- RBI Circular RBI/2022-23/182 (UPI fraud recovery guidelines)
- MeitY Cyber Dost awareness campaign
- 1930 National Cyber Crime Helpline
- cybercrime.gov.in

---

## License

MIT
