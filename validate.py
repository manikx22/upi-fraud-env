"""
validate.py — OpenEnv self-validation for UPI Fraud Prevention Environment.

Runs all checks that openenv validate would perform:
  1. openenv.yaml presence and required fields
  2. reset() returns valid TurnObservation
  3. step() returns (obs, reward, done, info) with correct types
  4. state() returns dict with required keys
  5. All 3 graders return float in [0.0, 1.0]
  6. Reward is always in [-1.0, 1.0]
  7. Structured signal fields present in observation
  8. No dangerous action ever returns positive reward
  9. Episode terminates correctly
  10. Scenario dataset integrity

Usage:
    python validate.py
    python validate.py --verbose
"""

import sys
import traceback
import yaml
from pathlib import Path

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
verbose = "--verbose" in sys.argv

results = []

def check(name, fn):
    try:
        msg = fn()
        results.append((True, name, msg or ""))
        print(f"  {PASS}  {name}" + (f" — {msg}" if msg and verbose else ""))
    except Exception as e:
        results.append((False, name, str(e)))
        print(f"  {FAIL}  {name}")
        print(f"         {e}")
        if verbose:
            traceback.print_exc()

print("\nUPI Fraud Prevention OpenEnv — Validation\n" + "─"*45)

# ── 1. openenv.yaml ────────────────────────────────────────────────────────────
print("\n[1] openenv.yaml")

def check_yaml_exists():
    assert Path("openenv.yaml").exists(), "openenv.yaml not found"
    return "found"

def check_yaml_fields():
    with open("openenv.yaml") as f:
        data = yaml.safe_load(f)
    required = ["name", "version", "description", "tasks", "action_space", "observation_space", "reward"]
    missing = [k for k in required if k not in data]
    assert not missing, f"Missing fields: {missing}"
    assert len(data["tasks"]) >= 3, "Must have at least 3 tasks"
    return f"{len(data['tasks'])} tasks defined"

def check_yaml_tasks():
    with open("openenv.yaml") as f:
        data = yaml.safe_load(f)
    for t in data["tasks"]:
        for field in ["id", "difficulty", "success_threshold"]:
            assert field in t, f"Task missing field: {field}"
    diffs = {t["difficulty"] for t in data["tasks"]}
    assert "easy" in diffs and "hard" in diffs, "Must have easy and hard tasks"
    return "all task fields valid"

check("openenv.yaml exists", check_yaml_exists)
check("required fields present", check_yaml_fields)
check("task definitions valid", check_yaml_tasks)

# ── 2. Core API ────────────────────────────────────────────────────────────────
print("\n[2] Core API: reset() / step() / state()")

from environment import UPIFraudEnv, Action, TurnObservation, Reward
from scenarios import ActionType

def check_reset():
    env = UPIFraudEnv("easy", 42)
    obs = env.reset()
    assert isinstance(obs, TurnObservation), "reset() must return TurnObservation"
    assert obs.scenario_id.startswith("SC"), "scenario_id should start with SC"
    assert obs.amount_at_risk > 0, "amount_at_risk must be positive"
    assert len(obs.available_actions) == 18, "must expose all 18 actions"
    assert obs.episode_step == 0, "episode_step must be 0 after reset"
    return f"inbox has {obs.total_turns} turns in first scenario"

def check_step_returns():
    env = UPIFraudEnv("easy", 42)
    env.reset()
    action = Action(action_type=ActionType.HANG_UP)
    result = env.step(action)
    assert len(result) == 4, "step() must return 4-tuple"
    obs, reward, done, info = result
    assert isinstance(obs, TurnObservation)
    assert isinstance(reward, Reward)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    return "4-tuple returned correctly"

def check_state():
    env = UPIFraudEnv("easy", 42)
    env.reset()
    s = env.state()
    required_keys = ["task_id", "seed", "episode_step", "done", "cumulative_reward",
                     "dangerous_actions", "optimal_actions", "money_lost_inr"]
    missing = [k for k in required_keys if k not in s]
    assert not missing, f"state() missing keys: {missing}"
    return f"all {len(required_keys)} required keys present"

def check_done_flag():
    env = UPIFraudEnv("easy", 42)
    env.reset()
    done = False
    steps = 0
    while not done and steps < 60:
        _, _, done, _ = env.step(Action(action_type=ActionType.HANG_UP))
        steps += 1
    assert done, "Episode must eventually terminate"
    assert env._done, "env._done must be True after episode ends"
    try:
        env.step(Action(action_type=ActionType.HANG_UP))
        assert False, "Should raise RuntimeError after done"
    except RuntimeError:
        pass
    return f"terminated after {steps} steps"

check("reset() returns TurnObservation", check_reset)
check("step() returns (obs, reward, done, info)", check_step_returns)
check("state() has required keys", check_state)
check("episode terminates and raises on post-done step", check_done_flag)

# ── 3. Reward integrity ────────────────────────────────────────────────────────
print("\n[3] Reward integrity")

def check_reward_range():
    env = UPIFraudEnv("easy", 42)
    env.reset()
    all_actions = list(ActionType)
    violations = []
    for _ in range(20):
        for at in all_actions[:5]:
            env2 = UPIFraudEnv("easy", 42)
            env2.reset()
            _, reward, _, _ = env2.step(Action(action_type=at))
            if not (-1.0 <= reward.value <= 1.0):
                violations.append(f"{at.value}: {reward.value}")
    assert not violations, f"Reward out of range: {violations}"
    return "all rewards in [-1.0, 1.0]"

def check_dangerous_negative():
    from scenarios import SCENARIOS
    env = UPIFraudEnv("easy", 42)
    obs = env.reset()
    dangerous = [ActionType.SHARE_OTP, ActionType.SHARE_PIN, ActionType.PAY_SCAMMER, ActionType.SCAN_QR]
    violations = []
    for at in dangerous:
        env2 = UPIFraudEnv("easy", 42)
        env2.reset()
        _, reward, _, _ = env2.step(Action(action_type=at))
        if reward.value > 0:
            violations.append(f"{at.value}: {reward.value}")
    assert not violations, f"Dangerous actions must not give positive reward: {violations}"
    return "all dangerous actions return <= 0"

def check_optimal_positive():
    from scenarios import SCENARIOS
    sc = SCENARIOS[0]
    turn = sc.turns[0]
    env = UPIFraudEnv("easy", 42)
    env.reset()
    _, reward, _, _ = env.step(Action(action_type=turn.optimal_action))
    assert reward.value > 0, f"Optimal action must give positive reward, got {reward.value}"
    assert reward.outcome == "optimal"
    return f"optimal action reward: +{reward.value}"

def check_reward_breakdown():
    env = UPIFraudEnv("easy", 42)
    env.reset()
    _, reward, _, _ = env.step(Action(action_type=ActionType.HANG_UP))
    assert isinstance(reward.breakdown, dict), "reward.breakdown must be dict"
    assert len(reward.breakdown) > 0, "breakdown must have at least one entry"
    assert reward.outcome in ["optimal", "acceptable", "dangerous", "neutral"]
    return f"breakdown has {len(reward.breakdown)} component(s)"

check("reward always in [-1.0, 1.0]", check_reward_range)
check("dangerous actions always <= 0", check_dangerous_negative)
check("optimal action gives positive reward", check_optimal_positive)
check("reward has breakdown dict and outcome", check_reward_breakdown)

# ── 4. Observation structured signals ─────────────────────────────────────────
print("\n[4] Structured observation signals")

def check_structured_fields():
    env = UPIFraudEnv("easy", 42)
    obs = env.reset()
    required = ["vpa_verified", "is_official_channel", "pressure_tactic",
                "call_spoofed", "amount_anomaly", "re_victimisation_attempt"]
    missing = [f for f in required if not hasattr(obs, f)]
    assert not missing, f"TurnObservation missing structured fields: {missing}"
    return f"all {len(required)} structured fields present"

def check_signals_populated():
    from scenarios import SCENARIOS
    populated = 0
    for sc in SCENARIOS:
        for t in sc.turns:
            if t.pressure_tactic is not None or t.vpa_verified is not None:
                populated += 1
    assert populated >= 15, f"Too few turns have structured signals: {populated}"
    return f"{populated} turns have at least one structured signal"

def check_revictimisation_flag():
    from scenarios import SCENARIOS
    revict_turns = [
        (sc.id, t.turn_id)
        for sc in SCENARIOS
        for t in sc.turns
        if t.re_victimisation_attempt
    ]
    assert len(revict_turns) >= 2, "Must have at least 2 re-victimisation turns"
    return f"{len(revict_turns)} re-victimisation turn(s) flagged"

check("all 6 structured fields in TurnObservation", check_structured_fields)
check("structured signals populated in scenarios", check_signals_populated)
check("re_victimisation_attempt flagged correctly", check_revictimisation_flag)

# ── 5. Graders ─────────────────────────────────────────────────────────────────
print("\n[5] Graders [0.0–1.0] and determinism")

from tasks import grade_task, TASKS

def check_grader_range():
    violations = []
    for task in TASKS:
        actions = [Action(action_type=ActionType.HANG_UP)] * 20
        score = grade_task(task.id, actions)["score"]
        if not (0.0 <= score <= 1.0):
            violations.append(f"{task.id}: {score}")
    assert not violations, f"Grader scores out of [0,1]: {violations}"
    return "all graders return [0.0, 1.0]"

def check_grader_determinism():
    actions = [Action(action_type=ActionType.HANG_UP)] * 15
    for task in TASKS:
        s1 = grade_task(task.id, actions)["score"]
        s2 = grade_task(task.id, actions)["score"]
        assert s1 == s2, f"Grader {task.id} is not deterministic: {s1} != {s2}"
    return "all graders deterministic"

def check_grader_not_constant():
    good = [Action(action_type=ActionType.DECLINE_REQUEST)] * 15
    bad  = [Action(action_type=ActionType.PAY_SCAMMER)] * 15
    for task in TASKS:
        sg = grade_task(task.id, good)["score"]
        sb = grade_task(task.id, bad)["score"]
        assert sg != sb, f"Grader {task.id} returns same score for good and bad actions"
    return "graders discriminate good vs bad actions"

def check_grader_pass_threshold():
    for task in TASKS:
        result = grade_task(task.id, [])
        assert "passed" in result
        assert "score" in result
        assert "success_threshold" in result
    return "all graders return passed/score/threshold"

check("grader scores in [0.0, 1.0]", check_grader_range)
check("graders are deterministic", check_grader_determinism)
check("graders not constant (discriminate good/bad)", check_grader_not_constant)
check("graders return passed/score/threshold", check_grader_pass_threshold)

# ── 6. Scenario dataset ────────────────────────────────────────────────────────
print("\n[6] Scenario dataset integrity")

def check_scenario_count():
    from scenarios import SCENARIOS
    assert len(SCENARIOS) >= 15, f"Need at least 15 scenarios, have {len(SCENARIOS)}"
    return f"{len(SCENARIOS)} scenarios"

def check_difficulty_spread():
    from scenarios import SCENARIOS_BY_DIFFICULTY
    for diff in ["easy", "medium", "hard"]:
        assert len(SCENARIOS_BY_DIFFICULTY[diff]) >= 2, f"Need >=2 {diff} scenarios"
    return f"easy:{len(SCENARIOS_BY_DIFFICULTY['easy'])} medium:{len(SCENARIOS_BY_DIFFICULTY['medium'])} hard:{len(SCENARIOS_BY_DIFFICULTY['hard'])}"

def check_turn_integrity():
    from scenarios import SCENARIOS
    for sc in SCENARIOS:
        assert len(sc.turns) >= 2, f"{sc.id} has only {len(sc.turns)} turn(s)"
        for t in sc.turns:
            assert -1.0 <= t.reward_if_dangerous <= 0.0, f"{sc.id} T{t.turn_id}: dangerous reward must be <= 0"
            assert 0.0 < t.reward_if_optimal <= 1.0, f"{sc.id} T{t.turn_id}: optimal reward must be > 0"
            assert t.optimal_action not in t.dangerous_actions, f"{sc.id} T{t.turn_id}: optimal cannot be dangerous"
    return "all turn rewards valid"

def check_unique_ids():
    from scenarios import SCENARIOS
    ids = [s.id for s in SCENARIOS]
    assert len(ids) == len(set(ids)), f"Duplicate scenario IDs: {[x for x in ids if ids.count(x)>1]}"
    return "all scenario IDs unique"

def check_scam_type_coverage():
    from scenarios import SCENARIOS, ScamType
    covered = {s.scam_type for s in SCENARIOS}
    assert len(covered) >= 6, f"Only {len(covered)} scam types covered"
    return f"{len(covered)} distinct scam types covered"

check("at least 15 scenarios", check_scenario_count)
check("all difficulty levels represented", check_difficulty_spread)
check("all turns have valid reward ranges", check_turn_integrity)
check("scenario IDs are unique", check_unique_ids)
check("at least 6 scam types covered", check_scam_type_coverage)

# ── 7. All 3 tasks run end-to-end ──────────────────────────────────────────────
print("\n[7] End-to-end episode runs")

def check_easy_episode():
    env = UPIFraudEnv("easy", 42)
    obs = env.reset()
    done = False
    steps = 0
    while not done and steps < 50:
        _, _, done, _ = env.step(Action(action_type=ActionType.DECLINE_REQUEST))
        steps += 1
    assert done
    return f"completed in {steps} steps"

def check_medium_episode():
    env = UPIFraudEnv("medium", 123)
    obs = env.reset()
    done = False
    steps = 0
    while not done and steps < 50:
        _, _, done, _ = env.step(Action(action_type=ActionType.TAKE_SCREENSHOT))
        steps += 1
    assert done
    return f"completed in {steps} steps"

def check_hard_episode():
    env = UPIFraudEnv("hard", 999)
    obs = env.reset()
    done = False
    steps = 0
    while not done and steps < 60:
        _, _, done, _ = env.step(Action(action_type=ActionType.HANG_UP))
        steps += 1
    assert done
    return f"completed in {steps} steps"

check("easy episode terminates", check_easy_episode)
check("medium episode terminates", check_medium_episode)
check("hard episode terminates", check_hard_episode)

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "─"*45)
passed = sum(1 for r in results if r[0])
failed = sum(1 for r in results if not r[0])
total  = len(results)

print(f"Results: {passed}/{total} passed", end="")
if failed:
    print(f"  |  {failed} FAILED")
    print("\nFailed checks:")
    for ok, name, msg in results:
        if not ok:
            print(f"  - {name}: {msg}")
else:
    print("  — all checks passed")

print()
sys.exit(0 if failed == 0 else 1)
