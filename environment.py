"""
UPI Fraud Prevention & Recovery — OpenEnv Environment
Full OpenEnv-compliant environment: step() / reset() / state() API
with Pydantic typed models, shaped rewards, and deterministic episodes.
"""

import random
from typing import Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field

from scenarios import (
    Scenario, Turn, ActionType, ScamType,
    SCENARIOS, SCENARIOS_BY_DIFFICULTY, get_scenario,
)


# ─── Pydantic Models ──────────────────────────────────────────────────────────

class Action(BaseModel):
    """Agent action for a single turn."""
    action_type: ActionType
    reasoning: Optional[str] = None   # Optional: agent's explanation


class TurnObservation(BaseModel):
    """What the agent sees at each step."""
    turn_id: int
    scenario_id: str
    scam_type: str
    difficulty: str
    amount_at_risk: int
    amount_lost: int
    user_profile: str
    context: str
    scammer_message: Optional[str]
    available_actions: list[str]
    turn_number: int
    total_turns: int
    episode_step: int
    last_action_result: Optional[str]
    cumulative_reward: float
    # ── Structured signals — parsed boolean flags agents can act on directly ──
    vpa_verified: Optional[bool]           # True=legit, False=fake/unknown VPA
    is_official_channel: Optional[bool]    # Caller/sender using verified official channel?
    pressure_tactic: Optional[str]         # "urgency"|"threat"|"authority"|"fomo"|None
    call_spoofed: Optional[bool]           # Phone number spoofed to look official?
    amount_anomaly: Optional[bool]         # Amount differs from expected/historical?
    re_victimisation_attempt: bool         # Is this a second-wave attack on same victim?


class Reward(BaseModel):
    """Shaped reward with full breakdown."""
    value: float                          # [-1.0, 1.0]
    breakdown: dict[str, float]
    message: str
    optimal_action: str
    agent_action: str
    outcome: str                          # "optimal" | "acceptable" | "dangerous" | "neutral"


class EpisodeSummary(BaseModel):
    """Returned when episode is done."""
    total_reward: float
    normalized_score: float              # [0.0, 1.0]
    turns_completed: int
    dangerous_actions_taken: int
    optimal_actions_taken: int
    acceptable_actions_taken: int
    money_saved: int                     # INR protected from scammer
    money_lost: int                      # INR lost due to wrong actions
    scenarios_completed: list[str]


# ─── Environment ──────────────────────────────────────────────────────────────

class UPIFraudEnv:
    """
    OpenEnv-compliant UPI Fraud Prevention & Recovery Environment.

    An AI agent guides a young Indian mobile user through active UPI scams.
    At each turn the agent sees the scam context and must choose the right
    protective or recovery action from a finite, realistic action set.

    Action space: 14 discrete actions (preventive, recovery, verification, info)
    Observation space: structured TurnObservation Pydantic model
    Reward: shaped, per-step, range [-1.0, 1.0]
    Episode: terminates when all scenario turns are exhausted or max_steps reached
    """

    ALL_ACTIONS = [a.value for a in ActionType]
    SAFE_ACTIONS = [
        ActionType.DECLINE_REQUEST, ActionType.HANG_UP, ActionType.BLOCK_CONTACT,
        ActionType.IGNORE, ActionType.CALL_BANK_HELPLINE, ActionType.RAISE_UPI_DISPUTE,
        ActionType.REPORT_TO_NPCI, ActionType.FILE_CYBERCRIME, ActionType.VERIFY_UPI_ID,
        ActionType.CHECK_QR_CODE, ActionType.CALL_OFFICIAL_NUMBER,
        ActionType.SHARE_WITH_FAMILY, ActionType.TAKE_SCREENSHOT, ActionType.DO_NOTHING,
    ]
    DANGEROUS_ACTIONS = [
        ActionType.SHARE_OTP, ActionType.SHARE_PIN,
        ActionType.PAY_SCAMMER, ActionType.SCAN_QR,
    ]

    def __init__(self, task_id: str = "easy", seed: int = 42):
        self.task_id = task_id
        self.seed = seed
        self._rng = random.Random(seed)
        self._scenarios: list[Scenario] = []
        self._current_scenario_idx: int = 0
        self._current_turn_idx: int = 0
        self._episode_step: int = 0
        self._done: bool = False
        self._history: list[dict] = []
        self._cumulative_reward: float = 0.0
        self._dangerous_count: int = 0
        self._optimal_count: int = 0
        self._acceptable_count: int = 0
        self._money_lost: int = 0
        self._start_time: datetime = datetime.now()

    def _get_task_scenarios(self) -> list[Scenario]:
        """Select scenarios for this task."""
        configs = {
            "easy":   {"difficulties": ["easy"],                   "n": 3},
            "medium": {"difficulties": ["easy", "medium"],         "n": 2},
            "hard":   {"difficulties": ["medium", "hard"],         "n": 2},
        }
        cfg = configs.get(self.task_id, configs["easy"])
        pool = []
        for diff in cfg["difficulties"]:
            pool.extend(SCENARIOS_BY_DIFFICULTY.get(diff, []))
        self._rng.shuffle(pool)
        return pool[: cfg["n"]]

    def reset(self) -> TurnObservation:
        """Reset environment and return initial observation."""
        self._rng = random.Random(self.seed)
        self._scenarios = self._get_task_scenarios()
        self._current_scenario_idx = 0
        self._current_turn_idx = 0
        self._episode_step = 0
        self._done = False
        self._history = []
        self._cumulative_reward = 0.0
        self._dangerous_count = 0
        self._optimal_count = 0
        self._acceptable_count = 0
        self._money_lost = 0
        self._start_time = datetime.now()
        return self._make_observation(last_result=None)

    def step(self, action: Action) -> tuple[TurnObservation, Reward, bool, dict]:
        """
        Execute action and advance the environment.

        Returns:
            observation: Next state
            reward: Shaped reward signal
            done: Whether episode is complete
            info: Metadata dict
        """
        if self._done:
            raise RuntimeError("Episode complete. Call reset() to start a new episode.")

        current_scenario = self._scenarios[self._current_scenario_idx]
        current_turn = current_scenario.turns[self._current_turn_idx]

        # Compute reward
        reward = self._compute_reward(action, current_turn, current_scenario)
        self._cumulative_reward += reward.value
        self._episode_step += 1

        # Track stats
        if reward.outcome == "optimal":
            self._optimal_count += 1
        elif reward.outcome == "acceptable":
            self._acceptable_count += 1
        elif reward.outcome == "dangerous":
            self._dangerous_count += 1
            if action.action_type == ActionType.PAY_SCAMMER:
                self._money_lost += current_scenario.amount_at_risk
            elif action.action_type == ActionType.SCAN_QR:
                self._money_lost += current_scenario.amount_at_risk

        # Record history
        self._history.append({
            "scenario_id": current_scenario.id,
            "turn_id": current_turn.turn_id,
            "action": action.action_type.value,
            "reward": reward.value,
            "outcome": reward.outcome,
            "step": self._episode_step,
        })

        # Advance turn / scenario
        self._current_turn_idx += 1
        if self._current_turn_idx >= len(current_scenario.turns):
            self._current_turn_idx = 0
            self._current_scenario_idx += 1

        # Check done
        done = (
            self._current_scenario_idx >= len(self._scenarios)
            or self._episode_step >= 50
        )
        self._done = done

        if done:
            obs = self._make_done_observation(reward.message)
            info = self._make_info(done=True)
            return obs, reward, True, info

        obs = self._make_observation(last_result=reward.message)
        info = self._make_info(done=False)
        return obs, reward, False, info

    def state(self) -> dict:
        """Return full current environment state."""
        current = None
        if not self._done and self._current_scenario_idx < len(self._scenarios):
            sc = self._scenarios[self._current_scenario_idx]
            t = sc.turns[self._current_turn_idx]
            current = {"scenario_id": sc.id, "turn_id": t.turn_id}

        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "episode_step": self._episode_step,
            "done": self._done,
            "cumulative_reward": round(self._cumulative_reward, 3),
            "current": current,
            "dangerous_actions": self._dangerous_count,
            "optimal_actions": self._optimal_count,
            "acceptable_actions": self._acceptable_count,
            "money_lost_inr": self._money_lost,
            "scenarios_total": len(self._scenarios),
            "scenario_idx": self._current_scenario_idx,
            "history_length": len(self._history),
        }

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _compute_reward(
        self,
        action: Action,
        turn: Turn,
        scenario: Scenario,
    ) -> Reward:
        breakdown: dict[str, float] = {}
        outcome = "neutral"

        if action.action_type == turn.optimal_action:
            breakdown["optimal_action"] = turn.reward_if_optimal
            outcome = "optimal"

        elif action.action_type in turn.acceptable_actions:
            breakdown["acceptable_action"] = turn.reward_if_acceptable
            outcome = "acceptable"

        elif action.action_type in turn.dangerous_actions:
            breakdown["dangerous_action"] = turn.reward_if_dangerous
            outcome = "dangerous"
            # Extra penalty for sharing credentials
            if action.action_type in [ActionType.SHARE_OTP, ActionType.SHARE_PIN]:
                breakdown["credential_leak"] = -0.2
            # Extra penalty for sending money
            if action.action_type in [ActionType.PAY_SCAMMER, ActionType.SCAN_QR]:
                breakdown["financial_loss"] = -0.1

        else:
            # Neutral action (e.g. do_nothing when it's not optimal or dangerous)
            breakdown["neutral_action"] = 0.0
            outcome = "neutral"

        # Bonus: agent provided reasoning
        if action.reasoning and len(action.reasoning) > 20:
            breakdown["reasoning_bonus"] = 0.02

        total = sum(breakdown.values())
        total = round(max(-1.0, min(1.0, total)), 3)

        message = self._reward_message(outcome, action.action_type, turn.optimal_action, total)

        return Reward(
            value=total,
            breakdown=breakdown,
            message=message,
            optimal_action=turn.optimal_action.value,
            agent_action=action.action_type.value,
            outcome=outcome,
        )

    def _reward_message(
        self,
        outcome: str,
        agent_action: ActionType,
        optimal: ActionType,
        total: float,
    ) -> str:
        if outcome == "optimal":
            return f"Correct — {agent_action.value} was the best action here."
        elif outcome == "acceptable":
            return f"Reasonable — {agent_action.value} works, but {optimal.value} would be better."
        elif outcome == "dangerous":
            return f"Dangerous — {agent_action.value} could cause serious harm. Should have: {optimal.value}."
        else:
            return f"Neutral — {agent_action.value} had no effect. Consider: {optimal.value}."

    def _current_turn(self) -> Optional[Turn]:
        if self._current_scenario_idx < len(self._scenarios):
            sc = self._scenarios[self._current_scenario_idx]
            if self._current_turn_idx < len(sc.turns):
                return sc.turns[self._current_turn_idx]
        return None

    def _make_observation(self, last_result: Optional[str]) -> TurnObservation:
        sc = self._scenarios[self._current_scenario_idx]
        turn = sc.turns[self._current_turn_idx]
        return TurnObservation(
            turn_id=turn.turn_id,
            scenario_id=sc.id,
            scam_type=sc.scam_type.value,
            difficulty=sc.difficulty,
            amount_at_risk=sc.amount_at_risk,
            amount_lost=sc.amount_lost,
            user_profile=sc.user_profile,
            context=turn.context,
            scammer_message=turn.scammer_message,
            available_actions=self.ALL_ACTIONS,
            turn_number=self._current_turn_idx + 1,
            total_turns=len(sc.turns),
            episode_step=self._episode_step,
            last_action_result=last_result,
            cumulative_reward=round(self._cumulative_reward, 3),
            vpa_verified=turn.vpa_verified,
            is_official_channel=turn.is_official_channel,
            pressure_tactic=turn.pressure_tactic,
            call_spoofed=turn.call_spoofed,
            amount_anomaly=turn.amount_anomaly,
            re_victimisation_attempt=turn.re_victimisation_attempt,
        )

    def _make_done_observation(self, last_result: Optional[str]) -> TurnObservation:
        """Observation for terminal state — uses last scenario/turn."""
        last_sc = self._scenarios[min(self._current_scenario_idx, len(self._scenarios) - 1)]
        last_turn = last_sc.turns[-1]
        return TurnObservation(
            turn_id=last_turn.turn_id,
            scenario_id=last_sc.id,
            scam_type=last_sc.scam_type.value,
            difficulty=last_sc.difficulty,
            amount_at_risk=last_sc.amount_at_risk,
            amount_lost=last_sc.amount_lost,
            user_profile=last_sc.user_profile,
            context="[Episode complete]",
            scammer_message=None,
            available_actions=self.ALL_ACTIONS,
            turn_number=len(last_sc.turns),
            total_turns=len(last_sc.turns),
            episode_step=self._episode_step,
            last_action_result=last_result,
            cumulative_reward=round(self._cumulative_reward, 3),
            vpa_verified=None,
            is_official_channel=None,
            pressure_tactic=None,
            call_spoofed=None,
            amount_anomaly=None,
            re_victimisation_attempt=False,
        )

    def _make_info(self, done: bool) -> dict:
        info = {
            "episode_step": self._episode_step,
            "cumulative_reward": round(self._cumulative_reward, 3),
            "dangerous_actions": self._dangerous_count,
            "money_lost_inr": self._money_lost,
            "done": done,
        }
        if done:
            info["episode_summary"] = self._build_summary().model_dump()
        return info

    def _build_summary(self) -> EpisodeSummary:
        total_steps = max(1, self._episode_step)
        # Normalize reward to [0,1]
        # Max possible reward ≈ 0.5 per step on average
        max_possible = total_steps * 0.5
        normalized = max(0.0, min(1.0, (self._cumulative_reward + max_possible) / (2 * max_possible)))

        money_saved = sum(
            s.amount_at_risk for s in self._scenarios
        ) - self._money_lost

        return EpisodeSummary(
            total_reward=round(self._cumulative_reward, 3),
            normalized_score=round(normalized, 3),
            turns_completed=self._episode_step,
            dangerous_actions_taken=self._dangerous_count,
            optimal_actions_taken=self._optimal_count,
            acceptable_actions_taken=self._acceptable_count,
            money_saved=money_saved,
            money_lost=self._money_lost,
            scenarios_completed=[s.id for s in self._scenarios[:self._current_scenario_idx + 1]],
        )
