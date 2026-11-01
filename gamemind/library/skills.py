"""Shared, per-game knowledge with explicit evidence and confidence.

A skill is a short, human-readable note plus the conditions under which it was
observed and the measurements that support it. Skills are never a proxy for a
trained policy: a level's own checkpoint and evaluation remain the record of
what that level's agent can do.

Skills belong to exactly one game. They are not shared across games, because
"the jump timing that works in Geometry Dash" says nothing about another game.
"""

from __future__ import annotations

from typing import Any, Mapping

from gamemind.config import GameConfig
from gamemind.library.models import LevelProfile, SessionRecord, SkillRecord
from gamemind.library.store import GameLibrary
from gamemind.utils.serialization import utc_now_iso


def action_signature(config: GameConfig) -> str:
    return ",".join(binding.name for binding in config.actions.bindings) or "(native action space)"


def conditions_for(config: GameConfig, *, level_id: str | None = None) -> dict[str, Any]:
    """The technical conditions that make a skill applicable."""
    conditions: dict[str, Any] = {
        "adapter": config.adapter,
        "observation_mode": config.observation.mode,
        "width": config.observation.width,
        "height": config.observation.height,
        "stack": config.observation.stack,
        "frame_skip": config.actions.frame_skip,
        "action_signature": action_signature(config),
    }
    if level_id:
        conditions["level_id"] = level_id
    return conditions


def is_compatible(skill: SkillRecord, config: GameConfig) -> tuple[bool, list[str]]:
    """Whether a skill's conditions still describe the target environment."""
    reasons: list[str] = []
    conditions = skill.conditions or {}
    if conditions.get("adapter") and conditions["adapter"] != config.adapter:
        reasons.append(f"adapter {conditions['adapter']} != {config.adapter}")
    if conditions.get("observation_mode") and conditions["observation_mode"] != config.observation.mode:
        reasons.append(
            f"observation mode {conditions['observation_mode']} != {config.observation.mode}"
        )
    for key, current in (
        ("width", config.observation.width),
        ("height", config.observation.height),
        ("stack", config.observation.stack),
    ):
        if conditions.get(key) is not None and conditions[key] != current:
            reasons.append(f"{key} {conditions[key]} != {current}")
    if conditions.get("frame_skip") is not None and conditions["frame_skip"] != config.actions.frame_skip:
        reasons.append(f"frame_skip {conditions['frame_skip']} != {config.actions.frame_skip}")
    if conditions.get("action_signature") and conditions["action_signature"] != action_signature(config):
        reasons.append("action bindings differ")
    return not reasons, reasons


def add_skill(
    library: GameLibrary,
    game_id: str,
    *,
    name: str,
    kind: str,
    summary: str,
    conditions: Mapping[str, Any] | None = None,
    evidence: list[Mapping[str, Any]] | None = None,
    confidence: float = 0.2,
    validated: bool = False,
) -> SkillRecord:
    skill = SkillRecord(
        skill_id=library.unique_skill_id(name),
        game_id=game_id,
        name=name,
        kind=kind,
        summary=summary,
        confidence=confidence,
        validated=validated,
        conditions=dict(conditions or {}),
        evidence=[dict(item) for item in (evidence or [])],
    )
    library.save_skill(skill)
    return skill


def update_skill(
    library: GameLibrary,
    skill_id: str,
    *,
    confidence: float | None = None,
    validated: bool | None = None,
    summary: str | None = None,
    conditions: Mapping[str, Any] | None = None,
) -> SkillRecord:
    skill = library.get_skill(skill_id)
    payload = skill.to_dict()
    if confidence is not None:
        payload["confidence"] = float(confidence)
    if validated is not None:
        payload["validated"] = bool(validated)
        if validated:
            payload["confidence"] = max(float(payload["confidence"]), 0.8)
    if summary is not None:
        payload["summary"] = summary
    if conditions is not None:
        payload["conditions"] = dict(conditions)
    payload["updated_at"] = utc_now_iso()
    updated = SkillRecord.from_dict(payload)
    library.save_skill(updated)
    return updated


def add_evidence(
    library: GameLibrary,
    skill_id: str,
    evidence: Mapping[str, Any],
    *,
    confidence_step: float = 0.15,
) -> SkillRecord:
    """Attach a measurement to a skill.

    Confidence rises with each *distinct* supporting observation and is capped
    below certainty: repeated measurements are evidence, not proof.
    """
    skill = library.get_skill(skill_id)
    key = (str(evidence.get("session_id")), str(evidence.get("metric")), str(evidence.get("level_id")))
    existing = {
        (str(item.get("session_id")), str(item.get("metric")), str(item.get("level_id")))
        for item in skill.evidence
    }
    payload = skill.to_dict()
    if key not in existing:
        payload["evidence"] = [*skill.evidence, dict(evidence)]
        payload["confidence"] = min(0.95, float(skill.confidence) + confidence_step)
    payload["updated_at"] = utc_now_iso()
    updated = SkillRecord.from_dict(payload)
    library.save_skill(updated)
    return updated


def session_evidence(level: LevelProfile, session: SessionRecord) -> dict[str, Any]:
    """A factual summary of one session that can support a skill."""
    evaluation = session.evaluation or {}
    return {
        "session_id": session.session_id,
        "level_id": level.id,
        "metric": "mean_max_progress_pct",
        "value": evaluation.get("mean_progress_pct")
        if evaluation.get("mean_progress_pct") is not None
        else session.metrics.get("best_progress_pct"),
        "timesteps": session.final_timesteps,
        "state": session.state,
        "verified_completion": session.verified,
        "recorded_at": session.finished_at or session.started_at,
    }


def applicable_skills(
    library: GameLibrary, game_id: str, config: GameConfig, *, validated_only: bool = True
) -> list[SkillRecord]:
    """Skills from this game whose conditions still match the target level."""
    result: list[SkillRecord] = []
    for skill in library.list_skills(game_id):
        if validated_only and not skill.validated:
            continue
        ok, _reasons = is_compatible(skill, config)
        if ok:
            result.append(skill)
    result.sort(key=lambda skill: skill.confidence, reverse=True)
    return result


def skills_for_level(
    library: GameLibrary, level: LevelProfile
) -> tuple[list[SkillRecord], list[tuple[SkillRecord, list[str]]]]:
    """Split a game's skills into applicable and inapplicable for one level."""
    config = library.level_config(level.game_id, level.id)
    applicable: list[SkillRecord] = []
    rejected: list[tuple[SkillRecord, list[str]]] = []
    for skill in library.list_skills(level.game_id):
        ok, reasons = is_compatible(skill, config)
        if ok:
            applicable.append(skill)
        else:
            rejected.append((skill, reasons))
    return applicable, rejected


__all__ = [
    "action_signature",
    "add_evidence",
    "add_skill",
    "applicable_skills",
    "conditions_for",
    "is_compatible",
    "session_evidence",
    "skills_for_level",
    "update_skill",
]
