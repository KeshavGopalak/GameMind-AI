"""Reward shaping and optional visual termination detectors."""

from gamemind.rewards.base import (
    RewardComponent,
    RewardContext,
    RewardFunction,
    build_reward_function,
)
from gamemind.rewards.components import (
    ConstantReward,
    CustomReward,
    EventReward,
    NativeReward,
    ProgressReward,
    SignalReward,
    SurvivalReward,
    TemplateMatchReward,
    create_component,
)
from gamemind.rewards.termination import (
    TemplateTerminationDetector,
    build_termination_detector,
)

__all__ = [
    "ConstantReward",
    "CustomReward",
    "EventReward",
    "NativeReward",
    "ProgressReward",
    "RewardComponent",
    "RewardContext",
    "RewardFunction",
    "SignalReward",
    "SurvivalReward",
    "TemplateMatchReward",
    "TemplateTerminationDetector",
    "build_reward_function",
    "build_termination_detector",
    "create_component",
]
