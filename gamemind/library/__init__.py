"""Persistent game/level library, sessions, skills, queues, and worker control.

The training engine itself lives in :mod:`gamemind.training`. This package is
the layer that remembers *what* was trained, for *which* game and level, with
which configuration, and what the measured result was.

Typical programmatic use::

    from gamemind.library import GameLibrary, TrainingController

    library = GameLibrary()
    library.ensure_migrated()
    game = library.get_game("geometry_dash")
    level = library.list_levels(game.id)[0]
    TrainingController(library).start_session(level, timesteps=5_000)

The Streamlit dashboard uses exactly these functions.
"""

from __future__ import annotations

from gamemind.library.catalog import ADAPTER_HELP, AdapterOption, catalog, option_for, readiness
from gamemind.library.control import TrainingController, WorkerState
from gamemind.library.models import (
    GameProfile,
    LevelProfile,
    QueueRecord,
    QueueTask,
    SessionRecord,
    SkillRecord,
    slugify,
)
from gamemind.library.progress import (
    LevelProgress,
    STATUS_LABELS,
    describe_level,
    diagnose,
    is_stagnating,
    session_rows,
)
from gamemind.library.queue import (
    create_queue,
    list_queues,
    load_queue,
    queue_summary,
    recover_queue,
    run_queue,
    save_queue,
)
from gamemind.library.runner import SessionRequest, evaluate_session_model, run_session
from gamemind.library.sessions import (
    best_validated_session,
    detect_completion,
    level_summary,
    new_session_id,
    record_from_run,
)
from gamemind.library.skills import (
    add_evidence,
    add_skill,
    applicable_skills,
    is_compatible,
    skills_for_level,
    update_skill,
)
from gamemind.library.store import (
    DEFAULT_SETTINGS,
    GameLibrary,
    default_library,
    level_config_name,
)
from gamemind.library.transfer import (
    Initialization,
    choose_initialization,
    compatibility_checks,
    find_transfer_source,
    level_model_inventory,
)
from gamemind.library.validation import ValidationReport, validate_level

__all__ = [
    "ADAPTER_HELP",
    "AdapterOption",
    "DEFAULT_SETTINGS",
    "GameLibrary",
    "GameProfile",
    "Initialization",
    "LevelProfile",
    "LevelProgress",
    "QueueRecord",
    "QueueTask",
    "STATUS_LABELS",
    "SessionRecord",
    "SessionRequest",
    "SkillRecord",
    "TrainingController",
    "ValidationReport",
    "WorkerState",
    "add_evidence",
    "add_skill",
    "applicable_skills",
    "best_validated_session",
    "catalog",
    "choose_initialization",
    "compatibility_checks",
    "create_queue",
    "default_library",
    "describe_level",
    "detect_completion",
    "diagnose",
    "evaluate_session_model",
    "find_transfer_source",
    "is_compatible",
    "is_stagnating",
    "level_config_name",
    "level_model_inventory",
    "level_summary",
    "list_queues",
    "load_queue",
    "new_session_id",
    "option_for",
    "queue_summary",
    "readiness",
    "record_from_run",
    "recover_queue",
    "run_queue",
    "run_session",
    "save_queue",
    "session_rows",
    "skills_for_level",
    "slugify",
    "update_skill",
    "validate_level",
]
