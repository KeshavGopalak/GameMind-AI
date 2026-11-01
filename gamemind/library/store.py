"""Persistent storage for games, levels, and their configurations.

Layout (created on first use, all paths relative to the project)::

    library/
      settings.json
      games/
        <game_id>/
          game.json              game profile
          game.yaml              default configuration for this game
          levels/
            <level_id>/
              level.json         level profile
              config.yaml        resolved configuration used for training
              sessions/          one JSON record per training session
              evaluations/       one JSON record per evaluation run
      knowledge/
        skills/<skill_id>.json
      state/                     worker/queue runtime state (not tracked by Git)

Large artifacts are *not* stored here. Models live under
``models/library/<game>/<level>/`` and checkpoints under
``checkpoints/library/<game>/`` so they stay in the existing, Git-ignored roots.
"""

from __future__ import annotations

import copy
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from gamemind.config import GameConfig, load_game_config
from gamemind.library.catalog import ADAPTER_HELP, option_for, readiness
from gamemind.library.models import (
    GameProfile,
    LevelProfile,
    SessionRecord,
    SkillRecord,
    slugify,
)
from gamemind.training.model_io import metadata_path_for
from gamemind.utils.errors import LibraryError
from gamemind.utils.paths import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_MODEL_DIR,
    PROJECT_ROOT,
    ensure_dir,
    resolve_project_path,
)
from gamemind.utils.serialization import (
    atomic_write_json,
    atomic_write_text,
    read_json,
    utc_now_iso,
)

DEFAULT_SETTINGS: dict[str, Any] = {
    "default_timesteps": 50_000,
    "default_eval_episodes": 3,
    "device": "auto",
    "evaluate_after_training": True,
}

# Games migrated into the library on first launch. Only these are seeded; the
# migration never overwrites an existing profile or level.
SEEDED_GAMES: tuple[tuple[str, str, str, tuple[tuple[str, str, str], ...]], ...] = (
    (
        "geometry_dash",
        "Geometry Dash",
        "configs/geometry_dash.yaml",
        (
            (
                "stereo_madness",
                "Stereo Madness",
                "Seeded from configs/geometry_dash.yaml during the library migration. "
                "The level used by the earlier 20,000-step checkpoint was not recorded, so "
                "initialization for this level defaults to scratch. Confirm the level name "
                "matches the level you open in game.",
            ),
        ),
    ),
)


def level_config_name(game_id: str, level_id: str) -> str:
    """Stable environment name for one level.

    A distinct ``game.name`` per level is what keeps checkpoint directories,
    model metadata, and resume validation separate for every level.
    """
    return f"{game_id}__{level_id}"


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except OSError as exc:
        raise LibraryError(f"Could not read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise LibraryError(f"Could not parse YAML in {path}:\n{exc}") from exc
    if not isinstance(data, Mapping):
        raise LibraryError(f"{path} must contain a YAML mapping.")
    return dict(data)


def _dump_yaml(payload: Mapping[str, Any]) -> str:
    return yaml.safe_dump(dict(payload), sort_keys=False)


# Which adapters can actually detect a finished level. Anything else must keep
# completion disabled rather than offering a switch that does nothing.
COMPLETION_ADAPTERS = {"geometry_dash": "bar"}


def _require_supported_completion(adapter: str, completion_mode: str) -> None:
    if completion_mode == "disabled":
        return
    supported = COMPLETION_ADAPTERS.get(str(adapter).strip().lower())
    if supported != completion_mode:
        raise LibraryError(
            f"The {completion_mode!r} completion detector is not implemented for the "
            f"{adapter!r} adapter. Leave completion detection disabled for this game; "
            f"GameMind will not invent a win signal it cannot read. "
            f"Detectors available: {', '.join(sorted(COMPLETION_ADAPTERS)) or 'none'}."
        )


class GameLibrary:
    """Filesystem-backed registry of games, levels, sessions, and skills."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = ensure_dir(resolve_project_path(root) if root else PROJECT_ROOT / "library")
        self.games_dir = ensure_dir(self.root / "games")
        self.knowledge_dir = ensure_dir(self.root / "knowledge")
        self.skills_dir = ensure_dir(self.knowledge_dir / "skills")
        self.state_dir = ensure_dir(self.root / "state")
        self.settings_path = self.root / "settings.json"

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    def game_dir(self, game_id: str) -> Path:
        return self.games_dir / game_id

    def game_json_path(self, game_id: str) -> Path:
        return self.game_dir(game_id) / "game.json"

    def game_config_path(self, game_id: str) -> Path:
        return self.game_dir(game_id) / "game.yaml"

    def levels_dir(self, game_id: str) -> Path:
        return self.game_dir(game_id) / "levels"

    def level_dir(self, game_id: str, level_id: str) -> Path:
        return self.levels_dir(game_id) / level_id

    def level_json_path(self, game_id: str, level_id: str) -> Path:
        return self.level_dir(game_id, level_id) / "level.json"

    def level_config_path(self, game_id: str, level_id: str) -> Path:
        return self.level_dir(game_id, level_id) / "config.yaml"

    def level_models_dir(self, game_id: str, level_id: str) -> Path:
        return ensure_dir(self.level_dir(game_id, level_id) / "models")

    def level_checkpoints_dir(self, game_id: str, level_id: str) -> Path:
        return ensure_dir(self.level_dir(game_id, level_id) / "checkpoints")

    def level_runs_dir(self, game_id: str, level_id: str) -> Path:
        return ensure_dir(self.level_dir(game_id, level_id) / "runs")

    def level_run_dir(self, game_id: str, level_id: str, run_name: str) -> Path:
        return self.level_runs_dir(game_id, level_id) / run_name

    def level_model_path(self, game_id: str, level_id: str, session_id: str) -> Path:
        return self.level_models_dir(game_id, level_id) / f"{session_id}.zip"

    def validation_path(self, game_id: str, level_id: str) -> Path:
        return self.level_dir(game_id, level_id) / "validation.json"

    def save_validation(
        self, game_id: str, level_id: str, payload: Mapping[str, Any]
    ) -> Path:
        path = self.validation_path(game_id, level_id)
        atomic_write_json(path, dict(payload))
        return path

    def load_validation(self, game_id: str, level_id: str) -> dict[str, Any] | None:
        value = read_json(self.validation_path(game_id, level_id), default=None)
        return value if isinstance(value, dict) else None

    def sessions_dir(self, game_id: str, level_id: str) -> Path:
        return ensure_dir(self.level_dir(game_id, level_id) / "sessions")

    def session_path(self, game_id: str, level_id: str, session_id: str) -> Path:
        return self.sessions_dir(game_id, level_id) / f"{session_id}.json"

    def evaluations_dir(self, game_id: str, level_id: str) -> Path:
        return ensure_dir(self.level_dir(game_id, level_id) / "evaluations")

    def evaluation_path(self, game_id: str, level_id: str, evaluation_id: str) -> Path:
        return self.evaluations_dir(game_id, level_id) / f"{evaluation_id}.json"

    def skill_path(self, skill_id: str) -> Path:
        return self.skills_dir / f"{skill_id}.json"

    def _unique_child(self, parent: Path, base: str) -> str:
        candidate = base
        counter = 2
        while (parent / candidate).exists():
            candidate = f"{base}_{counter}"
            counter += 1
        return candidate

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    def settings(self) -> dict[str, Any]:
        value = read_json(self.settings_path, default={})
        merged = dict(DEFAULT_SETTINGS)
        if isinstance(value, dict):
            merged.update(value)
        return merged

    def update_settings(self, **values: Any) -> dict[str, Any]:
        payload = self.settings()
        for key, value in values.items():
            if key not in DEFAULT_SETTINGS:
                raise LibraryError(f"Unknown setting {key!r}.")
            payload[key] = value
        atomic_write_json(self.settings_path, payload)
        return payload

    # ------------------------------------------------------------------
    # Games
    # ------------------------------------------------------------------
    def list_games(self, *, include_archived: bool = False) -> list[GameProfile]:
        games: list[GameProfile] = []
        if not self.games_dir.exists():
            return games
        for path in sorted(self.games_dir.iterdir()):
            if not path.is_dir():
                continue
            record_path = path / "game.json"
            if not record_path.exists():
                continue
            profile = GameProfile.from_dict(read_json(record_path, default={}) or {})
            if profile.archived and not include_archived:
                continue
            games.append(profile)
        return games

    def get_game(self, game_id: str) -> GameProfile:
        path = self.game_json_path(game_id)
        if not path.exists():
            raise LibraryError(
                f"No game profile {game_id!r} in the library. Open the Game Library and add it first."
            )
        return GameProfile.from_dict(read_json(path, default={}) or {})

    def create_game(
        self,
        name: str,
        adapter: str,
        *,
        game_id: str | None = None,
        description: str = "",
        setup_notes: str = "",
        config_text: str | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> GameProfile:
        """Register a game that this installation can actually connect to."""
        adapter = str(adapter or "").strip().lower()
        usable, why = readiness(adapter)
        if not usable:
            raise LibraryError(why)

        identifier = slugify(game_id or name, fallback="game")
        identifier = self._unique_child(self.games_dir, identifier)
        profile = GameProfile(
            id=identifier,
            name=name,
            adapter=adapter,
            description=description or "",
            setup_notes=setup_notes or "",
        )

        base_data, effective_adapter = self._initial_game_config(
            profile, config_text=config_text, params=params
        )
        profile.adapter = effective_adapter
        config = GameConfig.from_dict(base_data)
        if config.adapter.strip().lower() != effective_adapter:
            raise LibraryError(
                f"Configuration uses adapter {config.adapter!r} but the game profile asks for "
                f"{effective_adapter!r}. Use one adapter per game profile."
            )
        if not config.actions.has_bindings and config.adapter.strip().lower() in {
            "geometry_dash",
            "desktop",
        }:
            raise LibraryError(
                f"The {config.adapter!r} adapter needs action bindings; the supplied "
                "configuration has none."
            )

        directory = ensure_dir(self.game_dir(identifier))
        atomic_write_text(self.game_config_path(identifier), _dump_yaml(base_data))
        atomic_write_json(directory / "game.json", profile.to_dict())
        ensure_dir(self.levels_dir(identifier))
        return profile

    def _initial_game_config(
        self,
        profile: GameProfile,
        *,
        config_text: str | None,
        params: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], str]:
        if config_text is not None:
            try:
                data = yaml.safe_load(config_text)
            except yaml.YAMLError as exc:
                raise LibraryError(f"Could not parse the supplied configuration:\n{exc}") from exc
            if not isinstance(data, Mapping):
                raise LibraryError("The supplied configuration must be a YAML mapping.")
            data = copy.deepcopy(dict(data))
            data.setdefault("game", {})
            data["game"]["name"] = profile.id
            effective = str(data["game"].get("adapter") or profile.adapter).strip().lower()
            data["game"]["adapter"] = effective
            return data, effective

        option = option_for(profile.adapter)
        template = option.template_config if option is not None else None
        if not template:
            raise LibraryError(
                f"There is no ready-made configuration for adapter {profile.adapter!r}. "
                f"{ADAPTER_HELP} Provide a YAML configuration for this game instead."
            )
        template_path = resolve_project_path(template)
        if not template_path.exists():
            raise LibraryError(f"Template configuration is missing: {template_path}")
        data = _read_yaml(template_path)
        data = copy.deepcopy(data)
        data.setdefault("game", {})
        data["game"]["name"] = profile.id
        game_params = dict(data["game"].get("params") or {})

        if option is not None and option.needs_env_id:
            env_id = ""
            if params:
                env_id = str(params.get("env_id") or "").strip()
            if not env_id:
                raise LibraryError(
                    "This adapter needs a Gymnasium environment id, for example 'CartPole-v1'."
                )
            self._check_gymnasium_env_id(env_id)
            game_params["env_id"] = env_id
        elif params:
            game_params.update(dict(params))
        data["game"]["params"] = game_params
        return data, profile.adapter

    @staticmethod
    def _check_gymnasium_env_id(env_id: str) -> None:
        try:
            import gymnasium as gym

            gym.spec(env_id)
        except Exception as exc:  # missing package, unregistered id, broken install
            raise LibraryError(
                f"Gymnasium environment {env_id!r} is not available in this installation: {exc}. "
                "Install the package that provides it, or check the exact id."
            ) from exc

    def update_game(self, game_id: str, **fields: Any) -> GameProfile:
        profile = self.get_game(game_id)
        allowed = {"name", "description", "setup_notes", "archived"}
        unknown = set(fields) - allowed
        if unknown:
            raise LibraryError(f"Unknown game field(s): {', '.join(sorted(unknown))}.")
        payload = profile.to_dict()
        payload.update(fields)
        payload["updated_at"] = utc_now_iso()
        updated = GameProfile.from_dict(payload)
        atomic_write_json(self.game_json_path(game_id), updated.to_dict())
        return updated

    def archive_game(self, game_id: str, *, archived: bool = True) -> GameProfile:
        return self.update_game(game_id, archived=archived)

    def remove_game(self, game_id: str, *, remove_levels: bool = False) -> None:
        """Remove a game profile.

        Refuses while levels exist unless ``remove_levels`` is set, so a stray
        click cannot delete a game with trained levels. Model and checkpoint
        files on disk are never deleted here.
        """
        levels = self.list_levels(game_id, include_archived=True)
        if levels and not remove_levels:
            raise LibraryError(
                f"{game_id!r} still has {len(levels)} level(s). Remove or archive them first, "
                "or confirm by passing remove_levels=True."
            )
        directory = self.game_dir(game_id).resolve()
        if self.games_dir.resolve() not in directory.parents:
            raise LibraryError("Refusing to delete a path outside the library.")
        if directory.exists():
            shutil.rmtree(directory)

    def game_config(self, game_id: str) -> GameConfig:
        path = self.game_config_path(game_id)
        if not path.exists():
            raise LibraryError(f"Game configuration is missing: {path}")
        return load_game_config(path)

    def save_game_config_text(self, game_id: str, text: str) -> GameConfig:
        """Replace a game's default configuration.

        Existing levels keep their own ``config.yaml`` untouched: changing the
        default must not silently rewrite a trained level's observation pipeline
        or action timings. New levels use the new defaults.
        """
        profile = self.get_game(game_id)
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise LibraryError(f"Could not parse YAML:\n{exc}") from exc
        if not isinstance(data, Mapping):
            raise LibraryError("The configuration must be a YAML mapping.")
        data = copy.deepcopy(dict(data))
        data.setdefault("game", {})
        data["game"]["adapter"] = profile.adapter
        data["game"]["name"] = profile.id
        config = GameConfig.from_dict(data)
        atomic_write_text(self.game_config_path(game_id), _dump_yaml(data))
        return config

    # ------------------------------------------------------------------
    # Levels
    # ------------------------------------------------------------------
    def list_levels(self, game_id: str, *, include_archived: bool = False) -> list[LevelProfile]:
        directory = self.levels_dir(game_id)
        levels: list[LevelProfile] = []
        if not directory.exists():
            return levels
        for path in sorted(directory.iterdir()):
            if not path.is_dir():
                continue
            record = path / "level.json"
            if not record.exists():
                continue
            profile = LevelProfile.from_dict(read_json(record, default={}) or {})
            if profile.archived and not include_archived:
                continue
            levels.append(profile)
        return levels

    def get_level(self, game_id: str, level_id: str) -> LevelProfile:
        path = self.level_json_path(game_id, level_id)
        if not path.exists():
            raise LibraryError(
                f"No level profile {level_id!r} for game {game_id!r}. Add the level first."
            )
        return LevelProfile.from_dict(read_json(path, default={}) or {})

    def create_level(
        self,
        game_id: str,
        name: str,
        *,
        level_id: str | None = None,
        kind: str = "official",
        entry: str = "manual",
        completion_mode: str = "disabled",
        completion_pct: float = 99.0,
        completion_frames: int = 2,
        init_policy: str = "best_same_level",
        notes: str = "",
        config_text: str | None = None,
    ) -> LevelProfile:
        game = self.get_game(game_id)
        base = self.game_config(game_id)
        _require_supported_completion(base.adapter, completion_mode)
        directory = self.levels_dir(game_id)
        ensure_dir(directory)
        identifier = slugify(level_id or name, fallback="level")
        identifier = self._unique_child(directory, identifier)
        profile = LevelProfile(
            id=identifier,
            game_id=game.id,
            name=name,
            kind=kind,
            entry=entry,
            completion_mode=completion_mode,
            completion_pct=completion_pct,
            completion_frames=completion_frames,
            init_policy=init_policy,
            notes=notes,
        )
        ensure_dir(self.level_dir(game_id, identifier))
        self._write_level_config(game_id, profile, base=base, config_text=config_text)
        atomic_write_json(self.level_json_path(game_id, identifier), profile.to_dict())
        self.sessions_dir(game_id, identifier)
        self.evaluations_dir(game_id, identifier)
        return profile

    def update_level(self, game_id: str, level_id: str, **fields: Any) -> LevelProfile:
        profile = self.get_level(game_id, level_id)
        allowed = {
            "name",
            "kind",
            "entry",
            "completion_mode",
            "completion_pct",
            "completion_frames",
            "init_policy",
            "notes",
            "archived",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise LibraryError(f"Unknown level field(s): {', '.join(sorted(unknown))}.")
        payload = profile.to_dict()
        if "name" in fields and not str(fields["name"]).strip():
            raise LibraryError("A level name cannot be empty.")
        payload.update(fields)
        payload["updated_at"] = utc_now_iso()
        updated = LevelProfile.from_dict(payload)
        # The name is display-only; the id (and therefore every path) is stable.
        updated.id = profile.id
        _require_supported_completion(self.game_config(game_id).adapter, updated.completion_mode)
        atomic_write_json(self.level_json_path(game_id, level_id), updated.to_dict())
        self._write_level_config(game_id, updated)
        return updated

    def duplicate_level(
        self, game_id: str, level_id: str, *, new_name: str | None = None
    ) -> LevelProfile:
        """Copy a level's settings into a new level.

        Training history is intentionally not copied: sessions and checkpoints
        belong to the level that produced them.
        """
        source = self.get_level(game_id, level_id)
        source_config = self.level_config_path(game_id, level_id)
        return self.create_level(
            game_id,
            new_name or f"{source.name} copy",
            kind=source.kind,
            entry=source.entry,
            completion_mode=source.completion_mode,
            completion_pct=source.completion_pct,
            completion_frames=source.completion_frames,
            init_policy=source.init_policy,
            notes=source.notes,
            config_text=source_config.read_text(encoding="utf-8") if source_config.exists() else None,
        )

    def archive_level(self, game_id: str, level_id: str, *, archived: bool = True) -> LevelProfile:
        return self.update_level(game_id, level_id, archived=archived)

    def remove_level(self, game_id: str, level_id: str) -> dict[str, Any]:
        """Delete a level profile and its history, archiving trained artifacts.

        Models, checkpoints, and run directories are moved to
        ``library/archive/<game>/<level>-<timestamp>/`` before the profile is
        removed. A mis-click must never destroy the only copy of a policy.
        """
        profile = self.get_level(game_id, level_id)
        directory = self.level_dir(game_id, level_id).resolve()
        levels_root = self.levels_dir(game_id).resolve()
        if levels_root not in directory.parents:
            raise LibraryError("Refusing to delete a path outside the game's levels directory.")
        sessions = len(list(self.sessions_dir(game_id, level_id).glob("*.json")))
        archived_to: Path | None = None
        preserved = [
            name for name in ("models", "checkpoints", "runs") if (directory / name).exists()
        ]
        if preserved:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            archived_to = ensure_dir(self.root / "archive" / game_id / f"{level_id}-{stamp}")
            for name in preserved:
                shutil.move(str(directory / name), str(archived_to / name))
        if directory.exists():
            shutil.rmtree(directory)
        return {
            "level_id": profile.id,
            "sessions_removed": sessions,
            "archived_to": str(archived_to) if archived_to else None,
        }

    # ------------------------------------------------------------------
    # Level configuration
    # ------------------------------------------------------------------
    def level_config(self, game_id: str, level_id: str) -> GameConfig:
        path = self.level_config_path(game_id, level_id)
        if not path.exists():
            profile = self.get_level(game_id, level_id)
            return self._write_level_config(game_id, profile, base=self.game_config(game_id))
        return load_game_config(path)

    def _level_config_payload(
        self,
        game_id: str,
        profile: LevelProfile,
        base: GameConfig,
    ) -> dict[str, Any]:
        data = base.as_dict()
        data["game"]["name"] = level_config_name(game_id, profile.id)
        params = dict(data["game"].get("params") or {})
        if base.adapter.strip().lower() == "geometry_dash":
            params["allow_bar_completion"] = bool(profile.completion_enabled)
            params["completion_pct"] = float(profile.completion_pct)
            params["completion_frames"] = int(profile.completion_frames)
        data["game"]["params"] = params
        return data

    def _write_level_config(
        self,
        game_id: str,
        profile: LevelProfile,
        *,
        base: GameConfig | None = None,
        config_text: str | None = None,
    ) -> GameConfig:
        path = self.level_config_path(game_id, profile.id)
        base = base if base is not None else self.game_config(game_id)
        if config_text is not None:
            try:
                data = yaml.safe_load(config_text)
            except yaml.YAMLError as exc:
                raise LibraryError(f"Could not parse the level configuration:\n{exc}") from exc
            if not isinstance(data, Mapping):
                raise LibraryError("A level configuration must be a YAML mapping.")
            payload = copy.deepcopy(dict(data))
            payload.setdefault("game", {})
            payload["game"]["name"] = level_config_name(game_id, profile.id)
            profile_adapter = str(payload["game"].get("adapter") or base.adapter)
            payload["game"]["adapter"] = profile_adapter
            if profile_adapter.strip().lower() != base.adapter.strip().lower():
                raise LibraryError(
                    f"The level configuration uses adapter {profile_adapter!r} but the game uses "
                    f"{base.adapter!r}. Keep one adapter per game."
                )
            params = dict(payload["game"].get("params") or {})
            if profile_adapter.strip().lower() == "geometry_dash":
                params["allow_bar_completion"] = bool(profile.completion_enabled)
                params["completion_pct"] = float(profile.completion_pct)
                params["completion_frames"] = int(profile.completion_frames)
            payload["game"]["params"] = params
            config = GameConfig.from_dict(payload)
            # Write the validated payload, not the raw text, so the stored file
            # always matches the completion fields the level profile controls.
            atomic_write_text(path, _dump_yaml(payload))
            return config

        if path.exists():
            # Merge managed fields into the level's own configuration so renaming,
            # archiving, or toggling completion never resets jump timings,
            # observation settings, or other per-level edits.
            payload = copy.deepcopy(_read_yaml(path))
        else:
            payload = self._level_config_payload(game_id, profile, base)
        payload.setdefault("game", {})
        payload["game"]["name"] = level_config_name(game_id, profile.id)
        payload["game"]["adapter"] = base.adapter
        params = dict(payload["game"].get("params") or {})
        if base.adapter.strip().lower() == "geometry_dash":
            params["allow_bar_completion"] = bool(profile.completion_enabled)
            params["completion_pct"] = float(profile.completion_pct)
            params["completion_frames"] = int(profile.completion_frames)
        payload["game"]["params"] = params
        config = GameConfig.from_dict(payload)
        atomic_write_text(path, _dump_yaml(payload))
        return config

    def save_level_config_text(self, game_id: str, level_id: str, text: str) -> GameConfig:
        profile = self.get_level(game_id, level_id)
        base = self.game_config(game_id)
        return self._write_level_config(game_id, profile, base=base, config_text=text)

    # ------------------------------------------------------------------
    # Sessions and evaluations
    # ------------------------------------------------------------------
    def save_session(self, record: SessionRecord) -> Path:
        path = self.session_path(record.game_id, record.level_id, record.session_id)
        atomic_write_json(path, record.to_dict())
        return path

    def load_session(self, game_id: str, level_id: str, session_id: str) -> SessionRecord:
        path = self.session_path(game_id, level_id, session_id)
        if not path.exists():
            raise LibraryError(f"Session record not found: {path}")
        return SessionRecord.from_dict(read_json(path, default={}) or {})

    def list_sessions(self, game_id: str, level_id: str) -> list[SessionRecord]:
        directory = self.sessions_dir(game_id, level_id)
        records: list[SessionRecord] = []
        for path in sorted(directory.glob("*.json")):
            payload = read_json(path, default=None)
            if not isinstance(payload, dict):
                continue
            try:
                records.append(SessionRecord.from_dict(payload))
            except LibraryError:
                continue
        records.sort(key=lambda record: (record.started_at, record.session_id))
        return records

    def save_evaluation(
        self,
        game_id: str,
        level_id: str,
        evaluation: Mapping[str, Any],
        *,
        evaluation_id: str | None = None,
    ) -> Path:
        identifier = evaluation_id or slugify(
            time.strftime("%Y%m%d-%H%M%S") + "-eval", fallback="eval"
        )
        path = self.evaluation_path(game_id, level_id, identifier)
        atomic_write_json(path, dict(evaluation))
        return path

    def list_evaluations(self, game_id: str, level_id: str) -> list[dict[str, Any]]:
        directory = self.evaluations_dir(game_id, level_id)
        items: list[dict[str, Any]] = []
        for path in sorted(directory.glob("*.json")):
            payload = read_json(path, default=None)
            if isinstance(payload, dict):
                payload.setdefault("evaluation_id", path.stem)
                items.append(payload)
        return items

    # ------------------------------------------------------------------
    # Skills
    # ------------------------------------------------------------------
    def list_skills(self, game_id: str | None = None) -> list[SkillRecord]:
        skills: list[SkillRecord] = []
        for path in sorted(self.skills_dir.glob("*.json")):
            payload = read_json(path, default=None)
            if not isinstance(payload, dict):
                continue
            try:
                record = SkillRecord.from_dict(payload)
            except LibraryError:
                continue
            if game_id and record.game_id != game_id:
                continue
            skills.append(record)
        return skills

    def get_skill(self, skill_id: str) -> SkillRecord:
        path = self.skill_path(skill_id)
        if not path.exists():
            raise LibraryError(f"No skill {skill_id!r} in the knowledge store.")
        return SkillRecord.from_dict(read_json(path, default={}) or {})

    def save_skill(self, record: SkillRecord) -> Path:
        path = self.skill_path(record.skill_id)
        atomic_write_json(path, record.to_dict())
        return path

    def unique_skill_id(self, name: str) -> str:
        existing = {skill.skill_id for skill in self.list_skills()}
        base = slugify(name, fallback="skill")
        candidate = base
        counter = 2
        while candidate in existing:
            candidate = f"{base}_{counter}"
            counter += 1
        return candidate

    # ------------------------------------------------------------------
    # Migration and preservation
    # ------------------------------------------------------------------
    def ensure_migrated(self) -> list[str]:
        """Create the seeded profiles once, without touching existing data."""
        actions: list[str] = []
        for game_id, name, template, levels in SEEDED_GAMES:
            if self.game_json_path(game_id).exists():
                continue
            template_path = resolve_project_path(template)
            if not template_path.exists():
                continue
            base = load_game_config(template_path)
            profile = GameProfile(
                id=game_id,
                name=name,
                adapter=base.adapter.strip().lower(),
                description=(
                    "Imported from an existing GameMind configuration. Real-game training "
                    "requires the game window to be open and focused."
                ),
                setup_notes=(
                    "Open the game, select the level, enable Auto-Retry, then keep the window "
                    "focused while training."
                ),
            )
            directory = ensure_dir(self.game_dir(game_id))
            base_payload = base.as_dict()
            base_payload["game"]["name"] = game_id
            atomic_write_text(self.game_config_path(game_id), _dump_yaml(base_payload))
            atomic_write_json(directory / "game.json", profile.to_dict())
            ensure_dir(self.levels_dir(game_id))
            actions.append(f"Created game profile {game_id!r} from {template}.")
            for level_id, level_name, notes in levels:
                if self.level_json_path(game_id, level_id).exists():
                    continue
                self.create_level(
                    game_id,
                    level_name,
                    level_id=level_id,
                    kind="official",
                    completion_mode="disabled",
                    init_policy="scratch",
                    notes=notes,
                )
                actions.append(f"Created level profile {game_id}/{level_id!r}.")
        return actions

    def leveled_artifacts(self, game_id: str, level_id: str) -> list[dict[str, Any]]:
        """Models known to belong to one level, newest first."""
        directory = self.level_models_dir(game_id, level_id)
        items: list[dict[str, Any]] = []
        if not directory.exists():
            return items
        for path in sorted(directory.glob("*.zip")):
            meta = read_json(metadata_path_for(path), default={}) or {}
            items.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "num_timesteps": meta.get("num_timesteps"),
                    "created_at": meta.get("created_at"),
                    "run_id": meta.get("run_id"),
                }
            )
        items.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return items

    def legacy_artifacts(self, game: GameProfile, *, limit: int = 120) -> list[dict[str, Any]]:
        """Model files outside the library that belong to this game.

        These are preserved artifacts from earlier runs. They are listed for
        reference and for explicit reuse, never attached to a level silently.
        """
        known: set[str] = set()
        for level in self.list_levels(game.id, include_archived=True):
            for item in self.leveled_artifacts(game.id, level.id):
                known.add(str(Path(item["path"]).resolve()))
            for session in self.list_sessions(game.id, level.id):
                for key in ("model_path", "best_model_path", "resume_from"):
                    value = getattr(session, key, None)
                    if value:
                        known.add(str(Path(value).resolve()))

        candidates: Iterable[Path] = list(DEFAULT_MODEL_DIR.rglob("*.zip"))
        candidates = list(candidates) + list(DEFAULT_CHECKPOINT_DIR.rglob("*.zip"))
        found: list[dict[str, Any]] = []
        for path in candidates:
            if len(found) >= limit:
                break
            resolved = str(path.resolve())
            if resolved in known:
                continue
            try:
                resolved_path = path.resolve()
                resolved_path.relative_to(DEFAULT_MODEL_DIR.resolve())
                in_library = "library" in resolved_path.relative_to(DEFAULT_MODEL_DIR.resolve()).parts
            except ValueError:
                in_library = False
            if in_library:
                continue
            meta = read_json(metadata_path_for(path), default={}) or {}
            meta_game = str(meta.get("game") or "")
            meta_adapter = str(meta.get("adapter") or "")
            matches = meta_game in {game.id, game.adapter} or (
                not meta_game and meta_adapter == game.adapter
            )
            if not matches:
                continue
            found.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "game": meta_game or "(no metadata)",
                    "num_timesteps": meta.get("num_timesteps"),
                    "created_at": meta.get("created_at"),
                    "has_metadata": bool(meta),
                }
            )
        found.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return found

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def level_env_config(self, game_id: str, level_id: str) -> GameConfig:
        """The exact configuration a training session for this level uses."""
        return self.level_config(game_id, level_id)

    def counts(self, game_id: str) -> dict[str, int]:
        levels = self.list_levels(game_id, include_archived=True)
        sessions = sum(len(self.list_sessions(game_id, level.id)) for level in levels)
        return {"levels": len(levels), "sessions": sessions}


def default_library() -> GameLibrary:
    """Open the project library and make sure the seeded profiles exist.

    ``GAMEMIND_LIBRARY_ROOT`` relocates the library (tests, or a shared drive)
    without touching any other configuration.
    """
    import os

    root = os.environ.get("GAMEMIND_LIBRARY_ROOT")
    library = GameLibrary(root) if root else GameLibrary()
    library.ensure_migrated()
    return library


def clone_level_profile(profile: LevelProfile, **overrides: Any) -> LevelProfile:
    """Copy a level profile with changes, revalidating every field."""
    payload = profile.to_dict()
    payload.update(overrides)
    payload["updated_at"] = utc_now_iso()
    return LevelProfile.from_dict(payload)


__all__ = [
    "GameLibrary",
    "DEFAULT_SETTINGS",
    "clone_level_profile",
    "default_library",
    "level_config_name",
]
