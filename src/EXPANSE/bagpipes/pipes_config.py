"""Selecting and recording the Bagpipes configuration (SPS / nebular grids).

A Bagpipes run in EXPANSE picks its configuration from the run ``meta``:

- ``"config"``: name of a Bagpipes configuration, passed to
  ``bagpipes.set_config`` (e.g. ``"bpass"``, ``"BC03_v1_3"``, or any of
  ``bagpipes.list_available_configs()``).
- ``"use_bpass"``: legacy boolean flag. ``True`` is equivalent to
  ``"config": "bpass"``, ``False`` to the Bagpipes default.

Bagpipes writes the configuration it actually used into each posterior .h5
file; the helpers here read it back so that it can be stored with the EXPANSE
outputs and restored when a posterior is reloaded.
"""

import ast
import importlib
import os
import warnings

# Used only if the installed Bagpipes does not expose its own default.
_FALLBACK_DEFAULT_CONFIG = "BC03_v1_3"
_CONFIG_MODULE_PREFIX = "bagpipes.configs.config_"

# Keys written to the meta of the EXPANSE Bagpipes tables.
CONFIG_META_KEYS = {
    "name": "bagpipes_config",
    "stellar_file": "bagpipes_stellar_file",
    "neb_line_file": "bagpipes_neb_line_file",
    "neb_cont_file": "bagpipes_neb_cont_file",
}


def normalise_config_name(config):
    """Config name without the optional single leading underscore."""
    config = str(config)
    return config[1:] if config.startswith("_") else config


def resolve_config_from_meta(meta):
    """Return ``(config, use_bpass)`` requested by a run ``meta`` dictionary.

    ``config`` is None if no explicit configuration was requested, in which
    case ``use_bpass`` (default False) decides.
    """
    config = meta.get("config", None)
    use_bpass = bool(meta.get("use_bpass", False))
    if config is not None:
        config = normalise_config_name(config)
        if use_bpass and config != "bpass":
            raise ValueError(
                f"Conflicting Bagpipes configuration in meta: 'config' is '{config}' "
                "but 'use_bpass' is True. Remove 'use_bpass' when setting 'config'."
            )
    return config, use_bpass


def get_active_config_name(pipes=None):
    """Name of the Bagpipes configuration that is currently active."""
    if pipes is None:
        import bagpipes as pipes
    if hasattr(pipes, "get_config_name"):
        return pipes.get_config_name()
    name = getattr(pipes.config, "__name__", "")
    if name.startswith(_CONFIG_MODULE_PREFIX):
        return name[len(_CONFIG_MODULE_PREFIX) :]
    return None


def set_config_environment(config=None, use_bpass=False):
    """Set the environment variables Bagpipes reads its configuration from.

    These decide the configuration when Bagpipes is first imported (including
    in subprocesses / MPI ranks, which inherit the environment).
    """
    if config is not None:
        # use_bpass takes precedence over PIPES_CONFIG_NAME inside Bagpipes
        # whenever it is set (even to 0), so it has to go.
        os.environ.pop("use_bpass", None)
        os.environ["PIPES_CONFIG_NAME"] = normalise_config_name(config)
    else:
        os.environ["use_bpass"] = str(int(bool(use_bpass)))


def set_bagpipes_config(config=None, use_bpass=False):
    """Activate a Bagpipes configuration and return the bagpipes module.

    Parameters
    ----------
    config : str, optional
        Explicit configuration name, as accepted by ``bagpipes.set_config``.
    use_bpass : bool, optional
        Legacy flag, used only if ``config`` is None.

    Works whether or not Bagpipes has already been imported: the environment
    is set for a first import, and ``bagpipes.set_config`` is called if the
    active configuration is not the requested one.
    """
    if config is not None:
        config = normalise_config_name(config)
        if use_bpass and config != "bpass":
            raise ValueError(f"use_bpass=True conflicts with config='{config}'.")
    set_config_environment(config=config, use_bpass=use_bpass)

    import bagpipes as pipes

    if not hasattr(pipes, "set_config"):
        if config is not None:
            raise RuntimeError(
                "The installed Bagpipes has no set_config(); an explicit 'config' "
                "cannot be used with it (only the legacy 'use_bpass' flag)."
            )
        return pipes

    if config is not None:
        name = config
    elif use_bpass:
        name = "bpass"
    else:
        name = getattr(pipes, "default_config_name", _FALLBACK_DEFAULT_CONFIG)

    if get_active_config_name(pipes) != name:
        available = pipes.list_available_configs()
        if available and name not in available:
            raise ValueError(f"Unknown Bagpipes config '{name}'. Available configs: {available}")
        pipes.set_config(name)
        # set_config falls back to a default config if the import fails.
        if get_active_config_name(pipes) != name:
            raise RuntimeError(
                f"Bagpipes could not load config '{name}' "
                f"(active config is '{get_active_config_name(pipes)}')."
            )
        # set_config exports PIPES_CONFIG_NAME itself; keep ours consistent.
        set_config_environment(config=config, use_bpass=use_bpass)

    return pipes


def set_bagpipes_config_from_meta(meta):
    """``set_bagpipes_config`` for a run ``meta`` dictionary."""
    config, use_bpass = resolve_config_from_meta(meta)
    return set_bagpipes_config(config=config, use_bpass=use_bpass)


def get_config_module(config=None, use_bpass=False):
    """Bagpipes config module for ``config``, without changing the active one.

    With no arguments this is the active configuration.
    """
    if config is None and use_bpass:
        config = "bpass"
    if config is None:
        from bagpipes import config as config_module

        return config_module
    config = normalise_config_name(config)
    if get_active_config_name() == config:
        from bagpipes import config as config_module

        return config_module
    return importlib.import_module(_CONFIG_MODULE_PREFIX + config)


def parse_recorded_config(config_attr):
    """Parse the ``config`` attribute of a Bagpipes posterior .h5 file."""
    if config_attr is None:
        return None
    if isinstance(config_attr, bytes):
        config_attr = config_attr.decode("utf-8")
    if isinstance(config_attr, dict):
        return config_attr
    try:
        config_used = ast.literal_eval(str(config_attr))
    except (ValueError, SyntaxError):
        return None
    return config_used if isinstance(config_used, dict) else None


def read_recorded_config(h5_path):
    """Configuration dictionary recorded in a Bagpipes posterior .h5 file.

    Returns None if the file does not record one.
    """
    import h5py

    with h5py.File(h5_path, "r") as f:
        config_attr = f.attrs.get("config", None)
    return parse_recorded_config(config_attr)


def recorded_config_request(config_used):
    """Return ``(config, use_bpass)`` reproducing a recorded configuration.

    New files record the config ``name``; older ones only a ``type`` of
    ``'BPASS'`` or ``'BC03'``, which maps onto the legacy flag. Returns
    ``(None, None)`` if nothing usable was recorded.
    """
    if not config_used:
        return None, None
    if config_used.get("name", None) is not None:
        return normalise_config_name(config_used["name"]), False
    mtype = config_used.get("type", None)
    if mtype == "BPASS":
        return None, True
    if mtype == "BC03":
        return None, False
    return None, None


def config_meta_from_h5(h5_path):
    """Table-meta entries describing the config recorded in a posterior .h5.

    Empty if the file is missing or was written by a Bagpipes version that
    did not record the config name.
    """
    if not os.path.exists(h5_path):
        return {}
    config_used = read_recorded_config(h5_path)
    if not config_used or config_used.get("name", None) is None:
        return {}
    return {
        meta_key: str(config_used[key])
        for key, meta_key in CONFIG_META_KEYS.items()
        if config_used.get(key, None) is not None
    }


def warn_if_config_changed(config=None, use_bpass=False):
    """Warn if the active config is no longer the one requested for a run.

    This can happen if existing posteriors fitted with another configuration
    were reloaded during the run (Bagpipes switches to the recorded config).
    """
    import bagpipes as pipes

    if not hasattr(pipes, "set_config"):
        return
    if config is not None:
        name = normalise_config_name(config)
    elif use_bpass:
        name = "bpass"
    else:
        name = getattr(pipes, "default_config_name", _FALLBACK_DEFAULT_CONFIG)
    active = get_active_config_name(pipes)
    if active != name:
        warnings.warn(
            f"Bagpipes config changed during the run: requested '{name}' but '{active}' "
            "is now active. Existing posteriors fitted with a different config were "
            "probably reloaded; rerun with overwrite=True for a consistent set.",
            stacklevel=2,
        )
