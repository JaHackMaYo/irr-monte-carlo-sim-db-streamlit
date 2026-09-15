"""Supabase persistence for IRR input presets only.

Expected Streamlit secrets (.streamlit/secrets.toml):

[connections.supabase]
SUPABASE_URL = "https://<project-ref>.supabase.co"
SUPABASE_PUBLISHABLE_KEY = "sb_publishable_..."

The Supabase table is expected to be named ``input_presets`` with columns:
id, name, description, parameters, created_at, updated_at.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import numpy as np
import streamlit as st
from supabase import Client, create_client

TABLE_NAME = "input_presets"


def _connection_settings() -> tuple[str, str]:
    """Read the Supabase URL and API key from Streamlit secrets."""
    try:
        settings = st.secrets["connections"]["supabase"]
    except KeyError as exc:
        raise RuntimeError(
            "Supabase configuration not found. Add [connections.supabase] "
            "to .streamlit/secrets.toml."
        ) from exc

    url = str(settings.get("SUPABASE_URL", "")).strip()
    key = str(
        settings.get("SUPABASE_PUBLISHABLE_KEY")
        or settings.get("SUPABASE_KEY")
        or ""
    ).strip()

    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY are required in "
            ".streamlit/secrets.toml."
        )

    return url, key


@st.cache_resource
def get_supabase_client() -> Client:
    """Create and cache one Supabase client per Streamlit process."""
    url, key = _connection_settings()
    return create_client(url, key)


def _json_safe(value: Any) -> Any:
    """Convert application inputs to values accepted by a JSONB column."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        f"Input value of type {type(value).__name__} is not JSON serializable."
    )


def list_presets() -> list[dict[str, Any]]:
    """Return saved presets ordered by name."""
    response = (
        get_supabase_client()
        .table(TABLE_NAME)
        .select("id,name,description,created_at,updated_at")
        .order("name")
        .execute()
    )
    return list(response.data or [])


def get_preset(preset_id: int) -> dict[str, Any]:
    """Return the complete parameter dictionary for one preset."""
    response = (
        get_supabase_client()
        .table(TABLE_NAME)
        .select("id,name,description,parameters,created_at,updated_at")
        .eq("id", int(preset_id))
        .limit(1)
        .execute()
    )
    rows = response.data or []
    if not rows:
        raise ValueError("Preset not found.")
    return dict(rows[0])


def save_preset(
    name: str,
    parameters: dict[str, Any],
    description: str = "",
) -> dict[str, Any]:
    """Create a new named input preset."""
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("Preset name cannot be empty.")

    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "name": clean_name,
        "description": description.strip(),
        "parameters": _json_safe(parameters),
        "created_at": now,
        "updated_at": now,
    }
    response = (
        get_supabase_client()
        .table(TABLE_NAME)
        .insert(payload)
        .execute()
    )
    rows = response.data or []
    return dict(rows[0]) if rows else payload


def update_preset(
    preset_id: int,
    name: str,
    parameters: dict[str, Any],
    description: str = "",
) -> dict[str, Any]:
    """Replace the name, description, and inputs of an existing preset."""
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("Preset name cannot be empty.")

    payload = {
        "name": clean_name,
        "description": description.strip(),
        "parameters": _json_safe(parameters),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    response = (
        get_supabase_client()
        .table(TABLE_NAME)
        .update(payload)
        .eq("id", int(preset_id))
        .execute()
    )
    rows = response.data or []
    if not rows:
        raise ValueError("Preset not found or update not permitted.")
    return dict(rows[0])


def delete_preset(preset_id: int) -> None:
    """Delete a saved preset."""
    response = (
        get_supabase_client()
        .table(TABLE_NAME)
        .delete()
        .eq("id", int(preset_id))
        .execute()
    )
    if not (response.data or []):
        raise ValueError("Preset not found or deletion not permitted.")


def test_connection() -> bool:
    """Check that the table can be read without changing any data."""
    get_supabase_client().table(TABLE_NAME).select("id").limit(1).execute()
    return True
