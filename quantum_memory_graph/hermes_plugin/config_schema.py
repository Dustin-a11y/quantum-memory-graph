"""QMG Memory Provider configuration schema."""

from plugins.memory.config_schema import (
    ProviderConfigSchema,
    ProviderField,
    KIND_TEXT,
    KIND_BOOL,
    KIND_NUMBER,
    STORAGE_FLAT_JSON,
)

QMG_CONFIG_SCHEMA = ProviderConfigSchema(
    name="qmg",
    label="QMG Memory Graph",
    storage=STORAGE_FLAT_JSON,
    fields=[
        ProviderField(
            key="qmg_url",
            label="QMG Server URL",
            kind=KIND_TEXT,
            description="Base URL of the agent's QMG FastAPI server (e.g. http://127.0.0.1:8514)",
            placeholder="http://127.0.0.1:8503",
        ),
        ProviderField(
            key="agent_name",
            label="Agent Name",
            kind=KIND_TEXT,
            description="Name of this agent (used as source prefix for stored memories)",
            placeholder="dk",
        ),
        ProviderField(
            key="enabled",
            label="Enabled",
            kind=KIND_BOOL,
            description="Enable automatic turn-level memory sync to QMG",
            default=True,
        ),
        ProviderField(
            key="store_critical",
            label="Store Critical Turns",
            kind=KIND_BOOL,
            description="Store turns classified as CRITICAL (security, credentials, corrections)",
            default=True,
        ),
        ProviderField(
            key="store_durable",
            label="Store Durable Turns",
            kind=KIND_BOOL,
            description="Store turns classified as DURABLE (facts, preferences, decisions)",
            default=True,
        ),
        ProviderField(
            key="store_ephemeral",
            label="Store Ephemeral Turns",
            kind=KIND_BOOL,
            description="Store turns classified as EPHEMERAL (process noise, filler). NOT recommended.",
            default=False,
        ),
        ProviderField(
            key="min_turn_length",
            label="Minimum Turn Length",
            kind=KIND_NUMBER,
            description="Minimum character length for a turn to be considered for storage",
            default=10,
        ),
        ProviderField(
            key="max_memory_length",
            label="Maximum Memory Length",
            kind=KIND_NUMBER,
            description="Maximum characters per stored memory",
            default=500,
        ),
    ],
)
