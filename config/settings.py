from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM Ollama ───────────────────────────────────────────────────────────
    ollama_base_url: str = Field("http://localhost:11434", alias="OLLAMA_BASE_URL")
    claimshield_llm_model: str = Field("gemma4:latest", alias="CLAIMSHIELD_LLM_MODEL")
    claimshield_llm_provider: str = Field("ollama", alias="CLAIMSHIELD_LLM_PROVIDER")

    # ── Application ───────────────────────────────────────────────────────────
    claimshield_env: str = Field("development", alias="CLAIMSHIELD_ENV")
    claimshield_debug: bool = Field(True, alias="CLAIMSHIELD_DEBUG")
    claimshield_log_level: str = Field("INFO", alias="CLAIMSHIELD_LOG_LEVEL")
    claimshield_secret_key: str = Field("changez-en-production", alias="CLAIMSHIELD_SECRET_KEY")

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = Field("127.0.0.1", alias="API_HOST")
    api_port: int = Field(8000, alias="API_PORT")
    api_reload: bool = Field(True, alias="API_RELOAD")

    # ── Synthea ───────────────────────────────────────────────────────────────
    synthea_root: Path = Field(_PROJECT_ROOT / "synthea", alias="SYNTHEA_ROOT")
    synthea_output_dir: Path = Field(
        _PROJECT_ROOT / "synthea" / "output_claimshield",
        alias="SYNTHEA_OUTPUT_DIR",
    )
    claimshield_source_root: Path = Field(
        _PROJECT_ROOT / "synthea" / "claimshield_cases",
        alias="CLAIMSHIELD_SOURCE_ROOT",
    )

    # ── Stockage ──────────────────────────────────────────────────────────────
    claimshield_datasets_dir: Path = Field(
        _PROJECT_ROOT / "datasets", alias="CLAIMSHIELD_DATASETS_DIR"
    )
    claimshield_storage_dir: Path = Field(
        _PROJECT_ROOT / "storage", alias="CLAIMSHIELD_STORAGE_DIR"
    )
    claimshield_inbox_dir: Path = Field(
        _PROJECT_ROOT / "storage" / "inbox", alias="CLAIMSHIELD_INBOX_DIR"
    )
    claimshield_quarantine_dir: Path = Field(
        _PROJECT_ROOT / "storage" / "quarantine", alias="CLAIMSHIELD_QUARANTINE_DIR"
    )
    claimshield_processed_dir: Path = Field(
        _PROJECT_ROOT / "storage" / "processed", alias="CLAIMSHIELD_PROCESSED_DIR"
    )
    claimshield_rejected_dir: Path = Field(
        _PROJECT_ROOT / "storage" / "rejected", alias="CLAIMSHIELD_REJECTED_DIR"
    )
    claimshield_temp_dir: Path = Field(
        _PROJECT_ROOT / "storage" / "temp", alias="CLAIMSHIELD_TEMP_DIR"
    )

    # ── Base de données ───────────────────────────────────────────────────────
    database_url: str = Field(
        "sqlite+aiosqlite:///./storage/claimshield.db", alias="DATABASE_URL"
    )

    # ── LangGraph checkpoints ─────────────────────────────────────────────────
    langgraph_checkpoint_db: Path = Field(
        _PROJECT_ROOT / "storage" / "checkpoints.db",
        alias="LANGGRAPH_CHECKPOINT_DB",
    )

    # ── Pseudonymisation ─────────────────────────────────────────────────────
    pseudonymization_key: SecretStr = Field(
        default=SecretStr("claimshield-dev-pseudonymization-key-change-in-production"),
        alias="PSEUDONYMIZATION_KEY",
        description=(
            "Clé secrète HMAC-SHA256 pour la pseudonymisation des identifiants patients. "
            "Ne jamais écrire la valeur réelle dans le code ni dans les logs. "
            "Valeur par défaut = clé de développement uniquement."
        ),
    )

    # ── Sécurité fichiers ─────────────────────────────────────────────────────
    claimshield_max_file_size_mb: int = Field(20, alias="CLAIMSHIELD_MAX_FILE_SIZE_MB")
    claimshield_max_folder_size_mb: int = Field(200, alias="CLAIMSHIELD_MAX_FOLDER_SIZE_MB")
    claimshield_max_files_per_folder: int = Field(50, alias="CLAIMSHIELD_MAX_FILES_PER_FOLDER")
    claimshield_allowed_extensions: str = Field(
        "pdf,png,jpeg,jpg,json", alias="CLAIMSHIELD_ALLOWED_EXTENSIONS"
    )
    claimshield_allowed_mime_types: str = Field(
        "application/pdf,image/png,image/jpeg,application/json",
        alias="CLAIMSHIELD_ALLOWED_MIME_TYPES",
    )

    @field_validator("claimshield_max_file_size_mb", "claimshield_max_folder_size_mb", "claimshield_max_files_per_folder")
    @classmethod
    def _doit_etre_positif(cls, v: int, info) -> int:
        if v <= 0:
            raise ValueError(f"{info.field_name} doit être strictement positif, reçu : {v}")
        return v

    # ── Audit ─────────────────────────────────────────────────────────────────
    claimshield_audit_dir: Path = Field(
        _PROJECT_ROOT / "logs" / "audit", alias="CLAIMSHIELD_AUDIT_DIR"
    )

    # ── Raccourcis calculés (compat. avec l'ancienne API) ────────────────────
    @computed_field  # type: ignore[prop-decorator]
    @property
    def project_root(self) -> Path:
        return _PROJECT_ROOT

    @computed_field  # type: ignore[prop-decorator]
    @property
    def datasets_dir(self) -> Path:
        return self.claimshield_datasets_dir

    @computed_field  # type: ignore[prop-decorator]
    @property
    def storage_dir(self) -> Path:
        return self.claimshield_storage_dir

    @computed_field  # type: ignore[prop-decorator]
    @property
    def inbox_dir(self) -> Path:
        return self.claimshield_inbox_dir

    @computed_field  # type: ignore[prop-decorator]
    @property
    def quarantine_dir(self) -> Path:
        return self.claimshield_quarantine_dir

    @computed_field  # type: ignore[prop-decorator]
    @property
    def processed_dir(self) -> Path:
        return self.claimshield_processed_dir

    @computed_field  # type: ignore[prop-decorator]
    @property
    def rejected_dir(self) -> Path:
        return self.claimshield_rejected_dir

    @computed_field  # type: ignore[prop-decorator]
    @property
    def temp_dir(self) -> Path:
        return self.claimshield_temp_dir

    @computed_field  # type: ignore[prop-decorator]
    @property
    def max_file_size_bytes(self) -> int:
        return self.claimshield_max_file_size_mb * 1024 * 1024

    @computed_field  # type: ignore[prop-decorator]
    @property
    def max_folder_size_bytes(self) -> int:
        return self.claimshield_max_folder_size_mb * 1024 * 1024

    @computed_field  # type: ignore[prop-decorator]
    @property
    def max_files_per_folder(self) -> int:
        return self.claimshield_max_files_per_folder

    @computed_field  # type: ignore[prop-decorator]
    @property
    def allowed_extensions(self) -> list[str]:
        return [e.strip().lower() for e in self.claimshield_allowed_extensions.split(",")]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def allowed_mime_types(self) -> list[str]:
        return [m.strip().lower() for m in self.claimshield_allowed_mime_types.split(",")]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
