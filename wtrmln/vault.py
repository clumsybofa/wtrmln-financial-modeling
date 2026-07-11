"""Encrypted-at-rest storage for credentials the agent provisions during setup.

Credentials the *user* types during the human-login handoff never reach this
module or the model — this vault only holds artifacts the agent itself creates
inside the provider's UI after login (API keys, OAuth client ids/secrets,
report export URLs).
"""

import json
import os
from pathlib import Path

from cryptography.fernet import Fernet

VAULT_DIR = Path(__file__).resolve().parent.parent / "data"
KEY_PATH = VAULT_DIR / "vault.key"
STORE_PATH = VAULT_DIR / "vault.enc"


def _fernet() -> Fernet:
    key = os.environ.get("WTRMLN_VAULT_KEY")
    if key:
        return Fernet(key.encode())
    VAULT_DIR.mkdir(parents=True, exist_ok=True)
    if not KEY_PATH.exists():
        KEY_PATH.write_bytes(Fernet.generate_key())
        KEY_PATH.chmod(0o600)
    return Fernet(KEY_PATH.read_bytes())


def _load() -> dict:
    if not STORE_PATH.exists():
        return {}
    return json.loads(_fernet().decrypt(STORE_PATH.read_bytes()))


def _save(store: dict):
    VAULT_DIR.mkdir(parents=True, exist_ok=True)
    STORE_PATH.write_bytes(_fernet().encrypt(json.dumps(store).encode()))
    STORE_PATH.chmod(0o600)


def store_credential(connection_id: str, name: str, value: str):
    store = _load()
    store.setdefault(connection_id, {})[name] = value
    _save(store)


def list_credential_names(connection_id: str) -> list[str]:
    return sorted(_load().get(connection_id, {}).keys())


def get_credential(connection_id: str, name: str) -> str | None:
    return _load().get(connection_id, {}).get(name)
