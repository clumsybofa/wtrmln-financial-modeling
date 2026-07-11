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
    # Auto-generated keys live next to the ciphertext, so a copy of the data
    # directory decrypts the vault. Acceptable only for local development —
    # hosted deployments must supply WTRMLN_VAULT_KEY from a secrets manager
    # (the server refuses to start otherwise; this check is the backstop).
    if os.environ.get("WTRMLN_DEV_MODE") != "1":
        raise RuntimeError(
            "WTRMLN_VAULT_KEY is not set. Generate one with "
            "python -c \"from cryptography.fernet import Fernet; "
            "print(Fernet.generate_key().decode())\" and store it in your "
            "secrets manager, or set WTRMLN_DEV_MODE=1 for local development."
        )
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
