import os

import pytest

from wtrmln.agent import CREDENTIAL_NAME_RE, load_playbooks
from wtrmln.browser import _map_key


REQUIRED_META = ("name", "slug", "icon", "login_url", "description", "allowed_domains")


def test_all_playbooks_parse_with_required_fields():
    books = load_playbooks()
    assert len(books) >= 5
    for slug, meta in books.items():
        for field in REQUIRED_META:
            assert meta.get(field), f"{slug} missing {field}"
        assert meta["login_url"].startswith("https://")
        assert meta["body"].strip()
        # login host must be inside the allowlist, or the first navigation
        # would be blocked by the browser's own gate
        from urllib.parse import urlparse
        host = urlparse(meta["login_url"]).hostname
        allowed = [d.strip() for d in meta["allowed_domains"].split(",")]
        assert any(host == d or host.endswith("." + d) for d in allowed), \
            f"{slug}: login host {host} not covered by allowed_domains"


def test_credential_name_validation():
    assert CREDENTIAL_NAME_RE.match("xero_client_secret")
    assert not CREDENTIAL_NAME_RE.match("Bad Name")
    assert not CREDENTIAL_NAME_RE.match("")
    assert not CREDENTIAL_NAME_RE.match("x" * 65)
    assert not CREDENTIAL_NAME_RE.match("../etc/passwd")


def test_key_mapping():
    assert _map_key("Return") == "Enter"
    assert _map_key("ctrl+s") == "Control+s"
    assert _map_key("Page_Down") == "PageDown"
    assert _map_key("super") == "Meta"
    assert _map_key("a") == "a"


def test_vault_roundtrip(tmp_path, monkeypatch):
    from cryptography.fernet import Fernet
    monkeypatch.setenv("WTRMLN_VAULT_KEY", Fernet.generate_key().decode())
    import wtrmln.vault as vault
    monkeypatch.setattr(vault, "STORE_PATH", tmp_path / "vault.enc")
    vault.store_credential("conn1", "api_key", "secret-value")
    assert vault.get_credential("conn1", "api_key") == "secret-value"
    assert vault.list_credential_names("conn1") == ["api_key"]
    # ciphertext on disk, not plaintext
    assert b"secret-value" not in (tmp_path / "vault.enc").read_bytes()


def test_vault_fails_closed_without_key(tmp_path, monkeypatch):
    monkeypatch.delenv("WTRMLN_VAULT_KEY", raising=False)
    monkeypatch.delenv("WTRMLN_DEV_MODE", raising=False)
    import wtrmln.vault as vault
    monkeypatch.setattr(vault, "STORE_PATH", tmp_path / "vault.enc")
    with pytest.raises(RuntimeError, match="WTRMLN_VAULT_KEY"):
        vault.store_credential("conn1", "k", "v")
