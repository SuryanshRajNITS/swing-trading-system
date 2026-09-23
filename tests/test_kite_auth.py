from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.broker.auth import AuthenticationError, KiteCredentials, authenticate, load_credentials
from src.broker.secure_store import AccessTokenStore, TokenStoreError


def test_missing_credentials_fail_without_exposing_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KITE_API_KEY", raising=False)
    monkeypatch.delenv("KITE_API_SECRET", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")

    with pytest.raises(AuthenticationError, match="KITE_API_KEY.*KITE_API_SECRET"):
        load_credentials(env_file)


def test_authentication_exchanges_and_stores_token_without_printing_it(tmp_path: Path) -> None:
    class FakeKite:
        def __init__(self, *, api_key: str) -> None:
            assert api_key == "test-key"

        def login_url(self) -> str:
            return "https://kite.example/login"

        def generate_session(self, request_token: str, *, api_secret: str) -> dict[str, str]:
            assert request_token == "private-request-token"
            assert api_secret == "private-api-secret"
            return {"access_token": "private-access-token"}

    output: list[str] = []
    store = AccessTokenStore(
        tmp_path / "token.json",
        protector=lambda value: b"encrypted:" + value,
        unprotector=lambda value: value.removeprefix(b"encrypted:"),
    )
    result = authenticate(
        KiteCredentials("test-key", "private-api-secret"),
        token_store=store,
        kite_factory=FakeKite,  # type: ignore[arg-type]
        request_token_provider=lambda login_url: "private-request-token",
        output=output.append,
    )

    assert result == "private-access-token"
    assert "private-access-token" not in " ".join(output)
    assert "private-request-token" not in " ".join(output)
    document = json.loads(store.path.read_text(encoding="utf-8"))
    assert document["date"] == store._today().isoformat()
    assert "private-access-token" not in store.path.read_text(encoding="utf-8")


def test_same_day_token_is_reused_without_new_kite_session(tmp_path: Path) -> None:
    store = AccessTokenStore(
        tmp_path / "token.json",
        protector=lambda value: b"encrypted:" + value,
        unprotector=lambda value: value.removeprefix(b"encrypted:"),
    )
    store.save("private-access-token")
    output: list[str] = []

    result = authenticate(
        KiteCredentials("test-key", "private-api-secret"),
        token_store=store,
        kite_factory=lambda **kwargs: pytest.fail("Kite client should not be created when today's token exists"),
        output=output.append,
    )

    assert result == "private-access-token"
    assert "private-access-token" not in " ".join(output)


def test_token_exchange_failure_does_not_expose_tokens_or_secret(tmp_path: Path) -> None:
    class FailingKite:
        def __init__(self, *, api_key: str) -> None:
            pass

        def login_url(self) -> str:
            return "https://kite.example/login"

        def generate_session(self, request_token: str, *, api_secret: str) -> dict[str, str]:
            raise RuntimeError("private-request-token private-api-secret")

    output: list[str] = []
    store = AccessTokenStore(
        tmp_path / "token.json",
        protector=lambda value: b"encrypted:" + value,
        unprotector=lambda value: value.removeprefix(b"encrypted:"),
    )

    with pytest.raises(AuthenticationError, match="Kite token exchange failed") as error:
        authenticate(
            KiteCredentials("test-key", "private-api-secret"),
            token_store=store,
            kite_factory=FailingKite,  # type: ignore[arg-type]
            request_token_provider=lambda login_url: "private-request-token",
            output=output.append,
        )

    safe_messages = str(error.value) + " " + " ".join(output)
    assert "private-request-token" not in safe_messages
    assert "private-api-secret" not in safe_messages


def test_token_storage_failure_reports_safe_reason(tmp_path: Path) -> None:
    class FakeKite:
        def __init__(self, *, api_key: str) -> None:
            pass

        def login_url(self) -> str:
            return "https://kite.example/login"

        def generate_session(self, request_token: str, *, api_secret: str) -> dict[str, str]:
            return {"access_token": "private-access-token"}

    calls = 0

    def fail_second_encryption(value: bytes) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise TokenStoreError("Windows DPAPI could not encrypt the access token (Windows error 2).")
        return b"encrypted:" + value

    output: list[str] = []
    store = AccessTokenStore(
        tmp_path / "token.json",
        protector=fail_second_encryption,
        unprotector=lambda value: value.removeprefix(b"encrypted:"),
    )
    with pytest.raises(AuthenticationError, match="DPAPI.*error 2") as error:
        authenticate(
            KiteCredentials("test-key", "private-api-secret"),
            token_store=store,
            kite_factory=FakeKite,  # type: ignore[arg-type]
            request_token_provider=lambda login_url: "private-request-token",
            output=output.append,
        )

    assert "private-access-token" not in str(error.value)
    assert "private-request-token" not in str(error.value)
    assert "private-api-secret" not in str(error.value)


def test_secure_storage_is_checked_before_opening_kite_login(tmp_path: Path) -> None:
    def fail_to_encrypt(_: bytes) -> bytes:
        raise TokenStoreError("Windows DPAPI could not encrypt the access token (Windows error 2).")

    store = AccessTokenStore(
        tmp_path / "token.json",
        protector=fail_to_encrypt,
    )

    with pytest.raises(AuthenticationError, match="Local secure token storage.*error 2"):
        authenticate(
            KiteCredentials("test-key", "private-api-secret"),
            token_store=store,
            kite_factory=lambda **kwargs: pytest.fail("Kite must not be contacted before storage passes"),
            request_token_provider=lambda _: pytest.fail("Login URL must not be opened before storage passes"),
            output=lambda _: None,
        )


def test_token_from_previous_india_day_is_not_reused(tmp_path: Path) -> None:
    store = AccessTokenStore(tmp_path / "token.json", protector=lambda value: b"encrypted:" + value)
    store.save("private-access-token")
    document = json.loads(store.path.read_text(encoding="utf-8"))
    document["date"] = "2000-01-01"
    store.path.write_text(json.dumps(document), encoding="utf-8")

    assert store.load_today() is None
