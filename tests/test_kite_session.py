from __future__ import annotations

from pathlib import Path

import pytest
from kiteconnect.exceptions import TokenException

from src.broker.secure_store import AccessTokenStore
from src.broker.session import SessionVerificationError, verify_session


class FakeKite:
    def __init__(self, *, api_key: str) -> None:
        self.api_key = api_key
        self.access_token: str | None = None
        self.profile_calls = 0
        self.response: object = {"user_id": "private-user", "user_name": "Private Name"}

    def set_access_token(self, access_token: str) -> None:
        self.access_token = access_token

    def profile(self) -> object:
        self.profile_calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def make_store(path: Path) -> AccessTokenStore:
    return AccessTokenStore(
        path,
        protector=lambda value: b"encrypted:" + value,
        unprotector=lambda value: value.removeprefix(b"encrypted:"),
    )


def test_verify_session_calls_profile_without_exposing_profile_data(tmp_path: Path) -> None:
    store = make_store(tmp_path / "token.json")
    store.save("private-access-token")
    clients: list[FakeKite] = []

    def kite_factory(*, api_key: str) -> FakeKite:
        client = FakeKite(api_key=api_key)
        clients.append(client)
        return client

    verify_session(api_key="test-api-key", token_store=store, kite_factory=kite_factory)  # type: ignore[arg-type]

    assert len(clients) == 1
    assert clients[0].access_token == "private-access-token"
    assert clients[0].profile_calls == 1


def test_verify_session_requires_same_day_token(tmp_path: Path) -> None:
    with pytest.raises(SessionVerificationError, match="No saved Kite access token for today"):
        verify_session(
            api_key="test-api-key",
            token_store=make_store(tmp_path / "missing.json"),
            kite_factory=lambda **kwargs: pytest.fail("Kite should not be contacted without a token"),
        )


def test_profile_auth_error_is_sanitized(tmp_path: Path) -> None:
    store = make_store(tmp_path / "token.json")
    store.save("private-access-token")
    client = FakeKite(api_key="test-api-key")
    client.response = TokenException("contains private-access-token")

    with pytest.raises(SessionVerificationError, match="Kite rejected the saved session") as error:
        verify_session(
            api_key="test-api-key",
            token_store=store,
            kite_factory=lambda **kwargs: client,
        )

    assert "private-access-token" not in str(error.value)


def test_profile_response_is_never_returned_or_printed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    store = make_store(tmp_path / "token.json")
    store.save("private-access-token")
    client = FakeKite(api_key="test-api-key")
    verify_session(api_key="test-api-key", token_store=store, kite_factory=lambda **kwargs: client)

    assert "Private Name" not in capsys.readouterr().out
