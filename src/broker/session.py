"""Read-only Kite session verification."""

from __future__ import annotations

from typing import Callable

from kiteconnect import KiteConnect
from kiteconnect.exceptions import KiteException

from src.broker.auth import AuthenticationError, load_api_key
from src.broker.secure_store import AccessTokenStore, TokenStoreError


class SessionVerificationError(RuntimeError):
    """Raised when the saved Kite session cannot be verified safely."""


def create_authenticated_client(
    *,
    api_key: str | None = None,
    token_store: AccessTokenStore | None = None,
    kite_factory: Callable[..., KiteConnect] = KiteConnect,
) -> KiteConnect:
    """Create a client from today's saved token without exposing the token."""
    api_key = api_key or load_api_key()
    token_store = token_store or AccessTokenStore()
    try:
        access_token = token_store.load_today()
    except TokenStoreError as error:
        raise SessionVerificationError(f"Could not read today's saved Kite session securely: {error}") from None
    if not access_token:
        raise SessionVerificationError("No saved Kite access token for today. Run the Kite authentication command first.")

    kite = kite_factory(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


def verify_session(
    *,
    api_key: str | None = None,
    token_store: AccessTokenStore | None = None,
    kite_factory: Callable[..., KiteConnect] = KiteConnect,
) -> None:
    """Call Kite's read-only profile endpoint without exposing returned profile data."""
    kite = create_authenticated_client(api_key=api_key, token_store=token_store, kite_factory=kite_factory)
    try:
        profile = kite.profile()
    except KiteException:
        raise SessionVerificationError(
            "Kite rejected the saved session. Run the manual authentication flow again."
        ) from None
    except Exception:
        raise SessionVerificationError(
            "Could not reach Kite to verify the session. Check the network and retry."
        ) from None

    if not isinstance(profile, dict):
        raise SessionVerificationError("Kite returned an unexpected response while verifying the session.")


def main() -> None:
    try:
        verify_session()
    except (AuthenticationError, SessionVerificationError) as error:
        print(f"Kite session verification failed: {error}")
        raise SystemExit(1) from None
    print("Kite session verified. Profile details were not displayed.")


if __name__ == "__main__":
    main()
