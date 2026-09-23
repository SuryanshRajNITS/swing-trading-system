"""Manual Kite Connect login and localhost callback handling."""

from __future__ import annotations

import logging
import os
import time
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlsplit

from dotenv import load_dotenv
from kiteconnect import KiteConnect

from src.configuration import PROJECT_ROOT
from src.broker.secure_store import AccessTokenStore, TokenStoreError


REDIRECT_URL = "http://127.0.0.1:8000/callback"
CALLBACK_HOST = "127.0.0.1"
CALLBACK_PORT = 8000
CALLBACK_PATH = "/callback"
CALLBACK_TIMEOUT_SECONDS = 600


class AuthenticationError(RuntimeError):
    """Raised for a safe, user-facing authentication failure."""


@dataclass(frozen=True)
class KiteCredentials:
    api_key: str = field(repr=False)
    api_secret: str = field(repr=False)


def load_credentials(env_path: str | Path = PROJECT_ROOT / ".env") -> KiteCredentials:
    """Read required Kite credentials from the environment or local .env file."""
    load_dotenv(dotenv_path=Path(env_path), override=False)
    api_key = os.getenv("KITE_API_KEY", "").strip()
    api_secret = os.getenv("KITE_API_SECRET", "").strip()
    missing = [name for name, value in (("KITE_API_KEY", api_key), ("KITE_API_SECRET", api_secret)) if not value]
    if missing:
        raise AuthenticationError(f"Missing required Kite credential(s): {', '.join(missing)}. Set them in .env.")
    return KiteCredentials(api_key=api_key, api_secret=api_secret)


def load_api_key(env_path: str | Path = PROJECT_ROOT / ".env") -> str:
    """Load only the API key for authenticated read-only requests."""
    load_dotenv(dotenv_path=Path(env_path), override=False)
    api_key = os.getenv("KITE_API_KEY", "").strip()
    if not api_key:
        raise AuthenticationError("Missing required Kite credential: KITE_API_KEY. Set it in .env.")
    return api_key


def _capture_request_token(
    login_url: str,
    *,
    open_browser: Callable[[str], bool] = webbrowser.open,
    output: Callable[[str], object] = print,
    timeout_seconds: int = CALLBACK_TIMEOUT_SECONDS,
) -> str:
    """Display/open the login URL and wait for the registered localhost callback."""
    class CallbackHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            # The callback query contains the short-lived request_token.
            return

        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            if parsed.path != CALLBACK_PATH:
                self.send_error(404)
                return

            query = parse_qs(parsed.query)
            if query.get("status", [""])[0] != "success":
                self.server.auth_failure = True  # type: ignore[attr-defined]
                self.server.auth_complete = True  # type: ignore[attr-defined]
                body = b"Kite login was not successful. You can close this tab."
            else:
                tokens = query.get("request_token", [])
                if not tokens or not tokens[0]:
                    self.server.auth_failure = True  # type: ignore[attr-defined]
                    self.server.auth_complete = True  # type: ignore[attr-defined]
                    body = b"Kite callback did not include a request token. You can close this tab."
                else:
                    self.server.request_token = tokens[0]  # type: ignore[attr-defined]
                    self.server.auth_complete = True  # type: ignore[attr-defined]
                    body = b"Login callback received. Return to the application. You can close this tab."

            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    try:
        server = HTTPServer((CALLBACK_HOST, CALLBACK_PORT), CallbackHandler)
    except OSError:
        raise AuthenticationError(
            f"Could not listen on {CALLBACK_HOST}:{CALLBACK_PORT}. Close any application using that address and retry."
        ) from None

    output("Open this Kite login URL and complete login/2FA manually:")
    output(login_url)
    try:
        open_browser(login_url)
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            server.timeout = min(1.0, max(0.0, deadline - time.monotonic()))
            server.handle_request()
            if getattr(server, "auth_complete", False):
                break
        if not getattr(server, "auth_complete", False):
            raise AuthenticationError("Timed out waiting for the Kite localhost callback.")
        if getattr(server, "auth_failure", False):
            raise AuthenticationError("Kite login was cancelled or did not return a valid request token.")
        return server.request_token  # type: ignore[attr-defined]
    finally:
        server.server_close()


def authenticate(
    credentials: KiteCredentials | None = None,
    *,
    token_store: AccessTokenStore | None = None,
    kite_factory: Callable[..., KiteConnect] = KiteConnect,
    request_token_provider: Callable[[str], str] = _capture_request_token,
    output: Callable[[str], object] = print,
) -> str:
    """Reuse today's token or perform the manual browser callback flow."""
    credentials = credentials or load_credentials()
    token_store = token_store or AccessTokenStore()
    try:
        existing_token = token_store.load_today()
    except TokenStoreError as error:
        raise AuthenticationError(f"The local access token could not be read securely: {error}") from None
    if existing_token:
        output("A same-day Kite access token is already available locally.")
        return existing_token

    try:
        token_store.verify_encryption()
    except TokenStoreError as error:
        raise AuthenticationError(f"Local secure token storage is unavailable: {error}") from None

    kite = kite_factory(api_key=credentials.api_key)
    request_token = request_token_provider(kite.login_url())
    if not request_token:
        raise AuthenticationError("The Kite callback did not provide a request token.")
    try:
        session_data = kite.generate_session(request_token, api_secret=credentials.api_secret)
        access_token = session_data.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise AuthenticationError("Kite authentication response did not include an access token.")
        try:
            token_store.save(access_token)
        except TokenStoreError as error:
            raise AuthenticationError(f"Kite login succeeded, but token storage failed: {error}") from None
    except AuthenticationError:
        raise
    except Exception:
        raise AuthenticationError("Kite token exchange failed. Verify the app credentials and retry login.") from None

    logging.getLogger(__name__).info("Kite authentication completed; access token stored securely for today.")
    output("Kite authentication succeeded. The access token was stored securely for today.")
    return access_token


def main() -> None:
    try:
        authenticate()
    except AuthenticationError as exc:
        print(f"Authentication failed: {exc}")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
