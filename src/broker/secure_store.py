"""Windows DPAPI-backed storage for a same-day Kite access token."""

from __future__ import annotations

import base64
import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

from src.configuration import PROJECT_ROOT


TOKEN_FILE = PROJECT_ROOT / "data" / ".secrets" / "kite_access_token.json"
INDIA_TIMEZONE = ZoneInfo("Asia/Kolkata")
_ENTROPY = b"swing-trading-system:kite-access-token:v1"


class TokenStoreError(RuntimeError):
    """Raised when a local access token cannot be safely stored or read."""


def _windows_error_code(error: Exception) -> int | None:
    """Extract only a numeric Windows error code, never exception text or data."""
    code = getattr(error, "winerror", None)
    if isinstance(code, int):
        return code
    args = getattr(error, "args", ())
    if args and isinstance(args[0], int):
        return args[0]
    return None


def _windows_protect(data: bytes) -> bytes:
    if os.name != "nt":
        raise TokenStoreError("Secure token storage is available only on Windows.")
    try:
        import win32crypt

        encrypted = win32crypt.CryptProtectData(
            data,
            "Kite access token",
            _ENTROPY,
            None,
            None,
            1,
        )
        return encrypted
    except Exception as error:
        code = _windows_error_code(error)
        suffix = f" (Windows error {code})" if code is not None else ""
        raise TokenStoreError(
            f"Windows DPAPI could not encrypt the access token{suffix}."
        ) from None


def _windows_unprotect(data: bytes) -> bytes:
    if os.name != "nt":
        raise TokenStoreError("Secure token storage is available only on Windows.")

    try:
        import win32crypt

        _, decrypted = win32crypt.CryptUnprotectData(
            data,
            _ENTROPY,
            None,
            None,
            1,
        )
        return decrypted
    except Exception as error:
        code = _windows_error_code(error)
        suffix = f" (Windows error {code})" if code is not None else ""
        raise TokenStoreError(
            f"Windows DPAPI could not decrypt the stored access token{suffix}."
        ) from None


class AccessTokenStore:
    """Persist the token encrypted for the current Windows user and trading day."""

    def __init__(
        self,
        path: Path = TOKEN_FILE,
        *,
        protector: Callable[[bytes], bytes] = _windows_protect,
        unprotector: Callable[[bytes], bytes] = _windows_unprotect,
    ) -> None:
        self.path = Path(path)
        self._protect = protector
        self._unprotect = unprotector

    @staticmethod
    def _today() -> date:
        return datetime.now(INDIA_TIMEZONE).date()

    def verify_encryption(self) -> None:
        """Verify DPAPI encryption works before the user spends time logging in."""
        probe = b"swing-trading-system-storage-check"
        encrypted = self._protect(probe)
        if self._unprotect(encrypted) != probe:
            raise TokenStoreError("Windows DPAPI encryption check did not round-trip correctly.")

    def save(self, access_token: str) -> None:
        if not access_token:
            raise TokenStoreError("Refusing to store an empty access token.")
        encrypted = self._protect(access_token.encode("utf-8"))
        document = {
            "date": self._today().isoformat(),
            "encrypted_access_token": base64.b64encode(encrypted).decode("ascii"),
        }
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path.write_text(json.dumps(document), encoding="utf-8")
            temporary_path.replace(self.path)
        except OSError as error:
            code = error.winerror if isinstance(getattr(error, "winerror", None), int) else error.errno
            suffix = f" (Windows error {code})" if code is not None else ""
            raise TokenStoreError(f"The encrypted access token could not be written to local storage{suffix}.") from None
        finally:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass

    def load_today(self) -> str | None:
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
            if document.get("date") != self._today().isoformat():
                return None
            encrypted = base64.b64decode(document["encrypted_access_token"], validate=True)
            token = self._unprotect(encrypted).decode("utf-8")
        except FileNotFoundError:
            return None
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, UnicodeDecodeError):
            raise TokenStoreError("The local access-token file is invalid or unreadable.") from None
        except Exception:
            raise TokenStoreError("The local access token could not be decrypted for this Windows user.") from None
        if not token:
            raise TokenStoreError("The local access-token file contains no usable token.")
        return token
