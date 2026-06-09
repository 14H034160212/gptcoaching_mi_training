"""
Magic-link email auth with bearer-token sessions.

Storage layout under AUTH_DIR (default runs/auth/):
  users.json         email -> {email, user_id, created_at}
  magic_tokens.json  token -> {email, expires_at, used}
  sessions.json      token -> {user_id, expires_at}

Sessions are kept as bearer tokens so the frontend on *.pages.dev can
authenticate to the backend on a different domain without third-party
cookies (which browsers increasingly block).
"""
from __future__ import annotations

import json
import secrets
import time
import urllib.error
import urllib.request
from pathlib import Path
from threading import Lock
from typing import Optional

MAGIC_LINK_TTL_SECONDS = 15 * 60
SESSION_TTL_SECONDS = 30 * 24 * 3600


class AuthStore:
    def __init__(self, storage_dir: str = "runs/auth"):
        self.dir = Path(storage_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.users_path = self.dir / "users.json"
        self.magic_path = self.dir / "magic_tokens.json"
        self.sessions_path = self.dir / "sessions.json"
        self._lock = Lock()
        self._users = self._load(self.users_path)
        self._magic = self._load(self.magic_path)
        self._sessions = self._load(self.sessions_path)

    @staticmethod
    def _load(path: Path) -> dict:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        return {}

    @staticmethod
    def _save(path: Path, data: dict) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    def _ensure_user(self, email: str) -> dict:
        email = email.lower().strip()
        if email not in self._users:
            self._users[email] = {
                "email": email,
                "user_id": "u_" + secrets.token_urlsafe(12),
                "created_at": time.time(),
            }
            self._save(self.users_path, self._users)
        return self._users[email]

    def issue_magic_token(self, email: str) -> str:
        with self._lock:
            self._ensure_user(email)
            token = secrets.token_urlsafe(32)
            self._magic[token] = {
                "email": email.lower().strip(),
                "expires_at": time.time() + MAGIC_LINK_TTL_SECONDS,
                "used": False,
            }
            self._save(self.magic_path, self._magic)
            return token

    def consume_magic_token(self, token: str) -> Optional[dict]:
        with self._lock:
            entry = self._magic.get(token)
            if not entry or entry["used"] or entry["expires_at"] < time.time():
                return None
            entry["used"] = True
            self._magic[token] = entry
            self._save(self.magic_path, self._magic)
            return self._users.get(entry["email"])

    def issue_session(self, user_id: str) -> str:
        with self._lock:
            token = secrets.token_urlsafe(32)
            self._sessions[token] = {
                "user_id": user_id,
                "expires_at": time.time() + SESSION_TTL_SECONDS,
            }
            self._save(self.sessions_path, self._sessions)
            return token

    def get_session_user(self, session_token: str) -> Optional[dict]:
        with self._lock:
            entry = self._sessions.get(session_token)
            if not entry:
                return None
            if entry["expires_at"] < time.time():
                self._sessions.pop(session_token, None)
                self._save(self.sessions_path, self._sessions)
                return None
            for user in self._users.values():
                if user["user_id"] == entry["user_id"]:
                    return user
            return None

    def revoke_session(self, session_token: str) -> None:
        with self._lock:
            if self._sessions.pop(session_token, None) is not None:
                self._save(self.sessions_path, self._sessions)


def send_magic_link_email(
    api_key: str,
    from_addr: str,
    to_email: str,
    link: str,
) -> None:
    payload = json.dumps({
        "from": from_addr,
        "to": to_email,
        "subject": "Sign in to Kerrio.AI",
        "html": (
            "<p>Click the link below to sign in. The link expires in 15 minutes "
            "and can only be used once.</p>"
            f'<p><a href="{link}">{link}</a></p>'
            "<p>If you did not request this email you can ignore it.</p>"
        ),
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.resend.com/emails",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "kerrio-ai/1.0 (+https://gptcoaching-mi-training.pages.dev)",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        raise RuntimeError(f"Resend API {e.code}: {body}") from e
