"""Signing in, for the tool reached from outside the office (through Tailscale Funnel or another
tunnel). The office network needs none: see app.py, signed_in_only.

People and their passwords are in data/users.json, the passwords scrypt-hashed, never kept as
typed. Signing in gives the browser a cookie for 30 days, signed with data/signin.key; a new
password signs that person out everywhere. Five wrong passwords for a name lock it for 15
minutes, so a short password can't be guessed by trying them all.

    python -m finder.signin add <name>       asks for the password, adds or changes it
    python -m finder.signin remove <name>
    python -m finder.signin list
"""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sys
import threading
import time

from finder import store

USERS = store.DATA / "users.json"
KEY = store.DATA / "signin.key"
COOKIE = "signin"
DAYS = 30
MIN_LENGTH = 6
TRIES, LOCK_MINUTES = 5, 15

_lock = threading.Lock()
_failed: dict[str, list] = {}     # name -> [wrong tries, locked until]
_cache: dict = {"mtime": None, "users": {}}


def _hash(password: str, salt: bytes) -> str:
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2 ** 14, r=8, p=1).hex()


def users() -> dict:
    """{name in lower case: {"name", "salt", "hash"}}, read again whenever the file changes, so
    adding someone needs no restart."""
    try:
        mtime = USERS.stat().st_mtime
    except FileNotFoundError:
        return {}
    if mtime != _cache["mtime"]:
        _cache["users"], _cache["mtime"] = json.loads(USERS.read_text(encoding="utf-8")), mtime
    return _cache["users"]


def _key() -> bytes:
    if not KEY.exists():
        KEY.parent.mkdir(parents=True, exist_ok=True)
        KEY.write_bytes(secrets.token_bytes(32))
    return KEY.read_bytes()


def _sign(name: str, until: int, user: dict) -> str:
    # The password's hash is signed in too, so a new password ends the old cookies.
    return hmac.new(_key(), f"{name}|{until}|{user['hash']}".encode("utf-8"), "sha256").hexdigest()


def check(name: str, password: str) -> tuple[str | None, int]:
    """(the person's name as added, or None; minutes still locked, 0 if not locked)."""
    key = name.strip().lower()
    with _lock:
        tries, until = _failed.get(key, [0, 0])
        if until > time.time():
            return None, int((until - time.time()) // 60) + 1
    user = users().get(key)
    # A name nobody has still costs a hash, so the time taken does not tell which names exist.
    ok = hmac.compare_digest(_hash(password, bytes.fromhex(user["salt"]) if user else b"x" * 16),
                             user["hash"] if user else "")
    with _lock:
        if ok:
            _failed.pop(key, None)
            return user["name"], 0
        tries = _failed.get(key, [0, 0])[0] + 1
        _failed[key] = [0, time.time() + LOCK_MINUTES * 60] if tries >= TRIES else [tries, 0]
        if len(_failed) > 10_000:        # names made up by the thousand: forget the oldest
            _failed.pop(next(iter(_failed)))
        return None, LOCK_MINUTES if tries >= TRIES else 0


def cookie_for(name: str) -> str:
    until = int(time.time()) + DAYS * 86400
    return f"{name}|{until}|{_sign(name, until, users()[name.lower()])}"


def who(cookie: str | None) -> str | None:
    """The signed-in person's name, or None if the cookie is missing, expired, forged, or from
    before their password was changed."""
    try:
        name, until, mac = (cookie or "").rsplit("|", 2)
        user = users().get(name.lower())
        if user and int(until) > time.time() and hmac.compare_digest(mac, _sign(name, int(until), user)):
            return user["name"]
    except ValueError:
        pass
    return None


def add(name: str, password: str) -> None:
    if not name.strip() or "|" in name:
        raise ValueError("A name can't be empty or have | in it")
    if len(password) < MIN_LENGTH:
        raise ValueError(f"The password needs at least {MIN_LENGTH} characters (that one had {len(password)})")
    everyone = dict(users())
    salt = secrets.token_bytes(16)
    everyone[name.strip().lower()] = {"name": name.strip(), "salt": salt.hex(), "hash": _hash(password, salt)}
    _save(everyone)


def _save(everyone: dict) -> None:
    USERS.parent.mkdir(parents=True, exist_ok=True)
    USERS.write_text(json.dumps(everyone, indent=1), encoding="utf-8")
    _cache["mtime"] = None   # read it again even if the clock has not moved on


def remove(name: str) -> bool:
    everyone = dict(users())
    gone = everyone.pop(name.strip().lower(), None)
    _save(everyone)
    return gone is not None


if __name__ == "__main__":
    what, name = (sys.argv[1:] + ["", ""])[:2]
    if what == "add" and name:
        # Typed in plain sight, once: Python's hidden prompt loses keys in Windows PowerShell 5.1's
        # window (it stopped reading mid-password and PowerShell got the rest), and a password
        # seen as it is typed needs no second typing.
        password = input(f"Password for {name} (at least {MIN_LENGTH} characters; it shows as you type): ")
        try:
            add(name, password)
        except ValueError as e:
            sys.exit(f"{e}: nothing changed")
        print(f"{name} can sign in now (no restart needed). Signing in again is needed every {DAYS} days.\n"
              "Type cls to clear the password from this window.")
    elif what == "remove" and name:
        print(f"{name} removed" if remove(name) else f"Nobody called {name}")
    elif what == "list":
        print("\n".join(u["name"] for u in users().values()) or "Nobody yet")
    else:
        sys.exit(__doc__)
