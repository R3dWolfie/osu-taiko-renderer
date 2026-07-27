"""MAP RENDER LEADERBOARD — the per-map board on the osu!CATCH results screen.

VERBATIM PORT of the osu!STANDARD renderer's render/leaderboard.py (the same
render DB, the same best-per-player query, the same osu!-global hand-off and
Discord-avatar path) so catch reaches parity with std. The ONLY catch-specific
changes vs the std module: the avatar cache dir (r3d-catch), and count_katu is
carried on each entry so a flank card's "Miss" row can match catch's featured
results card, which counts miss = count_miss + count_katu (missed droplets).

The current play's results (catch's centred text stack) is flanked by compact
ranked cards of the OTHER renders of the SAME map, pulled from the R3D render DB
(LOCAL-ONLY — no osu!API, same rule as std's board).

This module is the DATA + AVATAR layer (pure); the card BAKING + compositing
live in lb_cards.py, reusing catch's own results font + palette so the flanking
cards match catch's featured text stack.

  * query_leaderboard()  best-per-player rows for a beatmap_md5 (one row per
                         player = their highest score), score DESC, fail-soft.
  * build_board()        splits the queried rows into left/right flanks
                         centred on the current play, computes its rank, and
                         the NEW-BEST / NEW-#1 rank moment. Adapts to sparse
                         maps (0/1/2/N others) — never assumes a full 8.
  * resolve_avatar()     Discord avatar (owner-chosen) via the bot token, with
                         an on-disk cache and a hard graceful fallback to the
                         procedural avatar chip. Never blocks/crashes a render.
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field

# best-per-player columns pulled for each flanking card. count_katu is catch's
# missed-droplet tally — pulled so a flank card's Miss row can match catch's
# featured card (miss = count_miss + count_katu); harmless/zero for other modes.
_LB_COLUMNS = (
    "player_name", "discord_user_id", "score", "accuracy", "grade",
    "max_combo", "mods_str", "mods", "count_300", "count_100", "count_50",
    "count_miss", "count_katu", "replay_md5",
)


@dataclass
class LeaderboardEntry:
    """One flanking card's data (a distinct player's best on this map)."""
    rank: int                       # absolute board rank (1-based)
    player_name: str
    score: int
    accuracy: float                 # display percent (0..100)
    grade: str
    max_combo: int
    mods_str: str
    mods: int
    counts: tuple[int, int, int, int]      # 300/100/50/miss
    count_katu: int = 0                     # catch missed-droplets (miss += this)
    discord_user_id: str | None = None
    # osu!-global path only: an absolute path to a PRE-FETCHED avatar PNG for
    # this player (the bot downloads osu avatars ahead of the render). None on
    # render-DB rows (they resolve a Discord avatar via discord_user_id). The
    # default keeps every existing LeaderboardEntry(...) construction valid.
    avatar_png: str | None = None


@dataclass
class BoardData:
    """The assembled board around the current (featured) play."""
    left: list[LeaderboardEntry] = field(default_factory=list)   # ranked ABOVE
    right: list[LeaderboardEntry] = field(default_factory=list)  # ranked BELOW
    rank: int = 1                   # the current play's rank on this map
    n_players: int = 1              # distinct players INCLUDING the current
    moment: str | None = None       # "NEW #1" | "NEW BEST" | None


# --- DB query ----------------------------------------------------------------------

def query_leaderboard(db_path, beatmap_md5: str,
                      exclude_replay_md5: str | None = None,
                      limit: int = 64) -> list[dict]:
    """Best-per-player renders of `beatmap_md5` from the R3D render DB
    (read-only) — ONE row per distinct player (their highest non-deleted
    score), ORDER BY score DESC. Local-only (owner decision, no osu!API).

    Uses SQLite's documented bare-column / MAX() behaviour: with a single
    MAX() aggregate and a GROUP BY, the other bare columns take their values
    from the max-score row — so each entry is a real render, not a mash-up.

    Fail-soft: missing/locked DB, schema drift or a blank md5 → [] (the board
    then degrades to the featured card alone). `exclude_replay_md5` drops the
    current replay's own row so it never appears twice."""
    if not beatmap_md5:
        return []
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0)
    except Exception:  # noqa: BLE001 — DB missing/locked → empty board
        return []
    try:
        cur = con.cursor()
        # MAX(score) drives SQLite's bare-column rule: the other columns then
        # take their values from each player's MAX-score row (a real render).
        # (A plain GROUP BY without the aggregate would return an ARBITRARY
        # row per player — NOT their best.)
        select_cols = ", ".join("MAX(score) AS score" if c == "score" else c
                                for c in _LB_COLUMNS)
        sql = (f"SELECT {select_cols} FROM renders "
               "WHERE beatmap_md5 = ? AND deleted = 0")
        params: list = [beatmap_md5]
        if exclude_replay_md5:
            sql += " AND replay_md5 != ?"
            params.append(exclude_replay_md5)
        sql += (" GROUP BY player_name COLLATE NOCASE "
                "ORDER BY score DESC LIMIT ?")
        params.append(int(limit))
        rows = cur.execute(sql, params).fetchall()
    except Exception:  # noqa: BLE001 — schema drift → empty board
        return []
    finally:
        con.close()
    return [dict(zip(_LB_COLUMNS, r)) for r in rows]


def query_player_discord_id(db_path, player_name: str,
                            beatmap_md5: str | None = None) -> str | None:
    """The CURRENT player's Discord user id from the R3D render DB (read-only)
    — so the FEATURED (centre) results card can show their real avatar via the
    SAME path the flanks use (resolve_avatar_bytes).

    The current render is driven by an .osr that carries only the player_name;
    the discord_user_id lives in the render DB, so this maps name → id. Prefers
    a real snowflake (a fetchable all-digit id) over the DB's `osu_<id>`
    placeholder (an osu! link with no Discord user), and — when `beatmap_md5`
    is given — an id seen on the SAME map first. Returns None when the player
    has no prior render / no linked id, or the DB is missing/locked/drifted:
    in every such case the featured card falls back to the procedural chip
    (e.g. a fresh render not yet written to the DB). Fail-soft, never raises."""
    if not player_name:
        return None
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0)
    except Exception:  # noqa: BLE001 — DB missing/locked → procedural fallback
        return None
    try:
        cur = con.cursor()
        sql = ("SELECT discord_user_id FROM renders "
               "WHERE player_name = ? COLLATE NOCASE AND deleted = 0 "
               "AND discord_user_id IS NOT NULL AND discord_user_id != ''")
        params: list = [player_name]
        if beatmap_md5:
            # rows on THIS map first, then the player's best score
            sql += " ORDER BY (beatmap_md5 = ?) DESC, score DESC"
            params.append(beatmap_md5)
        else:
            sql += " ORDER BY score DESC"
        sql += " LIMIT 32"
        cands = [str(r[0]) for r in cur.execute(sql, params).fetchall()]
    except Exception:  # noqa: BLE001 — schema drift → procedural fallback
        return None
    finally:
        con.close()
    # prefer a fetchable snowflake; else the first non-null (an osu_<id>
    # placeholder, which resolve_avatar_bytes turns into the procedural chip)
    for c in cands:
        if is_fetchable_id(c):
            return c
    return cands[0] if cands else None


def _entry_from_row(rank: int, row: dict) -> LeaderboardEntry:
    return LeaderboardEntry(
        rank=rank,
        player_name=str(row.get("player_name") or "?"),
        score=int(row.get("score") or 0),
        accuracy=float(row.get("accuracy") or 0.0),
        grade=str(row.get("grade") or "?"),
        max_combo=int(row.get("max_combo") or 0),
        mods_str=str(row.get("mods_str") or ""),
        mods=int(row.get("mods") or 0),
        counts=(int(row.get("count_300") or 0), int(row.get("count_100") or 0),
                int(row.get("count_50") or 0), int(row.get("count_miss") or 0)),
        count_katu=int(row.get("count_katu") or 0),
        discord_user_id=(str(row["discord_user_id"])
                         if row.get("discord_user_id") else None),
        avatar_png=row.get("avatar_png"),
    )


# --- board assembly (pure) ---------------------------------------------------------

# --- osu! GLOBAL leaderboard (bot hand-off) ----------------------------------------
# The service (cli/r3d_render.py) fetches the map's osu! global top scores via
# the osu!API, pre-fetches each player's osu avatar to a PNG, and writes a JSON
# list. rows_from_osu_json turns those entries into row dicts of the SAME shape
# query_leaderboard() returns, so build_board() consumes them UNCHANGED. Purely
# additive — the render-DB path is untouched and stays the default.

# osu! rank letters (XH/X = silver/gold SS, SH = silver S) -> the render-DB grade
# spellings (SS/S/A/B/C/D) so the results FOR_RANK colours + grade pill match.
_OSU_GRADE_NORM = {
    "XH": "SS", "X": "SS", "SSH": "SS", "SS": "SS",
    "SH": "S", "S": "S", "A": "A", "B": "B", "C": "C", "D": "D", "F": "D",
}


def _norm_osu_grade(g) -> str:
    s = str(g or "").strip().upper()
    return _OSU_GRADE_NORM.get(s, s or "?")


def _mods_str_to_csv(s) -> str:
    """Normalise a mods string to the render-DB comma form ("HD,DT") that the
    flank card splits on: pass comma'd input through, split a concatenated
    "HDDT" into 2-char acronyms, map empty/NM -> "" (no pills)."""
    s = str(s or "").strip()
    if not s or s.upper() in ("NM", "NOMOD"):
        return ""
    if "," in s:
        return ",".join(p.strip() for p in s.split(",") if p.strip())
    return ",".join(s[i:i + 2] for i in range(0, len(s), 2))


def rows_from_osu_json(entries: list) -> list[dict]:
    """Convert osu!-global JSON entries (the bot contract) into row dicts shaped
    like query_leaderboard() output, so build_board() consumes them unchanged.

    Each entry (already sorted by rank): {rank, username, score, accuracy
    (0..100), grade (osu letter), max_combo, mods (bitmask int), mods_str,
    count300, count100, count50, count_miss, avatar_png (path|null)}. Robust to
    missing keys. discord_user_id is None (osu players aren't Discord-linked);
    avatar_png carries the pre-fetched osu avatar the flank card loads."""
    rows: list[dict] = []
    for e in entries or []:
        if not isinstance(e, dict):
            continue
        rows.append({
            "player_name": str(e.get("username") or "?"),
            "discord_user_id": None,
            "score": int(e.get("score") or 0),
            "accuracy": float(e.get("accuracy") or 0.0),
            "grade": _norm_osu_grade(e.get("grade")),
            "max_combo": int(e.get("max_combo") or 0),
            "mods_str": _mods_str_to_csv(e.get("mods_str")),
            "mods": int(e.get("mods") or 0),
            "count_300": int(e.get("count300") or 0),
            "count_100": int(e.get("count100") or 0),
            "count_50": int(e.get("count50") or 0),
            "count_miss": int(e.get("count_miss") or 0),
            "count_katu": int(e.get("count_katu") or 0),  # catch missed droplets
            "replay_md5": None,
            "avatar_png": (str(e["avatar_png"]) if e.get("avatar_png") else None),
        })
    return rows


def compute_rank(other_scores, current_score: int) -> int:
    """The current play's 1-based rank among the OTHER players' best scores:
    1 + how many others strictly outscore it (ties rank the current play
    above the tied other)."""
    return 1 + sum(1 for s in other_scores if int(s) > int(current_score))


def build_board(rows: list[dict], current_player: str, current_score: int,
                prev_best_score: int | None = None,
                max_per_side: int = 4) -> BoardData:
    """Assemble the flanked board around the current play.

    `rows` = query_leaderboard() output (best-per-player, all players, score
    DESC). The current player is filtered OUT of the flanks (the featured
    card represents them); the current play's rank is computed against the
    remaining OTHER players. The board is CENTRED on the current play — up to
    `max_per_side` others ranked immediately above go on the left (nearest the
    centre = closest score), the same below on the right — so a deep play
    shows its neighbours and a sparse map shows whatever exists (0/1/2/N).

    Rank moment (owner-chosen, cheap — we already hold the numbers):
      * "NEW #1"   the play tops a populated board (beats every other player);
      * "NEW BEST" it beats the player's OWN previous best on this map;
      * None       otherwise (incl. a solo first render — nothing to beat)."""
    others = [r for r in rows
              if (r.get("player_name") or "").strip().lower()
              != (current_player or "").strip().lower()]
    others.sort(key=lambda r: int(r.get("score") or 0), reverse=True)

    rank = compute_rank([r.get("score") or 0 for r in others], current_score)
    n_players = len(others) + 1

    # assign absolute board ranks: others above the featured slot keep 1..R-1,
    # those below shift down to R+1.. (the featured play occupies rank R)
    above, below = [], []
    for i, r in enumerate(others):
        abs_rank = i + 1 if (i + 1) < rank else i + 2
        (above if abs_rank < rank else below).append(_entry_from_row(abs_rank, r))
    # left = the up-to-N nearest ABOVE, ascending rank so the closest score
    # (rank R-1) sits nearest the centre panel (rightmost of the left group)
    left = above[-max_per_side:] if max_per_side > 0 else []
    right = below[:max_per_side] if max_per_side > 0 else []

    moment = None
    if others and rank == 1:
        moment = "NEW #1"
    elif prev_best_score is not None and int(current_score) > int(prev_best_score):
        moment = "NEW BEST"

    return BoardData(left=left, right=right, rank=rank, n_players=n_players,
                     moment=moment)


# --- Discord avatars (owner-chosen) ------------------------------------------------
# discord_user_id → GET discord.com/api/v10/users/{id} (bot token) → avatar
# hash → cdn.discordapp.com/avatars/{id}/{hash}.png. Cached to disk so we
# never refetch; every failure path falls back to the procedural chip. The
# render must NEVER block or crash on avatars.

AVATAR_CACHE_DIR = os.path.expanduser("~/.cache/r3d-catch/avatars")
_AVATAR_TIMEOUT = 4.0             # per-request seconds
_NEG_CACHE: set[str] = set()      # ids we already failed this process


def avatar_cache_path(discord_user_id: str) -> str:
    """The on-disk cache path for a user's avatar PNG."""
    safe = "".join(c for c in str(discord_user_id) if c.isalnum() or c in "_-")
    return os.path.join(AVATAR_CACHE_DIR, f"{safe}.png")


def is_fetchable_id(discord_user_id) -> bool:
    """A real Discord snowflake (all-digit id) is fetchable; the DB's
    `osu_<id>` placeholder (a linked osu! account with no Discord user) and
    blanks are NOT → procedural fallback."""
    s = str(discord_user_id or "").strip()
    return bool(s) and s.isdigit()


def _discord_token() -> str | None:
    """The bot token from the env (R3D_DISCORD_BOT_TOKEN / DISCORD_BOT_TOKEN)
    — None disables fetching (→ procedural fallback everywhere)."""
    return (os.environ.get("R3D_DISCORD_BOT_TOKEN")
            or os.environ.get("DISCORD_BOT_TOKEN") or None)


def _socks5_socket(proxy_host: str, proxy_port: int, dest_host: str,
                   dest_port: int, timeout: float):
    """A minimal stdlib SOCKS5 CONNECT tunnel (no auth) — the box reaches
    discord.com only through the bot's local SOCKS5 proxy. Returns a connected
    raw socket to dest, or raises."""
    import socket
    import struct
    s = socket.create_connection((proxy_host, proxy_port), timeout=timeout)
    s.settimeout(timeout)
    s.sendall(b"\x05\x01\x00")                      # VER, 1 method, NO-AUTH
    if s.recv(2) != b"\x05\x00":
        raise OSError("socks5: no-auth rejected")
    host_b = dest_host.encode("idna")
    req = b"\x05\x01\x00\x03" + struct.pack("B", len(host_b)) + host_b + \
        struct.pack(">H", dest_port)
    s.sendall(req)
    rep = s.recv(4)
    if len(rep) < 2 or rep[1] != 0x00:
        raise OSError("socks5: connect failed")
    atyp = rep[3] if len(rep) >= 4 else 1
    ln = {1: 4, 4: 16}.get(atyp)
    if ln is None:                                  # domain
        ln = s.recv(1)[0]
    s.recv(ln + 2)                                  # bound addr + port
    return s


def _https_get(host: str, path: str, headers: dict, timeout: float,
               proxy_url: str | None) -> bytes:
    """A tiny HTTPS GET (stdlib only) that optionally tunnels through a
    SOCKS5 proxy. Raises on any non-200 / transport error."""
    import http.client
    import ssl
    ctx = ssl.create_default_context()
    if proxy_url and proxy_url.startswith("socks5"):
        from urllib.parse import urlparse
        pu = urlparse(proxy_url)
        raw = _socks5_socket(pu.hostname, pu.port or 1080, host, 443, timeout)
        sock = ctx.wrap_socket(raw, server_hostname=host)
        conn = http.client.HTTPSConnection(host, timeout=timeout)
        conn.sock = sock                            # pre-connected tunnel
    else:
        conn = http.client.HTTPSConnection(host, timeout=timeout, context=ctx)
    try:
        conn.request("GET", path, headers=headers)
        resp = conn.getresponse()
        if resp.status != 200:
            raise OSError(f"http {resp.status}")
        return resp.read()
    finally:
        conn.close()


def fetch_discord_avatar(discord_user_id: str, token: str,
                         proxy_url: str | None = None,
                         timeout: float = _AVATAR_TIMEOUT) -> bytes | None:
    """Fetch a user's avatar PNG bytes from Discord (bot token). Two hops:
    the user object (→ avatar hash), then the CDN image. Returns None on any
    failure (no avatar set, transport error, non-200). Best-effort only."""
    import json
    try:
        body = _https_get(
            "discord.com", f"/api/v10/users/{discord_user_id}",
            {"Authorization": f"Bot {token}", "User-Agent": "r3d-catch/1.0"},
            timeout, proxy_url)
        user = json.loads(body)
        avatar_hash = user.get("avatar")
        if not avatar_hash:
            return None                             # default avatar → fallback
        ext = "gif" if str(avatar_hash).startswith("a_") else "png"
        img = _https_get(
            "cdn.discordapp.com",
            f"/avatars/{discord_user_id}/{avatar_hash}.{ext}?size=128",
            {"User-Agent": "r3d-catch/1.0"}, timeout, proxy_url)
        return img
    except Exception:  # noqa: BLE001 — any failure → procedural fallback
        return None


def resolve_avatar_bytes(discord_user_id, *, token: str | None = None,
                         proxy_url: str | None = None,
                         allow_fetch: bool = True) -> bytes | None:
    """Cache-first avatar bytes for a user, or None → procedural fallback.

    Order: on-disk cache → (if fetchable id + token + allow_fetch) Discord
    fetch → cache the result. A negative in-process cache stops re-hitting a
    failing id. NEVER raises."""
    if not is_fetchable_id(discord_user_id):
        return None
    did = str(discord_user_id)
    path = avatar_cache_path(did)
    try:
        if os.path.exists(path):
            with open(path, "rb") as fh:
                data = fh.read()
            return data or None
    except Exception:  # noqa: BLE001
        pass
    if not allow_fetch or did in _NEG_CACHE:
        return None
    token = token or _discord_token()
    if not token:
        return None
    proxy_url = proxy_url if proxy_url is not None \
        else os.environ.get("DISCORD_PROXY_URL")
    data = fetch_discord_avatar(did, token, proxy_url)
    if not data:
        _NEG_CACHE.add(did)
        return None
    try:
        os.makedirs(AVATAR_CACHE_DIR, exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(data)
    except Exception:  # noqa: BLE001 — cache write is best-effort
        pass
    return data
