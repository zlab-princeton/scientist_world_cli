"""GitHub access: list the environments of an organization and clone them.

A token is optional. Without one only public environments are visible and
GitHub's anonymous rate limit applies (60 requests per hour). A token is
looked up in this order:

1. ``SCIWORLD_GITHUB_TOKEN``, ``GITHUB_TOKEN``, ``GH_TOKEN``
2. ``gh auth token`` (the GitHub CLI)
3. ``~/.git-credentials``
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from sciworld.milestones import MILESTONE_RANK, NOT_POSSIBLE

log = logging.getLogger(__name__)

DEFAULT_ORG = "ScientistWorld"
API_URL = "https://api.github.com"

# Topic an environment repository carries once its construction is complete.
DONE_TOPIC = "done"

_TOKEN_ENV_VARS = ("SCIWORLD_GITHUB_TOKEN", "GITHUB_TOKEN", "GH_TOKEN")
_ARXIV_ID = re.compile(r"^(\d{4}\.\d{4,5})(?:v\d+)?(?=$|[-_.])")


class GitHubError(RuntimeError):
    """A GitHub request or clone failed."""


@dataclass
class RepoInfo:
    """One environment repository."""

    arxiv_id: str
    milestone: str
    title: str
    repo_name: str
    clone_url: str
    topics: list = field(default_factory=list)
    created_at: str = ""
    pushed_at: str = ""
    html_url: str = ""
    private: bool = False

    def to_dict(self) -> dict:
        return {
            "arxiv_id": self.arxiv_id,
            "milestone": self.milestone,
            "title": self.title,
            "repo_name": self.repo_name,
            "clone_url": self.clone_url,
            "topics": list(self.topics),
            "created_at": self.created_at,
            "pushed_at": self.pushed_at,
            "html_url": self.html_url,
            "private": self.private,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RepoInfo":
        return cls(
            arxiv_id=d["arxiv_id"],
            milestone=d.get("milestone", "none"),
            title=d.get("title", ""),
            repo_name=d["repo_name"],
            clone_url=d.get("clone_url", ""),
            topics=list(d.get("topics", [])),
            created_at=d.get("created_at", ""),
            pushed_at=d.get("pushed_at", ""),
            html_url=d.get("html_url", ""),
            private=bool(d.get("private", False)),
        )

    @classmethod
    def from_api(cls, repo: dict) -> Optional["RepoInfo"]:
        """Build from a GitHub REST repository object; ``None`` if the
        repository is not named after an arXiv id."""
        name = repo.get("name") or ""
        arxiv_id = parse_arxiv_id(name)
        if not arxiv_id:
            return None
        description = repo.get("description") or ""
        milestone, title = parse_description(description)
        return cls(
            arxiv_id=arxiv_id,
            milestone=milestone,
            title=title or name,
            repo_name=name,
            clone_url=repo.get("clone_url") or "",
            topics=list(repo.get("topics") or []),
            created_at=repo.get("created_at") or "",
            pushed_at=repo.get("pushed_at") or "",
            html_url=repo.get("html_url") or "",
            private=bool(repo.get("private", False)),
        )


# ── Parsing ───────────────────────────────────────────────────────────────


def parse_arxiv_id(repo_name: str) -> Optional[str]:
    """``'2512.01557-some-title'`` -> ``'2512.01557'``."""
    match = _ARXIV_ID.match(repo_name or "")
    return match.group(1) if match else None


def parse_description(description: str) -> tuple:
    """Split a repository description ``'MILESTONE: Title'``.

    Returns ``(milestone, title)``. A description without a known milestone
    prefix is returned whole as the title, with milestone ``'none'``.
    """
    description = (description or "").strip()
    head, sep, tail = description.partition(":")
    candidate = head.strip().lower()
    if sep and (candidate in MILESTONE_RANK or candidate == NOT_POSSIBLE):
        return candidate, tail.strip()
    return "none", description


def parse_topics(topics: list) -> dict:
    """Group topic tags by their ``area-``, ``category-``, ``domain-`` and
    ``method-`` prefixes."""
    result: dict = {"areas": [], "categories": [], "domains": [], "methods": []}
    for prefix, key in (("area-", "areas"), ("category-", "categories"),
                        ("domain-", "domains"), ("method-", "methods")):
        result[key] = [t[len(prefix):] for t in topics if t.startswith(prefix)]
    return result


# ── Authentication ────────────────────────────────────────────────────────


def _token_from_gh() -> Optional[str]:
    if not shutil.which("gh"):
        return None
    try:
        result = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    token = result.stdout.strip()
    return token if result.returncode == 0 and token else None


def _token_from_git_credentials() -> Optional[str]:
    try:
        lines = (Path.home() / ".git-credentials").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        # The host must be github.com itself, not a name that starts with it.
        match = re.search(r"://[^:/@]+:([^@]+)@github\.com(?:[:/]|\s*$)", line)
        if match:
            return match.group(1)
    return None


# Where each token came from, for error messages; and tokens GitHub rejected,
# which are not sent again.
_TOKEN_SOURCES: dict = {}
_REJECTED_TOKENS: set = set()


def github_token() -> Optional[str]:
    """Return a GitHub token, or ``None`` to work anonymously."""
    token, source = None, ""
    for var in _TOKEN_ENV_VARS:
        value = (os.environ.get(var) or "").strip()
        if value:
            token, source = value, f"${var}"
            break
    if token is None:
        token = _token_from_gh()
        source = "`gh auth token`"
    if token is None:
        token = _token_from_git_credentials()
        source = "~/.git-credentials"
    if token is None or token in _REJECTED_TOKENS:
        return None
    _TOKEN_SOURCES[token] = source
    return token


def _reject(token: str) -> None:
    """Remember that GitHub refused *token* and say so once."""
    if token not in _REJECTED_TOKENS:
        _REJECTED_TOKENS.add(token)
        log.warning(
            "GitHub rejected the token from %s (expired or revoked?); continuing "
            "without it, so only public environments are visible",
            _TOKEN_SOURCES.get(token, "the environment"),
        )


def _redact(text: str, token: Optional[str]) -> str:
    if token:
        text = text.replace(token, "***")
    return re.sub(r"(://)[^/@\s]+@", r"\1***@", text)


_AUTH_HINT = (
    "Set GITHUB_TOKEN to a token that can read the organization, "
    "or log in with `gh auth login`."
)


# ── REST API ──────────────────────────────────────────────────────────────


def _request(path: str, token: Optional[str], accept: str, timeout: int) -> tuple:
    """GET a GitHub REST path. Returns ``(status, body_bytes, headers)``;
    ``status`` is ``0`` when the request could not be made at all."""
    request = urllib.request.Request(f"{API_URL}{path}", method="GET")
    request.add_header("Accept", accept)
    request.add_header("X-GitHub-Api-Version", "2022-11-28")
    request.add_header("User-Agent", "sciworld-cli")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read()
        except OSError:
            body = b""
        return exc.code, body, dict(exc.headers or {})
    except (urllib.error.URLError, OSError, ValueError) as exc:
        log.debug("GitHub request failed: %s: %s", path, exc)
        return 0, b"", {}


def _get(path: str, token: Optional[str], accept: str, timeout: int) -> tuple:
    """:func:`_request`, retried anonymously when GitHub rejects the token.

    GitHub answers 401 to a bad token even for public data, so a stale token
    in the environment would otherwise block everything.
    """
    if token and token in _REJECTED_TOKENS:
        token = None
    status, body, headers = _request(path, token, accept, timeout)
    if status == 401 and token:
        _reject(token)
        status, body, headers = _request(path, None, accept, timeout)
    return status, body, headers


def _json(body: bytes):
    try:
        return json.loads(body.decode("utf-8", errors="replace")) if body else None
    except ValueError:
        return None


def api_get(path: str, token: Optional[str] = None, timeout: int = 30) -> tuple:
    """GET a GitHub REST path. Returns ``(status, json_or_None, headers)``.

    Never raises for an HTTP error status; ``status`` is ``0`` when the
    request could not be made at all.
    """
    status, body, headers = _get(path, token, "application/vnd.github+json", timeout)
    return status, _json(body), headers


def _failure(action: str, status: int, data, headers: dict, token: Optional[str]) -> GitHubError:
    """Turn a failed response into an error with an actionable message."""
    message = data.get("message") if isinstance(data, dict) else None
    lowered = {k.lower(): v for k, v in (headers or {}).items()}
    if status == 0:
        return GitHubError(f"{action}: cannot reach api.github.com (check your network)")
    if status in (403, 429) and (
        lowered.get("x-ratelimit-remaining") == "0" or "rate limit" in (message or "").lower()
    ):
        hint = "" if token else f" {_AUTH_HINT}"
        return GitHubError(f"{action}: GitHub API rate limit reached.{hint}")
    if status == 401:
        source = f" from {_TOKEN_SOURCES[token]}" if token in _TOKEN_SOURCES else ""
        return GitHubError(f"{action}: the GitHub token{source} was rejected (expired or revoked?)")
    if status == 404 and not token:
        return GitHubError(f"{action}: not found, or private. {_AUTH_HINT}")
    return GitHubError(f"{action}: HTTP {status}{': ' + message if message else ''}")


def _last_page(headers: dict) -> int:
    """Number of the last page according to a ``Link`` response header."""
    link = next((v for k, v in (headers or {}).items() if k.lower() == "link"), "")
    match = re.search(r'[?&]page=(\d+)[^>]*>;\s*rel="last"', link)
    return int(match.group(1)) if match else 1


def list_environments(
    org: str = DEFAULT_ORG,
    token: Optional[str] = None,
    on_progress: Optional[Callable] = None,
    include_unfinished: bool = False,
) -> list:
    """Return the organization's environments as :class:`RepoInfo` objects.

    Only repositories tagged with the ``done`` topic are returned unless
    *include_unfinished* is set. The listing pages through the organization's
    repositories, so it is not subject to the 1000-result cap of the search
    API. *on_progress(fetched)* is called after each page.
    """
    def _page(number: int) -> tuple:
        status, data, headers = api_get(
            f"/orgs/{org}/repos?per_page=100&page={number}&type=all&sort=full_name", token,
        )
        if status != 200 or not isinstance(data, list):
            raise _failure(f"Listing environments of {org}", status, data, headers, token)
        return data, headers

    repos: list = []

    def _collect(items: list) -> None:
        for item in items:
            info = RepoInfo.from_api(item)
            if info is not None and (include_unfinished or DONE_TOPIC in info.topics):
                repos.append(info)
        if on_progress:
            on_progress(len(repos))

    first, headers = _page(1)
    _collect(first)
    last = _last_page(headers)
    if last > 1:
        # The remaining pages are independent requests: fetch a few at a time.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(6, last - 1)) as pool:
            for items, _headers in pool.map(_page, range(2, last + 1)):
                _collect(items)
    elif len(first) == 100:
        # No Link header: fall back to walking the pages one by one.
        number = 2
        while True:
            items, _headers = _page(number)
            _collect(items)
            if len(items) < 100:
                break
            number += 1
    repos.sort(key=lambda r: r.arxiv_id)
    return repos


def find_environment(
    arxiv_id: str, org: str = DEFAULT_ORG, token: Optional[str] = None,
) -> Optional[RepoInfo]:
    """Look up the environment of one paper without listing the organization.

    Returns ``None`` when no repository of *org* is named after *arxiv_id*.
    """
    status, data, headers = api_get(
        f"/search/repositories?q={arxiv_id}+in:name+org:{org}&per_page=20", token,
    )
    if status != 200 or not isinstance(data, dict):
        raise _failure(f"Looking up {arxiv_id} in {org}", status, data, headers, token)
    matches = []
    for item in data.get("items") or []:
        info = RepoInfo.from_api(item)
        if info is not None and info.arxiv_id == arxiv_id:
            matches.append(info)
    if not matches:
        return None
    # Prefer a finished environment if several repositories share the id.
    matches.sort(key=lambda r: (DONE_TOPIC not in r.topics, r.repo_name))
    return matches[0]


def fetch_file(
    repo_name: str, path: str, ref: Optional[str] = None,
    org: str = DEFAULT_ORG, token: Optional[str] = None,
) -> Optional[bytes]:
    """Return the content of one file of a repository, or ``None`` if the
    repository has no such file.

    Raises :class:`GitHubError` for every other failure (network, rate limit,
    access), so that "absent" is never confused with "could not be read".
    """
    query = f"?ref={ref}" if ref else ""
    # The raw media type returns the bytes themselves, whatever the size.
    status, body, headers = _get(
        f"/repos/{org}/{repo_name}/contents/{path}{query}", token,
        "application/vnd.github.raw+json", 60,
    )
    if status == 200:
        return body
    if status == 404:
        return None
    raise _failure(f"Reading {path} of {org}/{repo_name}", status, _json(body), headers, token)


def fetch_repo_tldr(repo_name: str, org: str = DEFAULT_ORG, token: Optional[str] = None) -> str:
    """Return the one-line summary from an environment's README, if any."""
    if token is None:
        token = github_token()
    status, data, _headers = api_get(f"/repos/{org}/{repo_name}/readme", token)
    if status != 200 or not isinstance(data, dict):
        return ""
    try:
        content = base64.b64decode(data.get("content", "")).decode("utf-8", errors="replace")
    except (ValueError, TypeError):
        return ""
    for line in content.splitlines():
        stripped = line.strip().lstrip("-* ").strip()
        lowered = stripped.lower()
        if lowered.startswith("**tl;dr:**") or lowered.startswith("**tldr:**"):
            return stripped.split(":**", 1)[-1].strip()
    return ""


# ── Clone ─────────────────────────────────────────────────────────────────

# Hands the token to git through the environment, so it never appears in a
# URL, on a command line, or in the clone's .git/config.
_CREDENTIAL_HELPER = (
    '!f() { test "$1" = get && echo username=x-access-token '
    '&& echo "password=${SCIWORLD_GIT_TOKEN}"; }; f'
)


def clone_repo(clone_url: str, dest: Path, token: Optional[str] = None, timeout: int = 900) -> None:
    """Shallow-clone *clone_url* into *dest*.

    Raises :class:`GitHubError` with the token redacted from the message.
    """
    if not shutil.which("git"):
        raise GitHubError("git is not installed or not on PATH")
    dest = Path(dest)
    if os.path.lexists(str(dest)) and (not dest.is_dir() or any(dest.iterdir())):
        # git would refuse it, and cleaning up after the failure must never
        # remove something that was there before.
        raise GitHubError(f"cannot clone into {dest}: it already exists and is not empty")
    existed = os.path.lexists(str(dest))
    if token and token in _REJECTED_TOKENS:
        token = None

    def _cleanup() -> None:
        if not existed:
            shutil.rmtree(dest, ignore_errors=True)

    def _clone(with_token: Optional[str]):
        env = dict(os.environ)
        env["GIT_TERMINAL_PROMPT"] = "0"
        cmd = ["git"]
        if with_token and clone_url.startswith("https://"):
            env["SCIWORLD_GIT_TOKEN"] = with_token
            # The empty value clears helpers configured by the user, so the
            # token found by sciworld is the one that is used.
            cmd += ["-c", "credential.helper=", "-c", f"credential.helper={_CREDENTIAL_HELPER}"]
        cmd += ["clone", "--depth", "1", "--quiet", clone_url, str(dest)]
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            _cleanup()
            raise GitHubError(f"git clone timed out after {timeout}s: {_redact(clone_url, with_token)}")

    def _is_auth_failure(stderr: str) -> bool:
        lowered = stderr.lower()
        return any(s in lowered for s in (
            "repository not found", "could not read username", "terminal prompts disabled",
            "authentication failed", "invalid username or password", "403",
        ))

    result = _clone(token)
    if result.returncode == 0:
        return
    _cleanup()
    stderr = _redact(result.stderr.strip(), token)
    if _is_auth_failure(stderr):
        if token:
            # A token that cannot read this repository must not stop the
            # clone of a public one.
            retry = _clone(None)
            if retry.returncode == 0:
                return
            _cleanup()
            source = _TOKEN_SOURCES.get(token, "the environment")
            hint = f"The token from {source} cannot read this repository."
        else:
            hint = f"The repository is private or does not exist. {_AUTH_HINT}"
        raise GitHubError(f"Cannot clone {_redact(clone_url, token)}. {hint}")
    raise GitHubError(f"git clone failed: {stderr or 'unknown error'}")
