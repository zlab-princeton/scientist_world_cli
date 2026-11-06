import subprocess

import pytest

from sciworld import github
from sciworld.github import GitHubError, RepoInfo


def repo(name, description="core_claim: A Title", topics=("done",), private=True):
    return {
        "name": name, "description": description, "topics": list(topics),
        "clone_url": f"https://github.com/Org/{name}.git", "html_url": f"https://github.com/Org/{name}",
        "created_at": "2026-01-01T00:00:00Z", "pushed_at": "2026-02-01T00:00:00Z", "private": private,
    }


# ── Parsing ──

@pytest.mark.parametrize("name, expected", [
    ("2512.01557-some-title", "2512.01557"),
    ("2401.0001-short", "2401.0001"),
    ("2512.01557", "2512.01557"),
    ("2512.01557v2-title", "2512.01557"),
    ("25120.1557-title", None),
    ("notes-2512.01557", None),
    ("", None),
])
def test_parse_arxiv_id(name, expected):
    assert github.parse_arxiv_id(name) == expected


@pytest.mark.parametrize("description, expected", [
    ("core_claim: Attention Is All You Need", ("core_claim", "Attention Is All You Need")),
    ("FULL: Title: With Colon", ("full", "Title: With Colon")),
    ("not_possible: Broken", ("not_possible", "Broken")),
    ("A title: with a colon but no milestone", ("none", "A title: with a colon but no milestone")),
    ("", ("none", "")),
    (None, ("none", "")),
])
def test_parse_description(description, expected):
    assert github.parse_description(description) == expected


def test_repo_info_from_api_and_round_trip():
    info = RepoInfo.from_api(repo("2512.01557-x", topics=("done", "area-nlp")))
    assert (info.arxiv_id, info.milestone, info.title) == ("2512.01557", "core_claim", "A Title")
    assert info.private is True
    assert RepoInfo.from_dict(info.to_dict()) == info
    assert RepoInfo.from_api(repo("not-a-paper")) is None
    assert RepoInfo.from_api(repo("2512.01557-x", description=None)).title == "2512.01557-x"


def test_parse_topics_groups_by_prefix():
    grouped = github.parse_topics(["done", "area-nlp", "method-rl", "domain-vision", "category-eval", "area-cv"])
    assert grouped == {"areas": ["nlp", "cv"], "categories": ["eval"], "domains": ["vision"], "methods": ["rl"]}


# ── Token discovery ──

def test_no_token_by_default():
    assert github.github_token() is None


def test_token_environment_variables_in_order(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "gh")
    assert github.github_token() == "gh"
    monkeypatch.setenv("GITHUB_TOKEN", "github")
    assert github.github_token() == "github"
    monkeypatch.setenv("SCIWORLD_GITHUB_TOKEN", "own")
    assert github.github_token() == "own"


def test_token_from_gh_cli_then_git_credentials(monkeypatch, isolated):
    (isolated / "home" / ".git-credentials").write_text(
        "https://someone:other@gitlab.com\nhttps://user:ghp_fromfile@github.com\n")
    assert github.github_token() == "ghp_fromfile"
    monkeypatch.setattr(github, "_token_from_gh", lambda: "from-gh")
    assert github.github_token() == "from-gh"


@pytest.mark.parametrize("line, expected", [
    ("https://user:tok@github.com", "tok"),
    ("https://user:tok@github.com/", "tok"),
    ("https://user:tok@github.com:443", "tok"),
    # Other hosts whose name merely starts with github.com.
    ("https://user:corp@github.company.com", None),
    ("https://user:corp@github.com.evil.example", None),
    ("https://user:corp@github.community", None),
])
def test_git_credentials_are_only_used_for_github_itself(isolated, line, expected):
    (isolated / "home" / ".git-credentials").write_text(line + "\n")
    assert github.github_token() == expected


def test_a_rejected_token_does_not_block_public_data(monkeypatch, caplog):
    """GitHub answers 401 to a stale token even for public repositories."""
    monkeypatch.setenv("GITHUB_TOKEN", "stale")
    seen = []

    def fake_request(path, token, accept, timeout):
        seen.append(token)
        if token:
            return 401, b'{"message": "Bad credentials"}', {}
        return 200, b'{"items": []}', {}

    monkeypatch.setattr(github, "_request", fake_request)
    with caplog.at_level("WARNING"):
        status, data, _ = github.api_get("/search/repositories?q=x", github.github_token())
    assert (status, data) == (200, {"items": []}) and seen == ["stale", None]
    assert "rejected the token from $GITHUB_TOKEN" in caplog.text
    # It is not sent again, and the rest of the program works anonymously.
    assert github.github_token() is None
    github.api_get("/x", "stale")
    assert seen == ["stale", None, None]


# ── REST failures ──

@pytest.mark.parametrize("status, data, headers, token, expected", [
    (0, None, {}, None, "cannot reach api.github.com"),
    (403, {"message": "API rate limit exceeded"}, {"X-RateLimit-Remaining": "0"}, None, "rate limit reached. Set GITHUB_TOKEN"),
    (429, {"message": "secondary rate limit"}, {}, "t", "rate limit reached."),
    (401, {"message": "Bad credentials"}, {}, "t", "token was rejected"),
    (404, {"message": "Not Found"}, {}, None, "not found, or private"),
    (500, {"message": "boom"}, {}, "t", "HTTP 500: boom"),
])
def test_failure_messages(status, data, headers, token, expected):
    assert expected in str(github._failure("Listing", status, data, headers, token))


def test_rate_limit_hint_is_not_shown_with_a_token():
    message = str(github._failure("x", 403, {"message": "rate limit"}, {}, "token"))
    assert "GITHUB_TOKEN" not in message


# ── Listing ──

def test_list_environments_pages_and_keeps_finished_papers(monkeypatch):
    pages = {
        1: [repo(f"2501.{i:05d}-p") for i in range(100)],
        2: [repo("2502.00001-done"), repo("2502.00002-wip", topics=("iteration-3",)), repo("docs")],
    }
    seen = []

    def fake_get(path, token=None, timeout=30):
        seen.append((path, token))
        page = int(path.split("&page=")[1].split("&")[0])
        return 200, pages.get(page, []), {}

    monkeypatch.setattr(github, "api_get", fake_get)
    progress = []
    repos = github.list_environments("Org", token="tok", on_progress=progress.append)
    assert len(repos) == 101 and repos[-1].arxiv_id == "2502.00001"
    assert [r.arxiv_id for r in repos] == sorted(r.arxiv_id for r in repos)
    assert len(seen) == 2 and all(token == "tok" for _, token in seen)
    assert "/orgs/Org/repos" in seen[0][0] and "type=all" in seen[0][0]
    assert progress == [100, 101]
    assert len(github.list_environments("Org", include_unfinished=True)) == 102


def test_list_environments_follows_the_link_header(monkeypatch):
    seen = []

    def fake_get(path, token=None, timeout=30):
        page = int(path.split("&page=")[1].split("&")[0])
        seen.append(page)
        link = '<https://api.github.com/x?per_page=100&page=2>; rel="next", <https://api.github.com/x?per_page=100&page=4>; rel="last"'
        return 200, [repo(f"25{page:02d}.{i:05d}-p") for i in range(100 if page < 4 else 7)], {"Link": link}

    monkeypatch.setattr(github, "api_get", fake_get)
    repos = github.list_environments("Org")
    assert len(repos) == 307 and sorted(seen) == [1, 2, 3, 4]
    assert github._last_page({"link": '<u?page=9>; rel="last"'}) == 9 and github._last_page({}) == 1


def test_a_failing_page_fails_the_listing(monkeypatch):
    def fake_get(path, token=None, timeout=30):
        page = int(path.split("&page=")[1].split("&")[0])
        if page == 3:
            return 502, None, {}
        return 200, [repo(f"25{page:02d}.{i:05d}-p") for i in range(100)], {"Link": '<u?page=5>; rel="last"'}

    monkeypatch.setattr(github, "api_get", fake_get)
    with pytest.raises(GitHubError, match="HTTP 502"):
        github.list_environments("Org")


def test_list_environments_raises_on_failure(monkeypatch):
    monkeypatch.setattr(github, "api_get", lambda *a, **k: (404, {"message": "Not Found"}, {}))
    with pytest.raises(GitHubError, match="Listing environments of Org"):
        github.list_environments("Org")


def test_find_environment_requires_an_exact_id(monkeypatch):
    items = [repo("2502.137590-other"), repo("2502.13759-wip", topics=()), repo("2502.13759-geo")]
    monkeypatch.setattr(github, "api_get", lambda *a, **k: (200, {"items": items}, {}))
    found = github.find_environment("2502.13759", org="Org")
    assert found.repo_name == "2502.13759-geo"  # the finished one is preferred
    monkeypatch.setattr(github, "api_get", lambda *a, **k: (200, {"items": []}, {}))
    assert github.find_environment("2502.13759") is None
    monkeypatch.setattr(github, "api_get", lambda *a, **k: (403, {"message": "rate limit"}, {}))
    with pytest.raises(GitHubError):
        github.find_environment("2502.13759")


def test_fetch_file_returns_the_bytes_or_none_when_absent(monkeypatch):
    calls = []

    def fake_request(path, token, accept, timeout):
        calls.append((path, accept))
        return 200, b'{"a": 1}', {}

    monkeypatch.setattr(github, "_request", fake_request)
    assert github.fetch_file("repo", "dir/f.json", ref="branch", org="Org") == b'{"a": 1}'
    # The raw media type has no size limit, unlike the base64 JSON form.
    assert calls == [("/repos/Org/repo/contents/dir/f.json?ref=branch", "application/vnd.github.raw+json")]
    monkeypatch.setattr(github, "_request", lambda *a, **k: (404, b'{"message": "Not Found"}', {}))
    assert github.fetch_file("repo", "f") is None


@pytest.mark.parametrize("status, body, headers, expected", [
    (403, b'{"message": "API rate limit exceeded"}', {"X-RateLimit-Remaining": "0"}, "rate limit"),
    (0, b"", {}, "cannot reach api.github.com"),
    (500, b'{"message": "boom"}', {}, "HTTP 500"),
])
def test_fetch_file_raises_when_the_file_could_not_be_read(monkeypatch, status, body, headers, expected):
    """Absent and unreadable are different answers."""
    monkeypatch.setattr(github, "_request", lambda *a, **k: (status, body, headers))
    with pytest.raises(GitHubError, match=expected):
        github.fetch_file("repo", "f")


# ── Clone ──

def _local_repo(path):
    path.mkdir()
    (path / "file.txt").write_text("content\n")
    def run(*args):
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
                              cwd=path, check=True, capture_output=True)
    run("init", "-q")
    run("add", ".")
    run("commit", "-q", "-m", "one")
    return path


def test_clone_from_a_local_repository(tmp_path):
    origin = _local_repo(tmp_path / "origin")
    dest = tmp_path / "clone"
    github.clone_repo(str(origin), dest)
    assert (dest / "file.txt").read_text() == "content\n"


def test_clone_failure_removes_the_partial_directory(tmp_path):
    dest = tmp_path / "clone"
    with pytest.raises(GitHubError):
        github.clone_repo(str(tmp_path / "does-not-exist"), dest)
    assert not dest.exists()


def test_clone_never_removes_what_was_there_before(tmp_path):
    origin = _local_repo(tmp_path / "origin")
    dest = tmp_path / "mine"
    dest.mkdir()
    (dest / "work.txt").write_text("keep\n")
    with pytest.raises(GitHubError, match="already exists and is not empty"):
        github.clone_repo(str(origin), dest)
    assert (dest / "work.txt").read_text() == "keep\n"
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(GitHubError):
        github.clone_repo(str(tmp_path / "does-not-exist"), empty)
    assert empty.is_dir()  # it existed before the call, so it stays
    github.clone_repo(str(origin), empty)
    assert (empty / "file.txt").exists()


def test_token_reaches_git_through_the_environment_only(tmp_path, monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured.setdefault("cmd", cmd)
        captured.setdefault("env", kwargs["env"])
        captured.setdefault("all", []).append(cmd)
        return subprocess.CompletedProcess(cmd, 128, "", "fatal: Authentication failed for 'https://x:ghp_SECRET@github.com/Org/r.git'")

    monkeypatch.setattr(github.subprocess, "run", fake_run)
    with pytest.raises(GitHubError) as error:
        github.clone_repo("https://github.com/Org/r.git", tmp_path / "c", token="ghp_SECRET")
    assert "ghp_SECRET" not in " ".join(captured["cmd"])
    assert captured["env"]["SCIWORLD_GIT_TOKEN"] == "ghp_SECRET"
    assert captured["env"]["GIT_TERMINAL_PROMPT"] == "0"
    assert captured["cmd"][1:4] == ["-c", "credential.helper=", "-c"]
    assert captured["cmd"][-2] == "https://github.com/Org/r.git"
    assert "ghp_SECRET" not in str(error.value)
    assert "cannot read this repository" in str(error.value)
    # A token that cannot read the repository must not stop a public clone:
    # the second attempt is anonymous.
    assert len(captured["all"]) == 2 and "credential.helper=" not in captured["all"][1]


def test_a_public_clone_succeeds_although_the_token_is_refused(tmp_path, monkeypatch):
    attempts = []

    def fake_run(cmd, **kwargs):
        attempts.append("SCIWORLD_GIT_TOKEN" in kwargs["env"])
        if attempts[-1]:
            return subprocess.CompletedProcess(cmd, 128, "", "fatal: Authentication failed")
        (tmp_path / "c").mkdir()
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(github.subprocess, "run", fake_run)
    github.clone_repo("https://github.com/Org/public.git", tmp_path / "c", token="stale")
    assert attempts == [True, False] and (tmp_path / "c").is_dir()


def test_anonymous_clone_failure_explains_how_to_authenticate(tmp_path, monkeypatch):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 128, "", "fatal: could not read Username for 'https://github.com': terminal prompts disabled")
    monkeypatch.setattr(github.subprocess, "run", fake_run)
    with pytest.raises(GitHubError, match="private or does not exist. Set GITHUB_TOKEN"):
        github.clone_repo("https://github.com/Org/r.git", tmp_path / "c")


def test_credential_helper_returns_the_token(tmp_path):
    """Run the helper the way git does and read back what it prints."""
    helper = github._CREDENTIAL_HELPER[1:]  # drop the leading "!"
    out = subprocess.run(["sh", "-c", f"{helper} get"], capture_output=True, text=True,
                         env={"SCIWORLD_GIT_TOKEN": "tok123", "PATH": "/usr/bin:/bin"})
    assert out.stdout.split() == ["username=x-access-token", "password=tok123"]
    stored = subprocess.run(["sh", "-c", f"{helper} store"], capture_output=True, text=True,
                            env={"SCIWORLD_GIT_TOKEN": "tok123", "PATH": "/usr/bin:/bin"})
    assert stored.stdout == ""
