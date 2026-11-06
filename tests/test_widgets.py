"""The parts of the interactive paper browser that can be tested without a terminal."""

from datetime import datetime, timedelta, timezone

from sciworld import widgets


def stamp(days_ago: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_every_paper_is_reachable_in_the_time_tab():
    """The buckets used to be "last 90 days", 2025, 2024 and "before 2024":
    an environment updated in another year was in none of them."""
    papers = [
        {"arxiv_id": "recent", "pushed_at": stamp(3)},
        {"arxiv_id": "quarter", "pushed_at": stamp(60)},
        {"arxiv_id": "y2026", "pushed_at": "2026-04-23T10:00:00Z"},
        {"arxiv_id": "y2023", "created_at": "2023-01-05T00:00:00+00:00"},
        {"arxiv_id": "undated"},
        {"arxiv_id": "garbled", "pushed_at": "not a date"},
    ]
    buckets = widgets._time_buckets(papers)
    labels = [label for label, _kind, _value in buckets]
    assert labels[:2] == ["Last 30 days", "Last 90 days"] and labels[-1] == "No date"
    assert "2026" in labels and "2023" in labels
    years = [label for label in labels if label.isdigit()]
    assert years == sorted(years, reverse=True)
    found = {}
    for label, kind, value in buckets:
        found[label] = [p["arxiv_id"] for p in widgets._papers_in_time_bucket(papers, kind, value)]
    assert found["Last 30 days"] == ["recent"] and found["Last 90 days"] == ["recent", "quarter"]
    assert found["2026"] == ["y2026"] or "y2026" in found["2026"]
    assert found["2023"] == ["y2023"] and found["No date"] == ["undated", "garbled"]
    assert {i for ids in found.values() for i in ids} == {p["arxiv_id"] for p in papers}


def test_no_undated_bucket_when_every_paper_has_a_date():
    labels = [b[0] for b in widgets._time_buckets([{"arxiv_id": "a", "pushed_at": "2025-03-01T00:00:00Z"}])]
    assert labels == ["Last 30 days", "Last 90 days", "2025"]


def test_the_visible_window_always_contains_the_cursor():
    for count in (0, 1, 15, 16, 40, 1000):
        for cursor in range(count):
            first = widgets._window(count, cursor, 15)
            assert first <= cursor < first + 15 and first % 15 == 0
    assert widgets._window(10, 9, 15) == 0


def test_topic_groups_are_unique_and_sorted():
    papers = [{"topics": ["done", "area-graph-learning", "method-diffusion"]},
              {"topics": ["domain-vision", "area-graph-learning"]}, {}]
    assert widgets._extract_topic_groups(papers) == [
        ("diffusion", "method-diffusion"), ("graph learning", "area-graph-learning"), ("vision", "domain-vision")]
