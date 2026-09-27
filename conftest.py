"""pytest -> TAP bridge for onp-spec-driven.

The onp-spec engine reads TAP ("ok N - title" / "not ok N - title") and
looks for acceptance-criterion / principle tags in each test TITLE. pytest
titles are node ids, which cannot carry those tags, so this bridge copies
the tags written in each test's docstring into the TAP title.

Enable with `--onp-tap` (onpspec.config.json already does it).
Skipped tests are emitted with a `# SKIP` directive — the engine never
counts them as proof (e.g. a GPU test on a machine without a GPU).
"""
import re

import pytest

# Deliberately written so the onp-spec annotation scanner does not see a tag here.
_TAG_RE = re.compile(r"@(?:spec|principle):(?:AC|P)-\d{3,}")
_KEY = "onp_tap_tags"


def pytest_addoption(parser):
    parser.addoption("--onp-tap", action="store_true", help="print TAP lines for onp-spec verify")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    doc = getattr(getattr(item, "function", None), "__doc__", None) or ""
    tags = " ".join(dict.fromkeys(_TAG_RE.findall(doc)))
    report.user_properties.append((_KEY, tags))


_results = {}


def pytest_runtest_logreport(report):
    tags = dict(report.user_properties).get(_KEY, "")
    prev = _results.get(report.nodeid)
    # failure in any phase dominates; a skip in setup marks the test skipped
    if report.failed:
        _results[report.nodeid] = ("fail", tags)
    elif report.skipped and prev is None:
        _results[report.nodeid] = ("skip", tags)
    elif report.when == "call" and report.passed and prev is None:
        _results[report.nodeid] = ("pass", tags)


def pytest_terminal_summary(terminalreporter, config):
    if not config.getoption("--onp-tap"):
        return
    tr = terminalreporter
    tr.write_line("TAP version 13")
    tr.write_line(f"1..{len(_results)}")
    for i, (nodeid, (status, tags)) in enumerate(_results.items(), start=1):
        title = f"{nodeid} {tags}".strip()
        if status == "pass":
            tr.write_line(f"ok {i} - {title}")
        elif status == "skip":
            tr.write_line(f"ok {i} - {title} # SKIP")
        else:
            tr.write_line(f"not ok {i} - {title}")
