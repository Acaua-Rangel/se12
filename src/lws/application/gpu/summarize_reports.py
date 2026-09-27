"""Use case: classify every microbenchmark report into valid or rejected
(missing principle P-006 metadata) — AC-023's cross-GPU summary."""

from __future__ import annotations

from lws.application.ports.gpu.report_store import ReportStore
from lws.domain.benchmark.summary import ClassifiedReports, ReportMetadata, classify_reports


def summarize_reports(store: ReportStore) -> ClassifiedReports:
    raw_reports = store.read_all_micro_reports()
    metadata_list = tuple(ReportMetadata(fields=report) for report in raw_reports)
    return classify_reports(metadata_list)
