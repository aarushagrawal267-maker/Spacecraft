"""Parsing of ``labeled_anomalies.csv``.

File format (official Telemanom file, 82 rows)::

    chan_id,spacecraft,anomaly_sequences,class,num_values
    P-1,SMAP,"[[2149, 2349], [4536, 4844], [3539, 3779]]","[contextual, contextual, contextual]",8505

* ``anomaly_sequences`` – inclusive ``[start, end]`` index pairs into the *test* stream.
* ``class`` – one class per sequence: ``point`` or ``contextual``.
* ``num_values`` – length of the test stream.

Duplicate channel rule
----------------------
Channel ``P-2`` (SMAP) appears twice with overlapping intervals
``[5350, 6575]`` and ``[5300, 6420]``. This project treats a channel as one
telemetry stream, so duplicate rows are merged: overlapping/adjacent intervals
are unioned (``[5300, 6575]``) and the merge is recorded on the interval so the
UI can disclose it. The merged class is the shared class if all merged rows
agree, otherwise ``mixed``.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LabelInterval:
    start: int
    end: int  # inclusive
    cls: str
    merged_from: list[list[int]] = field(default_factory=list)  # non-empty when merged

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass
class ChannelLabels:
    chan_id: str
    spacecraft: str
    num_values: int
    intervals: list[LabelInterval]
    source_rows: int  # number of CSV rows that described this channel


@dataclass
class LabelFile:
    channels: dict[str, ChannelLabels]
    raw_rows: int
    raw_sequences: int
    duplicates: dict[str, int]  # chan_id -> number of rows (only when > 1)
    issues: list[str]


def _parse_classes(text: str) -> list[str]:
    inner = text.strip().strip("[]")
    return [c.strip() for c in inner.split(",") if c.strip()]


def _merge(intervals: list[LabelInterval]) -> list[LabelInterval]:
    ordered = sorted(intervals, key=lambda iv: (iv.start, iv.end))
    out: list[LabelInterval] = []
    for iv in ordered:
        if out and iv.start <= out[-1].end + 1:
            last = out[-1]
            if not last.merged_from:
                last.merged_from = [[last.start, last.end]]
            last.merged_from.append([iv.start, iv.end])
            last.end = max(last.end, iv.end)
            if last.cls != iv.cls:
                last.cls = "mixed"
        else:
            out.append(LabelInterval(iv.start, iv.end, iv.cls))
    return out


def parse_label_text(text: str) -> LabelFile:
    reader = csv.DictReader(io.StringIO(text))
    expected = {"chan_id", "spacecraft", "anomaly_sequences", "class", "num_values"}
    if reader.fieldnames is None or not expected.issubset(set(reader.fieldnames)):
        raise ValueError(f"Unexpected label header: {reader.fieldnames}")

    rows_by_chan: dict[str, list[dict[str, str]]] = {}
    raw_rows = 0
    raw_sequences = 0
    issues: list[str] = []
    for row in reader:
        raw_rows += 1
        rows_by_chan.setdefault(row["chan_id"].strip(), []).append(row)

    channels: dict[str, ChannelLabels] = {}
    duplicates: dict[str, int] = {}
    for chan_id, rows in rows_by_chan.items():
        if len(rows) > 1:
            duplicates[chan_id] = len(rows)
        spacecrafts = {r["spacecraft"].strip() for r in rows}
        num_values = {int(r["num_values"]) for r in rows}
        if len(spacecrafts) != 1:
            issues.append(f"{chan_id}: conflicting spacecraft {sorted(spacecrafts)}")
        if len(num_values) != 1:
            issues.append(f"{chan_id}: conflicting num_values {sorted(num_values)}")
        n = max(num_values)
        intervals: list[LabelInterval] = []
        for r in rows:
            seqs = json.loads(r["anomaly_sequences"])
            classes = _parse_classes(r["class"])
            raw_sequences += len(seqs)
            if len(classes) != len(seqs):
                issues.append(f"{chan_id}: {len(seqs)} sequences but {len(classes)} classes")
                classes = (classes + ["unknown"] * len(seqs))[: len(seqs)]
            for (s, e), c in zip(seqs, classes):
                s, e = int(s), int(e)
                if not (0 <= s <= e < n):
                    issues.append(f"{chan_id}: interval [{s}, {e}] outside test length {n}")
                intervals.append(LabelInterval(s, e, c))
        channels[chan_id] = ChannelLabels(
            chan_id=chan_id,
            spacecraft=sorted(spacecrafts)[0],
            num_values=n,
            intervals=_merge(intervals),
            source_rows=len(rows),
        )
    return LabelFile(channels, raw_rows, raw_sequences, duplicates, issues)


def load_label_file(path: Path) -> LabelFile:
    return parse_label_text(path.read_text(encoding="utf-8"))
