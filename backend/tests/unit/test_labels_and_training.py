import numpy as np

from app.dataset.labels import parse_label_text
from app.ml.training import chronological_split

CSV = """chan_id,spacecraft,anomaly_sequences,class,num_values
P-1,SMAP,"[[2149, 2349], [4536, 4844], [3539, 3779]]","[contextual, contextual, contextual]",8505
P-2,SMAP,"[[5350, 6575]]",[point],8209
P-2,SMAP,"[[5300, 6420]]",[point],8209
T-9,MSL,"[[200, 300], [301, 310]]","[point, contextual]",1096
"""


def test_parse_and_sort_intervals():
    lf = parse_label_text(CSV)
    assert lf.raw_rows == 4 and lf.raw_sequences == 7
    p1 = lf.channels["P-1"]
    assert [(i.start, i.end) for i in p1.intervals] == [(2149, 2349), (3539, 3779), (4536, 4844)]
    assert all(i.cls == "contextual" for i in p1.intervals)
    assert p1.spacecraft == "SMAP" and p1.num_values == 8505


def test_duplicate_channel_rows_are_merged_and_disclosed():
    lf = parse_label_text(CSV)
    assert lf.duplicates == {"P-2": 2}
    p2 = lf.channels["P-2"]
    assert p2.source_rows == 2
    assert [(i.start, i.end) for i in p2.intervals] == [(5300, 6575)]
    assert p2.intervals[0].merged_from == [[5300, 6420], [5350, 6575]]
    assert p2.intervals[0].cls == "point"


def test_adjacent_intervals_merge_with_mixed_class():
    t9 = parse_label_text(CSV).channels["T-9"]
    assert [(i.start, i.end, i.cls) for i in t9.intervals] == [(200, 310, "mixed")]


def test_out_of_range_interval_reported():
    bad = "chan_id,spacecraft,anomaly_sequences,class,num_values\nX-1,SMAP,\"[[10, 50]]\",[point],40\n"
    assert any("outside" in i for i in parse_label_text(bad).issues)


def test_chronological_split_is_ordered_with_gap():
    tr, va = chronological_split(1000, 0.2, gap=50)
    assert va.min() > tr.max()
    assert va.min() - tr.max() - 1 == 50
    assert len(va) == 200
    tr, va = chronological_split(40, 0.2, gap=100, min_train=32)
    assert len(tr) == 32 and len(va) == 8  # gap shrinks for short series
    assert np.intersect1d(tr, va).size == 0
