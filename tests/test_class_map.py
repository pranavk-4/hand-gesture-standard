"""Guards the 18->6 class-remap contract (the highest-risk change vs the
research pipeline). No network, no GPU — pure index logic.
"""

from src.data.download_dataset import FULL_18CLASS_NAMES, build_class_index_map
from src.config import STANDARD_CLASSES


def test_keep_maps_to_distinct_new_indices():
    m = build_class_index_map(STANDARD_CLASSES)
    assert len(m) == 6
    assert sorted(m.values()) == [0, 1, 2, 3, 4, 5]


def test_remap_resolves_through_full_names():
    m = build_class_index_map(STANDARD_CLASSES)
    for src_idx, new_idx in m.items():
        assert FULL_18CLASS_NAMES[src_idx] == STANDARD_CLASSES[new_idx]


def test_excluded_classes_are_skipped():
    m = build_class_index_map(STANDARD_CLASSES)
    excluded = [i for i, n in enumerate(FULL_18CLASS_NAMES) if n not in STANDARD_CLASSES]
    assert len(excluded) == 12
    assert all(i not in m for i in excluded)


def test_standard_classes_sorted_for_imagefolder():
    assert STANDARD_CLASSES == sorted(STANDARD_CLASSES)
