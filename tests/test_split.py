import pytest

from credit_underwriting.data.schema import ID, TARGET
from credit_underwriting.split import split_applications


def test_split_sizes_and_no_overlap(applications):
    split = split_applications(applications, valid_size=0.2, test_size=0.2)
    sizes = [len(part) for part in split]
    assert sum(sizes) == len(applications)
    assert sizes[1] == pytest.approx(0.2 * len(applications), abs=2)
    assert sizes[2] == pytest.approx(0.2 * len(applications), abs=2)

    ids = [set(part[ID]) for part in split]
    assert not ids[0] & ids[1]
    assert not ids[0] & ids[2]
    assert not ids[1] & ids[2]


def test_split_is_stratified(applications):
    split = split_applications(applications)
    overall = applications[TARGET].mean()
    for part in split:
        assert part[TARGET].mean() == pytest.approx(overall, abs=0.01)
