import json
import pickle

import pytest
from openfermion import QubitOperator

from quasisymmetries.benchmark import BenchmarkData


def sample_data(tag="sample"):
    return BenchmarkData(
        tag=tag,
        symmetries=[
            QubitOperator("X0 Y2", 0.5 - 0.25j)
            + QubitOperator((), -1.0),
            QubitOperator("Z1"),
        ],
        non_commuting_l1=1.25,
        num_commuting_terms=7,
        sym_entropy=0.3,
        cut_entropies=[0.1, 0.2],
        dmrg_bd=12,
        single_sector_e=-2.5,
    )


def assert_data_equal(actual, expected):
    assert actual.tag == expected.tag
    assert actual.symmetries == expected.symmetries
    assert actual.non_commuting_l1 == expected.non_commuting_l1
    assert actual.num_commuting_terms == expected.num_commuting_terms
    assert actual.sym_entropy == expected.sym_entropy
    assert actual.cut_entropies == expected.cut_entropies
    assert actual.dmrg_bd == expected.dmrg_bd
    assert actual.single_sector_e == expected.single_sector_e


def test_json_single_round_trip(tmp_path):
    expected = sample_data()

    path = expected.save(tmp_path / "benchmark")

    assert path.suffix == ".json"
    payload = json.loads(path.read_text())
    assert payload["schema"] == "quasisymmetries.BenchmarkData"
    assert payload["version"] == 1
    assert_data_equal(BenchmarkData.load(tmp_path / "benchmark"), expected)


def test_json_collection_preserves_duplicate_tags(tmp_path):
    expected = [sample_data("same"), sample_data("same")]

    BenchmarkData.save_datasets(expected, tmp_path / "benchmarks")
    actual = BenchmarkData.load_datasets(tmp_path / "benchmarks")

    assert len(actual) == 2
    for actual_item, expected_item in zip(actual, expected):
        assert_data_equal(actual_item, expected_item)


def test_pickle_suffix_is_migrated_to_json(tmp_path):
    expected = sample_data()

    path = expected.save(tmp_path / "benchmark.pkl")

    assert path == tmp_path / "benchmark.json"
    assert_data_equal(BenchmarkData.load(tmp_path / "benchmark.pkl"), expected)


def test_legacy_pickle_remains_readable_with_warning(tmp_path):
    expected = sample_data()
    path = tmp_path / "legacy.pkl"
    with path.open("wb") as file_obj:
        pickle.dump(
            {
                "__type__": "BenchmarkData.datasets",
                "datasets": [expected.to_dict()],
            },
            file_obj,
        )

    with pytest.warns(UserWarning, match="Only load pickle files you trust"):
        actual = BenchmarkData.load_datasets(path)

    assert_data_equal(actual[0], expected)
