from pathlib import Path
import tempfile

from openfermion import QubitOperator

from quasisymmetries import mpo, tn


class FakeDriver:
    def __init__(self, scratch, **kwargs):
        self.scratch = scratch
        self.kwargs = kwargs


def test_owned_block2_scratch_is_unique_and_cleaned(monkeypatch):
    monkeypatch.setattr(tn, "DMRGDriver", FakeDriver)

    first = tn._create_block2_driver(scratch_prefix="qs_test_")
    second = tn._create_block2_driver(scratch_prefix="qs_test_")
    first_path = Path(first.scratch)
    second_path = Path(second.scratch)

    assert first_path.exists()
    assert second_path.exists()
    assert first_path != second_path
    assert first_path.name.startswith("qs_test_")

    (first_path / "block2-output").write_text("temporary")
    assert tn.cleanup_block2_driver(first)
    assert not first_path.exists()
    assert not tn.cleanup_block2_driver(first)

    assert tn.cleanup_block2_driver(second)
    assert not second_path.exists()


def test_caller_supplied_block2_scratch_is_preserved(monkeypatch, tmp_path):
    monkeypatch.setattr(tn, "DMRGDriver", FakeDriver)
    scratch = tmp_path / "caller-owned"

    driver = tn._create_block2_driver(scratch=scratch)
    (scratch / "keep-me").write_text("persistent")

    assert not tn.cleanup_block2_driver(driver)
    assert scratch.exists()
    assert (scratch / "keep-me").exists()


def test_qc_mpo_result_cleanup_only_removes_owned_scratch(tmp_path):
    scratch_obj = tempfile.TemporaryDirectory(
        prefix="qs_mpo_test_",
        dir=tmp_path,
    )
    scratch_path = Path(scratch_obj.name)
    result = {"_scratch_obj": scratch_obj}

    assert mpo.cleanup_qc_mpo_result(result)
    assert result["_scratch_obj"] is None
    assert not scratch_path.exists()
    assert not mpo.cleanup_qc_mpo_result(result)

    caller_scratch = tmp_path / "caller-mpo"
    caller_scratch.mkdir()
    caller_result = {"_scratch_obj": None}
    assert not mpo.cleanup_qc_mpo_result(caller_result)
    assert caller_scratch.exists()


def test_block2_dmrg_convergence_helper_cleans_scratch(monkeypatch):
    scratch_obj = tempfile.TemporaryDirectory(prefix="qs_dmrg_test_")
    scratch_path = Path(scratch_obj.name)

    class CalculationDriver:
        symm_type = "fake"
        _quasisymmetries_scratch_obj = scratch_obj

        def get_random_mps(self, **kwargs):
            return object()

        def dmrg(self, *args, **kwargs):
            return -1.0

    driver = CalculationDriver()
    monkeypatch.setattr(tn, "has_complex_entries", lambda operator: False)
    monkeypatch.setattr(
        tn,
        "QO_to_block2_MPO",
        lambda operator, n_qubits: (object(), driver),
    )

    result = tn.find_dmrg_conv_bd(
        QubitOperator("Z0"),
        n_qubits=1,
        exact_energy=-1.0,
        max_bd=1,
    )

    assert result == 1
    assert not scratch_path.exists()
    assert driver._quasisymmetries_scratch_obj is None
