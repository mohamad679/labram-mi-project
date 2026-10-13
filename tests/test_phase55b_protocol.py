"""Adversarial Phase 5.5b final-Validation gate tests (synthetic indices only)."""

import numpy as np
import pytest

from src.phase55b_protocol import (
    HISTORICAL_TEST_SUBJECTS,
    HISTORICAL_VALIDATION_SUBJECTS,
    INNER_EARLY_SUBJECTS,
    PermittedEpochSource,
    make_frozen_final_split,
    verify_train_only_nomination,
)


def fake_metadata():
    eligible = [i for i in range(1, 110) if i not in (88, 89, 92, 100, 104)]
    subjects = np.repeat(np.asarray(eligible), 174)
    assert len(subjects) == 18096
    val = set(HISTORICAL_VALIDATION_SUBJECTS)
    test = set(HISTORICAL_TEST_SUBJECTS)
    rows = np.arange(len(subjects))
    masks = {"train": ~np.isin(subjects, list(val | test)),
             "validation": np.isin(subjects, list(val)),
             "test": np.isin(subjects, list(test))}
    return subjects, {k: rows[v] for k, v in masks.items()}


def accepted_evidence():
    return {"selection": {"selected": "eegnet_epoch_standardized"},
            "phase55a_gate": "PASS_FOR_COMPLETED_TRAIN_ONLY_CANDIDATE_SELECTION_WITH_DECLARED_LIMITS",
            "dataset": {"train_subjects": 75, "validation_or_test_predictions_in_zip": 0}}


def test_frozen_67_fit_8_early_14_validation_15_test_partition():
    ids, p = fake_metadata()
    out = make_frozen_final_split(p, ids)
    assert len(out["fit_subjects"]) == 67
    assert tuple(out["early_subjects"]) == INNER_EARLY_SUBJECTS
    assert len(out["fit"]) + len(out["early"]) == 13050
    assert len(np.intersect1d(out["fit"], out["early"])) == 0
    assert len(out["validation"]) == 2436
    assert len(out["test"]) == 2610
    assert set(ids[out["fit"]]).isdisjoint(ids[out["validation"]])
    assert set(ids[out["early"]]).isdisjoint(ids[out["test"]])


def test_reject_unexpected_validation_participant():
    ids, p = fake_metadata()
    idx = p["validation"][0]
    ids[idx] = 3
    with pytest.raises((ValueError, RuntimeError)):
        make_frozen_final_split(p, ids)


def test_reject_unexpected_test_participant():
    ids, p = fake_metadata()
    ids[p["test"][0]] = 2
    with pytest.raises((ValueError, RuntimeError)):
        make_frozen_final_split(p, ids)


def test_reject_duplicate_train_index():
    ids, p = fake_metadata()
    p["train"][1] = p["train"][0]
    with pytest.raises(ValueError, match="sorted and unique"):
        make_frozen_final_split(p, ids)


def test_train_source_denies_both_heldouts_before_underlying_read():
    _, p = fake_metadata()
    backing = {int(p["train"][0]): "valid"}
    guard = PermittedEpochSource(backing, p["train"], "train")
    assert guard[p["train"][0]] == "valid"
    with pytest.raises(PermissionError):
        guard[p["validation"][0]]
    with pytest.raises(PermissionError):
        guard[p["test"][0]]


def test_validation_source_denies_test_and_train():
    _, p = fake_metadata()
    backing = {int(p["validation"][0]): "allowed"}
    guard = PermittedEpochSource(backing, p["validation"], "validation")
    assert guard[p["validation"][0]] == "allowed"
    with pytest.raises(PermissionError):
        guard[p["test"][0]]
    with pytest.raises(PermissionError):
        guard[p["train"][0]]


@pytest.mark.parametrize("value", [True, False, [2], np.asarray([2])])
def test_source_refuses_non_scalar_or_boolean_index(value):
    _, p = fake_metadata()
    guard = PermittedEpochSource({}, p["train"], "train")
    with pytest.raises(PermissionError):
        guard[value]


def test_test_role_prohibited():
    _, p = fake_metadata()
    with pytest.raises(ValueError, match="Only train or historical validation"):
        PermittedEpochSource({}, p["test"], "test")


def test_wrong_role_row_count_rejected():
    _, p = fake_metadata()
    with pytest.raises(ValueError, match="expected exactly"):
        PermittedEpochSource({}, p["train"][:-1], "train")


def test_accept_precommitted_normalized_eegnet_nomination():
    verify_train_only_nomination(accepted_evidence())


def test_reject_alternative_architecture_from_heldout_fishing():
    proof = accepted_evidence()
    proof["selection"]["selected"] = "eegnex_epoch_standardized"
    with pytest.raises(ValueError, match="nominated"):
        verify_train_only_nomination(proof)


def test_reject_tainted_nomination_or_wrong_gate():
    for key, bad_value in [("train_subjects", 74),
                           ("validation_or_test_predictions_in_zip", 2)]:
        proof = accepted_evidence()
        proof["dataset"][key] = bad_value
        with pytest.raises(ValueError):
            verify_train_only_nomination(proof)
    proof = accepted_evidence()
    proof["phase55a_gate"] = "PENDING"
    with pytest.raises(ValueError):
        verify_train_only_nomination(proof)
