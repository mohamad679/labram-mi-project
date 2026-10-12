"""Network retry behavior for the local PhysioNet cache builder; no real downloads."""
from __future__ import annotations

from unittest.mock import Mock

import pytest
from requests.exceptions import ConnectionError as RequestsConnectionError

from scripts import prepare_portable_eeg


def test_retry_network_error_then_succeeds(monkeypatch):
    sleep = Mock()
    monkeypatch.setattr(prepare_portable_eeg.time, "sleep", sleep)
    expected = (object(), object(), object())
    load = Mock(side_effect=[RequestsConnectionError("PhysioNet timeout"), expected])
    result = prepare_portable_eeg._load_with_retry(
        7, load, max_retries=3, retry_delay=0.5,
    )
    assert result is expected
    assert load.call_count == 2
    assert load.call_args.kwargs["subjects"] == [7]
    sleep.assert_called_once_with(0.5)


def test_retry_is_bounded_and_reports_subject(monkeypatch):
    sleep = Mock()
    monkeypatch.setattr(prepare_portable_eeg.time, "sleep", sleep)
    load = Mock(side_effect=TimeoutError("socket"))
    with pytest.raises(RuntimeError, match="Subject 7: network failure after 3 attempts"):
        prepare_portable_eeg._load_with_retry(
            7, load, max_retries=2, retry_delay=2.0,
        )
    assert load.call_count == 3
    assert [c.args[0] for c in sleep.call_args_list] == [2.0, 4.0]


def test_invalid_eeg_data_error_not_swallowed(monkeypatch):
    sleep = Mock()
    monkeypatch.setattr(prepare_portable_eeg.time, "sleep", sleep)
    load = Mock(side_effect=ValueError("EEG channel mismatch"))
    with pytest.raises(ValueError, match="EEG channel mismatch"):
        prepare_portable_eeg._load_with_retry(
            7, load, max_retries=10, retry_delay=15,
        )
    assert load.call_count == 1
    sleep.assert_not_called()


def test_backoff_capped(monkeypatch):
    sleep = Mock()
    monkeypatch.setattr(prepare_portable_eeg.time, "sleep", sleep)
    load = Mock(side_effect=[RequestsConnectionError("temporary")] * 6 + ["ok"])
    assert prepare_portable_eeg._load_with_retry(
        13, load, max_retries=6, retry_delay=60,
    ) == "ok"
    assert [x.args[0] for x in sleep.call_args_list] == [60, 120, 120, 120, 120, 120]
