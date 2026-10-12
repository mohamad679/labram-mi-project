"""Offline unit tests for Kaggle public-S3 imagery EDF downloader."""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.prepare_physionet_s3 import RUNS, edf_path, fetch_run, s3_key


def test_public_s3_key_and_exact_six_imagery_runs():
    assert RUNS == (4, 6, 8, 10, 12, 14)
    assert s3_key(7, 14) == "eegmmidb/1.0.0/S007/S007R14.edf"
    assert edf_path(Path("/tmp/data"), 7, 14).as_posix().endswith(
        "/MNE-eegbci-data/files/eegmmidb/1.0.0/S007/S007R14.edf"
    )


def test_s3_download_skips_already_complete_raw_edf(tmp_path):
    class FakeS3:
        calls = 0

        def head_object(self, **kwargs):
            assert kwargs["Bucket"] == "physionet-open"
            assert kwargs["Key"] == s3_key(1, 14)
            return {"ContentLength": 3}

        def download_file(self, bucket, key, destination):
            assert bucket == "physionet-open"
            assert key == s3_key(1, 14)
            self.calls += 1
            Path(destination).write_bytes(b"EDF")

    client = FakeS3()
    assert fetch_run(client, 1, 14, tmp_path)[0] == 3
    assert fetch_run(client, 1, 14, tmp_path) == (3, 0.0)
    assert client.calls == 1


def test_s3_does_not_commit_partial_transfer(tmp_path):
    class FakeS3:
        def head_object(self, **kwargs):
            return {"ContentLength": 10}

        def download_file(self, bucket, key, destination):
            Path(destination).write_bytes(b"partial")

    with pytest.raises(OSError, match="Incomplete S3"):
        fetch_run(FakeS3(), 2, 4, tmp_path)
    path = edf_path(tmp_path, 2, 4)
    assert not path.exists()
    assert not path.with_suffix(".edf.partial").exists()
