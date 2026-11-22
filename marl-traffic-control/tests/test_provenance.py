from traffic_signal_marl.provenance import git_metadata


def test_source_archive_has_no_git_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert git_metadata(tmp_path) == (None, None)
