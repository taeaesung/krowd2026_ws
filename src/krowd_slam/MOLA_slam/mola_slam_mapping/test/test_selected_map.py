import pytest

from mola_slam_mapping.selected_map_node import resolved_map_dir


def test_a_run_directory_resolves_to_an_absolute_path(tmp_path):
    run = tmp_path / "20260914_220126"
    run.mkdir()
    link = tmp_path / "latest"
    link.symlink_to(run)

    # `latest` moves with every mapping run, so consumers are told the real run.
    assert resolved_map_dir(str(link)) == str(run.resolve())
    assert resolved_map_dir(f" {run} ") == str(run.resolve())


def test_an_unusable_map_dir_is_rejected(tmp_path):
    for bad in ("", "   ", str(tmp_path / "missing"), str(tmp_path / "file.txt")):
        with pytest.raises(ValueError):
            resolved_map_dir(bad)
