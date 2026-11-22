import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("train_script", ROOT / "scripts/train.py")
TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN)


def test_checkpoint_resume_matches_uninterrupted_run(tmp_path, monkeypatch):
    config = json.loads((ROOT / "configs/train.json").read_text())
    config.update(episode_seconds=20, episodes=2, device="cpu")
    route = tmp_path / "empty.rou.xml"
    route.write_text("<routes/>")
    reference = TRAIN.train(config, [route], tmp_path / "reference")
    original = TRAIN.update
    calls = []

    def interrupt(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise InterruptedError("test interruption")
        return original(*args, **kwargs)

    output = tmp_path / "interrupted"
    with monkeypatch.context() as patch:
        patch.setattr(TRAIN, "update", interrupt)
        with pytest.raises(InterruptedError):
            TRAIN.train(config, [route], output)
    with (output / "train.jsonl").open("a") as log:
        log.write('{"episode": 1}\n')
    resumed = TRAIN.train(config, [route], output, resume=True)
    expected = torch.load(reference, weights_only=False)
    actual = torch.load(resumed, weights_only=False)
    assert actual["episode"] == expected["episode"] == 1
    for key, weights in expected["model"].items():
        assert torch.equal(actual["model"][key], weights)
    assert [json.loads(line)["episode"] for line in (output / "train.jsonl").read_text().splitlines()] == [0, 1]
