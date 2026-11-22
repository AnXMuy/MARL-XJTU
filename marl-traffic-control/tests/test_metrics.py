from traffic_signal_marl.evaluation.episode import _tripinfo_metrics


def test_no_completed_vehicles_is_not_zero_wait(tmp_path):
    file = tmp_path / "trip.xml"
    file.write_text("<tripinfos/>")
    result = _tripinfo_metrics(file, 10)
    assert result["unfinished_vehicles"] == 10
    assert result["completed_mean_waiting_time_seconds"] is None


def test_completed_metrics_and_censoring(tmp_path):
    file = tmp_path / "trip.xml"
    file.write_text('<tripinfos><tripinfo waitingTime="12" duration="100" timeLoss="20"/></tripinfos>')
    result = _tripinfo_metrics(file, 2)
    assert result["completed_mean_waiting_time_seconds"] == 12
    assert result["unfinished_vehicles"] == 1
