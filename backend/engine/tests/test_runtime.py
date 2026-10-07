import json

from netpattern_engine import __version__
from netpattern_engine.runtime import main


def _run(tmp_path, capsys, request):
    request_file = tmp_path / "request.json"
    result_file = tmp_path / "result.json"
    request_file.write_text(request if isinstance(request, str) else json.dumps(request))

    exit_code = main(["job", "--request", str(request_file), "--output", str(result_file)])

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    result = json.loads(result_file.read_text())
    return exit_code, events, result


def test_smoke_job_streams_events_and_writes_result(tmp_path, capsys):
    exit_code, events, result = _run(tmp_path, capsys, {"run_id": "run_1", "kind": "smoke"})

    assert exit_code == 0
    assert [event["type"] for event in events] == ["run_started", "env_info", "run_completed"]
    assert events[0]["run_id"] == events[-1]["run_id"] == "run_1"
    assert result["status"] == "completed"
    assert result["environment"]["engine"] == __version__


def test_unknown_job_kind_fails_with_result(tmp_path, capsys):
    exit_code, events, result = _run(tmp_path, capsys, {"run_id": "run_2", "kind": "nonsense"})

    assert exit_code == 1
    assert events[-1]["type"] == "run_failed"
    assert "unsupported job kind" in events[-1]["error"]
    assert result == {"run_id": "run_2", "status": "failed", "error": events[-1]["error"]}


def test_cancel_token_follows_the_request_file(tmp_path):
    from netpattern_engine.training.trainer import CancelToken

    token = CancelToken(tmp_path / "cancel.request")
    assert not token.cancelled

    (tmp_path / "cancel.request").touch()

    assert token.cancelled


def test_invalid_request_json_fails_with_result(tmp_path, capsys):
    exit_code, events, result = _run(tmp_path, capsys, "{not json")

    assert exit_code == 1
    assert [event["type"] for event in events] == ["run_failed"]
    assert result["status"] == "failed"
