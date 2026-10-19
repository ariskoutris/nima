"""Model servers run the configured command on a free port and stop with their process group."""

import sys
from contextlib import contextmanager

from mystery_dungeon.models import ModelEntry
from mystery_dungeon.models import server as model_server

SERVED = ModelEntry("local", "Local", "fake:Model",
                    ("{python}", "-m", "qev.cli", "serve", "--port", "{port}"), "/v1/models")


def fake_process(monkeypatch, calls):
    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def bind(self, address):
            assert address == ("127.0.0.1", 0)

        def getsockname(self):
            return ("127.0.0.1", 9000)

    class Process:
        pid = 1234

        def poll(self):
            return None

        def wait(self, timeout=None):
            assert timeout == 10

    @contextmanager
    def ready(url, timeout):
        calls.append(url)
        yield

    monkeypatch.setattr(model_server.socket, "socket", Socket)
    monkeypatch.setattr(model_server.subprocess, "Popen",
                        lambda args, **kwargs: calls.append((args, kwargs)) or Process())
    monkeypatch.setattr(model_server, "urlopen", ready)
    monkeypatch.setattr(model_server.os, "killpg", lambda pid, sig: calls.append((pid, sig)))


def test_server_runs_the_filled_in_command_in_its_own_process_group(tmp_path, monkeypatch):
    calls = []
    fake_process(monkeypatch, calls)
    server = model_server.ModelServer(tmp_path)
    assert server.start(SERVED) == "http://127.0.0.1:9000"
    assert calls[0][0] == [sys.executable, "-m", "qev.cli", "serve", "--port", "9000"]
    assert calls[0][1]["start_new_session"] is True
    assert calls[0][1]["cwd"] == model_server.CONFIG.parent
    assert calls[1] == "http://127.0.0.1:9000/v1/models"
    assert server.start(SERVED) == "http://127.0.0.1:9000" and len(calls) == 2
    server.close()
    assert calls[-1] == (1234, model_server.signal.SIGTERM)


def test_starting_another_entry_replaces_the_running_server(tmp_path, monkeypatch):
    calls = []
    fake_process(monkeypatch, calls)
    server = model_server.ModelServer(tmp_path)
    server.start(SERVED)
    server.start(ModelEntry("other", "Other", "fake:Model", ("llama-server", "{port}"), "/health"))
    assert (1234, model_server.signal.SIGTERM) in calls
    assert calls[-2][0] == ["llama-server", "9000"] and calls[-1].endswith("/health")


def test_an_entry_without_a_command_gets_its_url_and_leaves_the_server_running(tmp_path,
                                                                                monkeypatch):
    calls = []
    fake_process(monkeypatch, calls)
    server = model_server.ModelServer(tmp_path)
    server.start(SERVED)
    assert server.start(ModelEntry("hosted", "Hosted", "fake:Model", url="http://gpu:8080")) == "http://gpu:8080"
    assert not any(call == (1234, model_server.signal.SIGTERM) for call in calls)


def test_only_port_and_python_placeholders_are_filled_in():
    entry = ModelEntry("x", "X", "a:B",
                       ("{python}", "--port={port}", '{"enable_thinking": false}'), "/r")
    assert model_server.serve_command(entry, 9000) == [
        sys.executable, "--port=9000", '{"enable_thinking": false}']

