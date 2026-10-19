"""The configured model table is validated and adapters receive their server URL."""

import pytest

from mystery_dungeon.models import CONFIG, ModelEntry, load_model, read_models


class UrlModel:
    def __init__(self, url=None):
        self.url = url
        self.last_response = None

    def choose_action(self, task, knowledge, frame, memory):
        raise NotImplementedError

    def is_complete(self, task, frame, memory):
        raise NotImplementedError


def write(tmp_path, text):
    path = tmp_path / "config.toml"
    path.write_text(text)
    return path


def test_entries_keep_file_order_and_fields(tmp_path):
    path = write(tmp_path, """
[models.local]
label = "Local"
spec = "a:B"
serve = ["llama-server", "--port", "{port}"]
ready = "/health"

[models.remote]
label = "Remote"
spec = "c:D"
url = "http://gpu:8080"
""")
    assert read_models(path) == {
        "local": ModelEntry("local", "Local", "a:B",
                            ("llama-server", "--port", "{port}"), "/health"),
        "remote": ModelEntry("remote", "Remote", "c:D", url="http://gpu:8080"),
    }


@pytest.mark.parametrize("table,error", [
    ('label = "X"', TypeError),
    ('label = "X"\nspec = "a:B"\nport = 1', TypeError),
    ('label = "X"\nspec = "a:B"\nserve = ["s"]\nready = "/r"\nurl = "http://h"', ValueError),
    ('label = "X"\nspec = "a:B"\nserve = ["s"]', ValueError),
    ('label = "X"\nspec = "a:B"\nserve = ["s"]\nready = "health"', ValueError),
    ('label = "X"\nspec = "a:B"\nserve = "llama-server --port {port}"\nready = "/r"', ValueError),
    ('label = "X"\nspec = "a:B"\nfiles = "w.gguf"', ValueError),
])
def test_invalid_entries_raise(tmp_path, table, error):
    with pytest.raises(error):
        read_models(write(tmp_path, f"[models.bad]\n{table}\n"))


def test_config_without_models_raises(tmp_path):
    with pytest.raises(ValueError, match="no \\[models.NAME\\]"):
        read_models(write(tmp_path, "[game]\nrom = 'x'\n"))


def test_project_config_is_valid():
    assert next(iter(read_models(CONFIG))) == "onejev"


def test_load_model_passes_the_server_url():
    assert load_model(f"{__name__}:UrlModel", "http://127.0.0.1:9000").url == "http://127.0.0.1:9000"
    assert load_model(f"{__name__}:UrlModel").url is None
