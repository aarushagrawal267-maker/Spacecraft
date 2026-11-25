"""Dataset downloads must not forward Kaggle credentials to a redirect on another host."""

import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from app.dataset import acquire


def _serve(handler_cls):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture
def servers(monkeypatch):
    seen: dict[str, str | None] = {}

    class Storage(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            seen[self.path] = self.headers.get("Authorization")
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"payload")

        def log_message(self, *a):
            pass

    storage = _serve(Storage)
    other_host = f"http://localhost:{storage.server_port}"  # different netloc from 127.0.0.1:<port>

    class Api(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path == "/cross":
                target = f"{other_host}/signed"
            elif self.path == "/same":
                target = "/final"
            else:
                seen[self.path] = self.headers.get("Authorization")
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"payload")
                return
            self.send_response(302)
            self.send_header("Location", target)
            self.end_headers()

        def log_message(self, *a):
            pass

    api = _serve(Api)
    # Talk to the local servers directly, never through an environment proxy.
    monkeypatch.setattr(acquire, "_OPENER", urllib.request.build_opener(
        urllib.request.ProxyHandler({}), acquire._StripAuthOnCrossHostRedirect))
    yield f"http://127.0.0.1:{api.server_port}", seen
    api.shutdown()
    storage.shutdown()


def test_authorization_dropped_on_cross_host_redirect(servers, tmp_path):
    base, seen = servers
    acquire._download(f"{base}/cross", tmp_path / "a.zip", retries=1, headers={"Authorization": "Basic c2VjcmV0"})
    assert (tmp_path / "a.zip").read_bytes() == b"payload"
    assert seen == {"/signed": None}


def test_authorization_kept_on_same_host_redirect(servers, tmp_path):
    base, seen = servers
    acquire._download(f"{base}/same", tmp_path / "b.zip", retries=1, headers={"Authorization": "Basic c2VjcmV0"})
    assert seen == {"/final": "Basic c2VjcmV0"}
