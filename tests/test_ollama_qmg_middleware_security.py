"""Regression tests for safe upstream error responses in the Ollama example."""

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

_MISSING_DEPENDENCIES = [
    name for name in ("flask", "requests") if importlib.util.find_spec(name) is None
]
if _MISSING_DEPENDENCIES:
    raise unittest.SkipTest(
        "Optional example dependencies unavailable: "
        + ", ".join(_MISSING_DEPENDENCIES)
    )


_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "ollama_qmg_middleware.py"
)
_SPEC = importlib.util.spec_from_file_location("ollama_qmg_middleware", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load {_MODULE_PATH}")
_middleware = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_middleware)


class TestUpstreamErrorResponses(unittest.TestCase):
    """The public API must not echo upstream exception details."""

    @classmethod
    def setUpClass(cls):
        setattr(_middleware, "QMG_AVAILABLE", False)
        cls.client = _middleware.app.test_client()

    def _assert_safe_error(self, stream):
        sensitive_detail = (
            "upstream internal URL http://10.0.0.8:11434/api/chat "
            "with Authorization=Bearer secret-value"
        )
        with patch.object(
            _middleware.requests,
            "post",
            side_effect=_middleware.requests.exceptions.RequestException(
                sensitive_detail
            ),
        ):
            response = self.client.post(
                "/api/chat",
                json={
                    "stream": stream,
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.get_json(), {"error": "Ollama request failed"})
        self.assertNotIn(sensitive_detail, response.get_data(as_text=True))
        self.assertNotIn("secret-value", response.get_data(as_text=True))

    def test_streaming_upstream_error_is_generic(self):
        self._assert_safe_error(stream=True)

    def test_non_streaming_upstream_error_is_generic(self):
        self._assert_safe_error(stream=False)


if __name__ == "__main__":
    unittest.main()
