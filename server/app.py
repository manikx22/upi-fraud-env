"""OpenEnv server package entrypoint for multi-mode deployment validators."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_ROOT_SERVER_PATH = Path(__file__).resolve().parents[1] / "server.py"
_SPEC = spec_from_file_location("upi_fraud_root_server", _ROOT_SERVER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Unable to load server module from {_ROOT_SERVER_PATH}")

_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

app = _MODULE.app


def main() -> None:
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

