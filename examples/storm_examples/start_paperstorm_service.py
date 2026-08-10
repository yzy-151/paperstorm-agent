"""Start the PaperStorm FastAPI service for local demos."""

import argparse
import os
import socket
import sys
from pathlib import Path


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def build_parser():
    parser = argparse.ArgumentParser(description="Start PaperStorm Agent service.")
    parser.add_argument("--service-root", default="./results/paperstorm_demo_service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None, help="默认 8002；被占用时自动顺延")
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--log-level", default="info")
    return parser


def print_demo_runbook(args):
    service_url = "http://{0}:{1}".format(args.host, args.port)
    print("PaperStorm service URL: {0}".format(service_url))
    print("Dashboard file: frontend/paperstorm_dashboard/index.html")
    print("Demo lifecycle: submit -> queued -> running -> succeeded/failed")
    print("Use run_mode=fake for a no-key local demo; use paperstorm for real LLM runs.")


def _port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) != 0


def preflight(args):
    """Check dependencies, port availability and optional API keys with actionable hints."""
    try:
        import uvicorn  # noqa: F401
    except ImportError:
        print("ERROR: uvicorn is not installed.")
        print("Fix: pip install uvicorn")
        raise SystemExit(1)
    if args.port is None:
        port = 8002
        if not _port_is_free(args.host, port):
            next_port = port + 1
            print(
                "WARNING: port {0} is already in use; suggesting {1} "
                "(use --port to override).".format(port, next_port)
            )
            while not _port_is_free(args.host, next_port):
                next_port += 1
            port = next_port
        args.port = port
        print("Using port: {0}".format(port))
    elif not _port_is_free(args.host, args.port):
        print(
            "ERROR: port {0} is already in use; try --port {1}.".format(
                args.port, args.port + 1
            )
        )
        raise SystemExit(1)
    if not (os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("MINIMAX_API_KEY")):
        print(
            "NOTE: no DEEPSEEK_API_KEY / MINIMAX_API_KEY found. "
            "fake mode works without a key; real retrieval/LLM mode needs one. "
            "Fix: set DEEPSEEK_API_KEY=... or MINIMAX_API_KEY=... (see README)."
        )


def main():
    args = build_parser().parse_args()
    preflight(args)
    print_demo_runbook(args)
    import uvicorn

    from examples.storm_examples.paperstorm_service_api import create_app

    uvicorn.run(
        create_app(service_root=Path(args.service_root)),
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
