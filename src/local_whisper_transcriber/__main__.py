from __future__ import annotations

import sys


def _is_self_test_requested() -> bool:
    return any("self-test" in arg.strip().strip("\"'").lower() for arg in sys.argv[1:])


def main() -> int:
    if _is_self_test_requested():
        from local_whisper_transcriber.selftest import run_self_test

        return run_self_test()

    from local_whisper_transcriber.gui import main as run_gui

    return run_gui()


if __name__ == "__main__":
    raise SystemExit(main())
