from __future__ import annotations

import os
import subprocess
import sys

from dotenv import load_dotenv


def main() -> int:
    load_dotenv()
    token = os.environ.get("JUPYTER_TOKEN", "text-detection-local-token")
    port = os.environ.get("JUPYTER_PORT", "8888")
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "jupyter",
            "lab",
            f"--IdentityProvider.token={token}",
            f"--ServerApp.port={port}",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
