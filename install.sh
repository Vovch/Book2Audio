#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$ROOT"

MODE="${1:-}"
SPEC="."
if [[ "${MODE}" == "amd" ]]; then
  case "$(uname -s)" in
    CYGWIN* | MINGW* | MSYS*) ;;
    *)
      echo "ERROR: The amd profile (torch-directml) is only for native Windows." >&2
      echo "       On this OS run: ./install.sh" >&2
      exit 1
      ;;
  esac
  SPEC='.[amd]'
fi

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  echo "ERROR: python3 or python not found. Install Python 3.10+." >&2
  exit 1
fi

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Creating virtual environment..."
  "${PYTHON}" -m venv .venv
fi

echo "Upgrading pip..."
.venv/bin/python -m pip install -U pip

echo "Installing Book2Audio (${SPEC})..."
.venv/bin/pip install "${SPEC}"

echo ""
echo "Done. Activate:  source .venv/bin/activate"
echo "Then run:       book2audio-tts"
