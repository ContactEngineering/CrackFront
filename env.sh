#!/bin/bash

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
[ -d "$ROOT/venv" ] && source $ROOT/venv/bin/activate
export PYTHONPATH="$ROOT:$PYTHONPATH"