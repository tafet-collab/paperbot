#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
python bot.py paper --symbol BTC-USD --start 10 >> paper.log 2>&1
