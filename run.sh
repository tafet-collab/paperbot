#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source .venv/bin/activate || true
python bot.py paper --symbol ${SYMBOL:-BTC-USD} --start ${START_EQUITY:-10} >> paper.log 2>&1
