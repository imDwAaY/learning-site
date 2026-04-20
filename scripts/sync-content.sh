#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p "$ROOT/content"
mkdir -p "$ROOT/content/notes"
mkdir -p "$ROOT/content/imdwaay-learning"

rm -rf "$ROOT/content/notes"
rm -rf "$ROOT/content/imdwaay-learning"

mkdir -p "$ROOT/content/notes"
mkdir -p "$ROOT/content/imdwaay-learning"

rsync -a \
  --delete \
  --exclude ".git" \
  --exclude ".obsidian" \
  "$ROOT/sources/yeezus/notes/" \
  "$ROOT/content/notes/"

rsync -a \
  --delete \
  --exclude ".git" \
  --exclude ".obsidian" \
  "$ROOT/sources/imdwaay-learning/" \
  "$ROOT/content/imdwaay-learning/"