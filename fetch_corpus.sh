#!/usr/bin/env bash
# Re-fetch the census corpus. Public domain, Project Gutenberg.
set -euo pipefail
cd "$(dirname "$0")"
curl -sSL -o corpus_raw.txt "https://www.gutenberg.org/cache/epub/11/pg11.txt"
echo "corpus_raw.txt: $(wc -c < corpus_raw.txt) bytes"
echo "sha256: $(sha256sum corpus_raw.txt | cut -d' ' -f1)"
