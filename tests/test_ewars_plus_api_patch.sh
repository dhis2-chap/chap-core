#!/bin/bash
# Smoke test for the ewars_plus_api CLIM-617 patch.
#
# Builds the patched image and runs the in-image regression test that
# exercises the per-year-lag bind_rows assembly fix on synthetic fixtures.
# Independent of compose; safe to run in CI.

set -e

repo_root=$(cd "$(dirname "$0")/.." && pwd)

echo "=== Building chap-core/ewars_plus_api:clim-617 ==="
docker build \
  -t chap-core/ewars_plus_api:clim-617 \
  "$repo_root/external_models/ewars_plus_api_patch"

echo "=== Running in-image regression test (test_bind_rows_assembly.R) ==="
docker run --rm \
  chap-core/ewars_plus_api:clim-617 \
  Rscript /home/app/tests/test_bind_rows_assembly.R

echo "=== ewars_plus_api patch smoke test PASSED ==="
