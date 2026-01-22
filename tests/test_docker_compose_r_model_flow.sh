#!/bin/bash
# This file runs the R model integration test via Docker Compose.
# It registers the minimalist_example_r model template via REST API,
# creates a configured model, and runs a backtest evaluation.

set -e

echo "=== R Model Integration Test ==="

echo "=== Cleaning up previous containers ==="
docker compose down

echo "=== Starting Docker Compose stack ==="
docker compose -f compose.yml -f compose.r-model.integration.test.yml up --build --detach --force-recreate

echo "=== Postgres initialization logs (immediate) ==="
docker compose logs postgres

echo "=== Waiting for containers to initialize ==="
sleep 15

echo "=== Container status ==="
docker compose ps

echo "=== Postgres logs after waiting ==="
docker compose logs postgres

echo "=== Chap container logs ==="
docker compose logs --tail=100 chap

echo "=== Worker container logs ==="
docker compose logs --tail=50 worker

echo "=== Waiting for test container to complete ==="
exit_code=$(docker wait chap_r_model_test)

echo "=== Test container logs ==="
docker compose -f compose.yml -f compose.r-model.integration.test.yml logs chap_r_model_test

echo "=== Test exit code: $exit_code ==="

echo "=== Final container status ==="
docker compose ps

if [ "$exit_code" -ne "0" ]; then
    echo "=== Test failed! Dumping all container logs ==="
    docker compose logs
    docker compose down
    echo "Test container exited with code: $exit_code"
    exit 1
fi

echo "=== Tests passed! Cleaning up ==="
docker compose down
