#!/bin/bash
# This file runs the chap_test docker image specified in compose.test.yml,
# which runs pytest inside the container and checks that the exit code is 0 (i.e. no failed tests)

set -e

echo "=== Cleaning up previous containers ==="
docker compose down

echo "=== Starting Docker Compose stack ==="
docker compose -f compose.yml -f compose.integration.test.yml up --build --detach --force-recreate

echo "=== Waiting for containers to initialize ==="
sleep 10

echo "=== Container status ==="
docker compose ps

echo "=== Chap container logs ==="
docker compose logs --tail=100 chap

echo "=== Worker container logs ==="
docker compose logs --tail=50 worker

echo "=== Postgres container logs ==="
docker compose logs --tail=50 postgres

echo "=== Attaching to test container ==="
docker attach chap_frontend_emulator

echo "=== Getting test exit code ==="
exit_code=$(docker inspect chap_frontend_emulator --format='{{.State.ExitCode}}')

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
