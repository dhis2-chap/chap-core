#!/bin/bash
set -e

echo "Testing docker compose with chapkit models..."

# Clean up any existing containers
echo "Cleaning up existing containers..."
docker compose down --volumes || true

# Start services in detached mode
echo "Starting services..."
docker compose up -d --build --force-recreate

# Function to check if any container has exited
check_for_exits() {
    if docker compose -f compose.yml -f compose-models.yml ps --format json | grep -q '"State":"exited"'; then
        echo ""
        echo "ERROR: One or more services exited unexpectedly"
        docker compose -f compose.yml -f compose-models.yml ps
        echo ""
        echo "Last 50 lines of logs:"
        docker compose -f compose.yml -f compose-models.yml logs --tail=50
        docker compose -f compose.yml -f compose-models.yml down --volumes
        exit 1
    fi
}

# Function to wait for a service to be healthy
wait_for_service() {
    local url=$1
    local name=$2
    local max_wait=30
    local elapsed=0

    echo "Waiting for $name to be healthy (timeout: ${max_wait}s)..."
    while [ $elapsed -lt $max_wait ]; do
        # Check if any containers have exited before continuing
        check_for_exits

        if curl -sf "$url" >/dev/null 2>&1; then
            echo "✓ $name is healthy"
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
        echo -n "."
    done
    echo ""
    echo "ERROR: $name did not become healthy within ${max_wait}s"
    check_for_exits  # Final check in case it just exited
    return 1
}

# Wait for services to be healthy
if ! wait_for_service "http://localhost:5001/api/v1/health" "chtorch"; then
    docker compose -f compose.yml -f compose-models.yml logs chtorch --tail=50
    docker compose -f compose.yml -f compose-models.yml down --volumes
    exit 1
fi

if ! wait_for_service "http://localhost:8000/health" "chap"; then
    docker compose -f compose.yml -f compose-models.yml logs chap --tail=50
    docker compose -f compose.yml -f compose-models.yml down --volumes
    exit 1
fi

# Final check for exits
check_for_exits

echo "✓ All services started successfully and are healthy"

# Clean up
echo "Cleaning up..."
docker compose -f compose.yml -f compose-models.yml down --volumes

echo "✓ Test completed successfully"
