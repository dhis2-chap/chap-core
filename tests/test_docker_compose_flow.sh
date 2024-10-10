# This file runs the chap_test docker image specified in compose.test.yml,
# which runs pytest inside the container and checks that the exit code is 0 (i.e. no failed tests)

set -e
docker build -t climate_health-chap:latest -f Dockerfile .
docker compose -f compose.yml -f compose.test.yml up --build --detach
docker attach chap_test

exit_code=$(docker inspect chap_test --format='{{.State.ExitCode}}')
docker compose down
[ "$exit_code" -eq "0" ] || { echo "Variable is not zero."; exit 1; }
