# This file runs the chap_test docker image specified in compose.test.yml,
# which runs pytest inside the container and checks that the exit code is 0 (i.e. no failed tests)

set -e
docker compose -f compose.yml -f compose.integration.test.yml up --build --detach --force-recreate
docker attach chap_frontend_emulator

exit_code=$(docker inspect chap_frontend_emulator --format='{{.State.ExitCode}}')
docker compose down
[ "$exit_code" -eq "0" ] || { echo "Variable is not zero."; exit 1; }
