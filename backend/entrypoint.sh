#!/bin/sh
# Entry point: `pytest ...` runs the suite on an isolated, fresh in-process
# SQLite database (see tests/conftest.py); anything else starts the API.
if [ "$1" = "pytest" ]; then
    tmp_db="$(mktemp -t hw_tests_XXXXXX.db)"
    export DATABASE_URL="sqlite+pysqlite:///${tmp_db}"
    shift
    exec pytest "$@"
fi
exec "$@"
