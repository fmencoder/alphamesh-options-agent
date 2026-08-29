#!/bin/sh
# Fail fast and legibly when the journal directory is not writable.
#
# A Railway volume is mounted root-owned by default. This container runs as a
# non-root user, so without this check the first Journal write would fail deep
# inside the agent with an opaque sqlite error, several seconds into a cycle.
# Better to refuse to start and say exactly what to fix.
set -e

DB_PATH="${DATABASE_PATH:-/data/alphamesh.db}"
DB_DIR=$(dirname "$DB_PATH")

mkdir -p "$DB_DIR" 2>/dev/null || true

if [ ! -w "$DB_DIR" ]; then
    echo "FATAL: journal directory '$DB_DIR' is not writable by uid $(id -u)." >&2
    echo "" >&2
    echo "The audit journal must survive restarts. Fix one of:" >&2
    echo "  - mount the Railway volume at '$DB_DIR' and make it writable by" >&2
    echo "    uid 10001 (or run the service as root)" >&2
    echo "  - set DATABASE_PATH to a writable path" >&2
    exit 78
fi

exec "$@"
