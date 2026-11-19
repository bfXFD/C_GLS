#!/usr/bin/env bash
set -euo pipefail

# Compatibility alias. New automation should use the direction-explicit name.
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_fsm_independent_revisit_desk2_to_desk.sh" "$@"
