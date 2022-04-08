#!/usr/bin/env bash

set -ex

BATCH_SIZE=3
declare -a mixins=( $(find tests -name "test_*.py") )
declare -a array1=( "$(echo "${mixins[@]}" | xargs -n$BATCH_SIZE)" )
# array2 has no output as it is empty. When the new folder is added
# it will automatically be listed and tested in CI
declare -a array2=( $(ls -d tests/integration/*/ | grep -v '__pycache__'))
dest=( "${array1[@]}" "${array2[@]}")

printf '%s\n' "${dest[@]}" | jq -R . | jq -cs .
