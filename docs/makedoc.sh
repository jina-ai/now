#!/usr/bin/env bash

set -ex

rm -rf api _build && make clean

make dirhtml
