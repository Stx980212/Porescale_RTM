#!/bin/bash
rm -rf build/
./scripts/build.sh
cd build
./reactive_transport | tee runninglog.txt
