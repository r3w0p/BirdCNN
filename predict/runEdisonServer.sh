#!/bin/sh

mkdir -p ./edison
cd ./edison
python3 -m http.server 8000
