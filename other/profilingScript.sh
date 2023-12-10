#!/bin/bash

# call example - call in folder containing this file
# bash profilingScript.sh nnTestMultiCR3BPchopped profiledCR3BPchopped

# Get the input script filename from the command line argument
script="$1"

# Get the output filename from the command line argument or use default
output="${2:-output.dat}"

# Run the profiling command
# python -m cProfile -o "$output" "$script"
python -m cProfile -o "./profilingData/$output.dat" "./hardware implementation/$script.py"
# run snakeviz
snakeviz "profilingData/$output.dat"