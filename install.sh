#!/bin/bash

# Check if pip is installed
command -v pip >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Error: pip is not installed. Please install pip first."
    exit 1
fi

# Run the pip install command in the current directory
pip install .