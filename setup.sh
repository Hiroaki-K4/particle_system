#!/bin/bash

if [ ! -d "glfw" ]; then
    git clone https://github.com/glfw/glfw.git
    cd glfw
    cmake -S . -B build
    cd build
    make
fi
