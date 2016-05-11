#!/bin/sh

g++ -c -isystem include -Isrc -std=c++14 -Wpedantic -Wall -Wextra -Werror -Ofast -fopenmp src/Precompiled.h -o src/Precompiled.h.gch
