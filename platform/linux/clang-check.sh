#!/bin/sh

find src \( -name "*.cpp" -o -name "*.cu" \) | xargs -I file --verbose clang-check -analyze -fixit file -- -isystem include -Isrc -std=c++14 -Wpedantic -Wall -Wextra -Werror -Ofast -x c++
