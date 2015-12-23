#!/bin/sh

if [ ! -d bin/data ]; then
  cp -R data bin/
fi

if [ ! -f bin/raycer.ini ]; then
  cp misc/raycer.ini bin/
fi
