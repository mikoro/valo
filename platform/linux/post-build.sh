#!/bin/sh

if [ ! -d bin/data ]; then
  cp -R data bin/
fi

if [ ! -f bin/valo.ini ]; then
  cp misc/valo.ini bin/
fi
