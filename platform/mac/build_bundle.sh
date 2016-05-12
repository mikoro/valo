#!/bin/sh

rm -rf Valo.app

mkdir -p Valo.app/Contents/MacOS
mkdir -p Valo.app/Contents/Resources
mkdir -p Valo.app/Contents/Libs

cp platform/mac/Info.plist Valo.app/Contents
cp -R data Valo.app/Contents/Resources
cp misc/icons/valo.icns Valo.app/Contents/Resources
cp misc/valo.ini Valo.app/Contents/Resources
cp bin/valo Valo.app/Contents/MacOS

platform/mac/dylibbundler -od -b -x ./Valo.app/Contents/MacOS/valo -d ./Valo.app/Contents/Libs/ -p @executable_path/../Libs/
chmod a+x Valo.app/Contents/Libs/*
