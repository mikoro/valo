## Windows

- Install boost headers and binaries (http://sourceforge.net/projects/boost/files/boost-binaries/1.60.0/)
- Open solution with VS2015
- Adjust project include paths to point to the boost libraries
- Compile the code
- Run bin/{Debug,Release}/raycer.exe

### Boost

If using ICC, the boost framework needs to be compiled separately. Probably a good idea to do it with MSVC too.

Download the sources and run the commands from a corresponding developer console (MSVC or ICC):

    msvc:
    bootstrap.bat
    b2 --build-type=minimal toolset=msvc-14.0 address-model=64 stage
    
	intel:
	compilervars.bat intel64 vs2015
	bootstrap.bat
    b2 --build-type=minimal toolset=intel-16.0-vc14 address-model=64 stage

## Linux

- Install boost
- Install GLFW
- Compile:
    ```
    export CXX=<compiler>
    make -j4
    ```
- Copy test scene data to the bin/data directory (download link below)
- Run:
    ```
    cd bin && ./raycer
    ```

The GLFW library maybe named as *glfw* or *glfw3*. If there is a linking error, try adding or removing the last number.

## Mac

Can be compiled with the Apple clang supplied with Xcode 7.0 (OS X 10.11).

- Install Xcode + Command Line Tools
- Install MacPorts
- Install boost (macports)
- Install glfw (macports)
- Install libomp (macports)
- Compile:
    ```
    make -j4
    ```
- Copy test scene data to the bin/data directory (download link below)
- Run:
    ```
    cd bin && ./raycer
    ```
- Build app bundle:
    ```
    platform/mac/build_bundle.sh
    ```

See remarks of the linux build.

## Framework versions

- boost 1.60.0
- GLFW 3.1.2
- CATCH v1.2.1
- cereal 1.1.2
- stb (github 947bdcd027)
- tinyformat (github 3913307c28)
