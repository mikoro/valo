# Valo

C++11/OpenMP/CUDA physically based renderer with an interactive preview mode.

[Image gallery](https://www.flickr.com/photos/136293057@N06/albums/72157665827123423)

* Author: [Mikko Ronkainen](http://mikkoronkainen.com)
* Website: [github.com/mikoro/valo](https://github.com/mikoro/valo)

![Screenshot](http://mikoro.github.io/images/valo/readme-screenshot.jpg "Screenshot")

## Download

Download the latest version:

| Windows 64-bit                                                                                               | Mac OS X (10.9+)                                                                                       | Linux 64-bit (Ubuntu 15.10)                                                                                      |
|--------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| [valo-0.1.0-win.zip](https://github.com/mikoro/valo/releases/download/v0.1.0/valo-0.1.0-win.zip)             | N/A | N/A |
| [valo-0.1.0-win-cuda.zip](https://github.com/mikoro/valo/releases/download/v0.1.0/valo-0.1.0-win-cuda.zip)   | | |
| [valo-0.1.0-win-intel.zip](https://github.com/mikoro/valo/releases/download/v0.1.0/valo-0.1.0-win-intel.zip) | | |

You will also need the [Visual C++ Redistributable for Visual Studio 2015](https://www.microsoft.com/en-us/download/details.aspx?id=48145).

The CUDA version runs on NVIDIA GTX 900 Series or newer and the Intel-specific version runs on Ivy Bridge processors or newer.

## Features

- Physically based renderer
  - Uses path tracing to solve the rendering equation
  - Materials with BRDFs and PDFs (only Lambertian at the moment)
  - (Multiple) importance sampling
  - Russian roulette
  - Direct light sampling
- Volumetric effects with ray marching
- Film for accumulating samples
- Antialiasing with different filters
- Tonemapping with different operators
- Image based and procedural textures
- BVHs with different widths (2/4/8), optimal and fast SAH builder and vectorized traversal
- Random numbers based on PCG
- Very fast OBJ file loader with material support
- GPU rendering with CUDA (based on megakernel approach, not so fast)
- Offline console rendering and interactive windowed mode

## Instructions

Running the program will open the first test scene in an interactive windowed mode. Use WASD to move and left mouse button to look around. Cycle through the test scenes with number keys.

Edit the valo.ini file to further configure the program. All options can be overridden with command line switches.

### Controls

For the interactive mode:

| Key                     | Action                                                                                |
|-------------------------|---------------------------------------------------------------------------------------|
| **W/A/S/D**             | Move around (+ Q/E for up/down)                                                       |
| **Mouse left**          | Look around                                                                           |
| **Ctrl**                | Move slower                                                                           |
| **Shift**               | Move faster                                                                           |
| **Alt**                 | Move even faster                                                                      |
| **Insert/Delete**       | Adjust move speed                                                                     |
| **Space**               | Stop moving                                                                           |
| **P**                   | Toggle moving on/off                                                                  |
| **R**                   | Reset camera                                                                          |
| **F**                   | Toggle filtering on/off                                                               |
| **M**                   | Toggle normal mapping on/off                                                          |
| **N**                   | Toggle normal interpolation on/off                                                    |
| **B**                   | Toggle normal visualization on/off                                                    |
| **V**                   | Toggle triangle interpolation visualization on/off                                    |
| **F1**                  | Cycle info panel states                                                               |
| **F2**                  | Select renderer                                                                       |
| **F3**                  | Select camera projection                                                              |
| **F4**                  | Select integrator                                                                     |
| **F5**                  | Select filter                                                                         |
| **F6**                  | Select tonemapper                                                                     |
| **F7/F8**               | Decrease/increase internal rendering resolution                                       |
| **Ctrl+F1**             | Save scene to file (not implemented)                                                  |
| **Ctrl+F2**             | Save camera state to file                                                             |
| **Ctrl+F3**             | Save image to file                                                                    |
| **Ctrl+F4**             | Save film to file                                                                     |
| **F12**                 | Take a screenshot                                                                     |
| **Page Up/Down**        | Increase/decrease field of view                                                       |
| **Ctrl+Page Up/Down**   | Increase/decrease exposure                                                            |
| **1-0**                 | Select test scenes 1-10                                                               |
| **Ctrl+1-0**            | Select test scenes 11-20                                                              |

## Test scene data sources

Websites where the test scene data was sourced from:

[Blend Swap](http://www.blendswap.com/) (user Jay-Artist)
[McGuire Graphics Data](http://graphics.cs.williams.edu/data/meshes.xml)

## Build

See the [build instructions](https://github.com/mikoro/valo/blob/master/build.md).

## License

    Valo
    Copyright © 2016 Mikko Ronkainen
    
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.
