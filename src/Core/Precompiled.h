// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <random>
#include <regex>
#include <stdexcept>
#include <string>
#include <sstream>
#include <thread>
#include <vector>

#include <omp.h>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#include <ppl.h>
#else
#include <errno.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <signal.h>
#endif

#include <GL/gl3w.h>
#include <GL/glcorearb.h>
#include <GLFW/glfw3.h>

#ifdef __linux
#include <dlfcn.h>
#include <GL/glx.h>
#include <parallel/algorithm>
#endif

#include "tinyformat/tinyformat.h"
