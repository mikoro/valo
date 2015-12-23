#include <algorithm>
#include <array>
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
#include <limits>
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

#include <boost/align/aligned_allocator.hpp>
#include <boost/asio.hpp>
#include <boost/array.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#else
#include <errno.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include <GL/gl3w.h>
#include <GL/glcorearb.h>
#include <GLFW/glfw3.h>

#ifdef __linux
#include <dlfcn.h>
#include <GL/glx.h>
#endif

#include "tinyformat/tinyformat.h"

#ifdef __APPLE__
// cereal and tclap needs to be here, otherwise won't compile on Apple clang
#include "cereal/cereal.hpp"
#include "cereal/archives/json.hpp"
#include "cereal/archives/xml.hpp"
#include "cereal/archives/binary.hpp"
#include "cereal/types/vector.hpp"
#include "cereal/types/string.hpp"

#include <Carbon/Carbon.h>
#include <CoreFoundation/CoreFoundation.h>
#endif

#include "Common.h"
