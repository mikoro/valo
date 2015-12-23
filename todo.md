pathtracer
 - russian roulette
 - direct light sampling
 - pathtracer materials

textures
 - mip map generation
 - ray differentials
 - EWA texture filtering

misc
 - check if ONB is right handed
 - check command line switches where used
 - move external program launch to sysutils
 - move apple curdir stuff to sysutils
 - check if interrupt works on mac/linux
 - add possibility to save and restore film state
 - add material inheritance
 - add light inheritance
 - move rayStartOffset to minDistance
 - templetize image (float and double images)
 - change textures to float, output image to double
 - add image data to serialization
 - implement info text panel + more statistics
 - implement random class with different backends (pcg)
 - add spline curve class
 - add general transform class (translation with spline curves etc.)
 - add BVH -> QBVH conversion
 - implement QBVH travelsal and SIMD triangle intersect with ispc
 - replace private constructors with =deleted
 - add scope exit to obj loader and consolerunner text color
 - replace align macro with alignas
 - aabb optimization for triangles after transformation
