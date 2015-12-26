pathtracer
 - russian roulette
 - direct light sampling
 - pathtracer materials

textures
 - mip map generation
 - ray differentials
 - EWA texture filtering

misc
 - add possibility to save and restore film state
 - film state and output image periodic saving
 - move properties behind getters/setters
 - move rayStartOffset to minDistance
 - move image pool to scene and add serialization
 - implement random class with different backends (pcg)
 - add material inheritance
 - add light inheritance
 - add spline curve class
 - add general transform class (translation with spline curves etc.)
 - add BVH -> QBVH conversion
 - implement QBVH travelsal and SIMD triangle intersect with ispc
 - replace private constructors with =deleted
 - add scope exit to obj loader and consolerunner text color
 - replace align macro with alignas
 - aabb optimization for triangles after transformation
