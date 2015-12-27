pathtracer
 - russian roulette
 - direct light sampling
 - pathtracer materials

textures
 - mip map generation
 - ray differentials
 - EWA texture filtering

misc
 - fix coordinate system
 - move properties behind getters/setters
 - remove unnecessary initializers
 - add possibility to save and restore film state
 - film state and output image periodic saving

 - move image pool to scene and add serialization
 - move rayStartOffset to minDistance
 - add material inheritance
 - add light inheritance
 - add spline curve class
 - add general transform class (translation with spline curves etc.)
 - primitive and primitive group with transform
 - add BVH -> QBVH conversion
 - implement QBVH travelsal and SIMD triangle intersect with ispc
 - replace private constructors with =deleted
 - add scope exit to obj loader and consolerunner text color
 - replace align macro with alignas

