raytracer
 - height map

pathtracer
 - russian roulette
 - direct light sampling
 - pathtracer materials

textures
 - mip map generation
 - ray differentials
 - EWA texture filtering

misc
 - cereal to public
 - remove csg suppport
 - remove primitives and use triangles only
 - remove some textures
 - remove ply loader
 - simplify material
 - add material inheritance
 - add light inheritance
 - use only normal mapping (remove rest)
 - change tclap to boost program options and move settings to root
 - move rayStartOffset to minDistance
 - templetize image (float and double images)
 - change textures to float, output image to double
 - fast preview with "dot product lighting"
 - add image data to serialization
 - implement info text panel + more statistics
 - move ray-scene intersection to scene
 - implement random class with different backends (pcg)
 - add spline curve class
 - add general transform class (translation with spline curves etc.)
 - add BVH -> QBVH conversion
 - implement QBVH travelsal and SIMD triangle intersect with ispc
 - replace private constructors with =deleted
 - add scope exit to obj loader and consolerunner text color
 - replace align macro with alignas
 - disable reinhard averaging when not in interactive mode
 - aabb optimization for triangles after transformation
