# ME-E4100 Advanced Computer Graphics, Spring 2016
# Lehtinen / Kemppinen, Ollikainen
#
# Assignment 1: Accelerated Ray Tracing

Student name: Mikko Ronkainen
Student number: xxx

# Which parts of the assignment did you complete? Mark them 'done'.
# You can also mark non-completed parts as 'attempted' if you spent a fair amount of
# effort on them. If you do, explain the work you did in the problems/bugs section
# and leave your 'attempt' code in place (commented out if necessary) so we can see it.

NOTE: You may have to add "C:\Program Files (x86)\Windows Kits\10\Include\XXXXXX\ucrt" to VS includes
and "C:\Program Files (x86)\Windows Kits\10\Lib\XXXXXX\ucrt\x64" to VS lib dirs before compiling.

NOTE: Compile the program with ReleaseFull configuration!

R1 BVH construction and traversal (5p): done [example_1.bat]
        R2 BVH saving and loading (1p): done [example_2.bat] (read instructions below)
              R3 Simple texturing (1p): done [example_3.bat]
             R4 Ambient occlusion (2p): done [example_4_1.bat] [example_4_2.bat]
         R5 Simple multithreading (1p): done (Tracer.cpp:83)

BVH saving and loading:
- run [example_1.bat]
- press ctrl + 3 (this saves everything from the scene to a single binary file)
- run [example_2.bat]

# Did you do any extra credit work?

BVH Surface Area Heuristic
 - The older version of the renderer had random, middle, regular and median splits
 - The median split was the fastest
 - In this newer version I threw out everything else and implemented full per object split (split by every triangle)
 - This per object split was about 20% faster to render than earlier best median split (and faster to build with better algorithm)

Efficient SAH building
 - The builder should be faster than the reference
 - Goes through all possible split locations
 - Uses parallelization when sorting triangles
 
Optimize your tracer
 - There is an early exit when travelling BVH and just looking for occlusion
 - Both BVH tree building and traversal are implemented with local stack
 - Minimum triangle count can be adjusted when building the BVH
 - Things should be quite sanely arranged in the memory and no pointer chasing
 
Comparison
 - There are comparison_x.bat files that should replicate the scenes that are used in the assigment code
 - The timing info for every step is printed to the console with millisecond accuracy
 - Megarays/s is also printed along with other metrics

Implement proper antialiasing for primary rays [example_5.bat]
 - Will shoot 64 rays around a pixel and combine them with a Mitchell filter
 - There is a film object that store cumulative colors and filter weights
 - Treats every pixel in isolation, so one color sample from a single ray affects only one pixel (no shared memory writes between threads)

Better sampling by low discrepancy sequences [example_5.bat]
 - The sample positions around a pixel are generated with correlated multi-jittered sampling
 - http://graphics.pixar.com/library/MultiJitteredSampling/paper.pdf
 - From the paper: "Though based on jittering, this method is competitive with low discrepancy quasi-Monte Carlo sequences while avoiding some of the structured artifacts to which they are prone."

Alpha and specular textures [example_6.bat]
 - The flower pot has alpha textures
 - Floor tiles and brick walls have specular textures

Scene file system for multiple objects
 - Not exactly what was asked in the task but here's a description what my system does:
   - The whole scene (settings, lights, triangles, bvh etc.) is described in a single large struct with substructs and vectors
   - Everything is annotated to be used with the C++ Cereal serialization framework
   - Cereal can save and load the scene using xml/json/binary
   - Scene can contain multiple models from different .obj files
   - Each .obj file can be transformed with its own transformation matrix when loaded
   - The obj file loader is my own and has been extended to support new material info (in .mtl files)
   
Tangent space normal mapping [example_6.bat]
 - Normals, tangents and bitangents are calculated when loading the triangles
 - Press N when looking at the pillars in the crytek sponza to toggle normal mapping on/off

Whitted integrator [example_7.bat]
 - Ray reflections and transmissions

Texture filtering
 - Textures are bilinear filtered
 - Image class support for bicubic filtering

BVH visualization [example_1.bat] [example_6.bat]
 - Press Ctrl + left to select the left child of the tree
 - Press Ctrl + right to select the right child
 - Press Ctrl + up to go back up a level in the tree

Area lights with soft shadows
 - The point light is modeled as a disc that is then sampled with a CMJ sampler
 
Parallelize your ray tracer using SIMD [example_9.bat]
 - QBVH (BVH4) has been implemented. It is constructed as a four wide tree from the start so it is not collapsed from a binary tree
 - The ray-box intersection with 4 AABBs has been vectorized (it is coded serially, but the compiler vectorizes it)
 - AABBs are in a SoA layout in the memory
 - It is not faster than the serial version
  - The triangle intersection needs to be vectorized still (maybe have fixed 8 triangles in a SoA layout in every leaf)
  - The BVH traversal speed may be memory bound?

# Are there any known problems/bugs remaining in your code?

# Did you collaborate with anyone in the class?

# Any other comments you'd like to share about the assignment or the course so far?
