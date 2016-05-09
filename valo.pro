TARGET = valo
TEMPLATE = app
DESTDIR = bin
OBJECTS_DIR = build
CONFIG += c++11
QMAKE_LIBDIR += platform/linux/lib
QMAKE_CXXFLAGS += -fopenmp -march=native
LIBS += -lstdc++ -ldl -lm -lpthread -lGL -lglfw -lboost_system -lboost_filesystem -lboost_program_options -fopenmp
QMAKE_POST_LINK += platform/linux/post-build.sh

INCLUDEPATH += include \
               src

HEADERS += src/BVH/BVH.h \
           src/BVH/BVH1.h \
           src/BVH/BVH4.h \
           src/BVH/Common.h \
           src/Core/AABB.h \
           src/Core/App.h \
           src/Core/Camera.h \
           src/Core/Common.h \
           src/Core/Film.h \
           src/Core/Image.h \
           src/Core/Intersection.h \
           src/Core/Precompiled.h \
           src/Core/Ray.h \
           src/Core/Scene.h \
           src/Core/Triangle.h \
           src/Filters/BellFilter.h \
           src/Filters/BoxFilter.h \
           src/Filters/Filter.h \
           src/Filters/GaussianFilter.h \
           src/Filters/LanczosSincFilter.h \
           src/Filters/MitchellFilter.h \
           src/Filters/TentFilter.h \
           src/Integrators/DotIntegrator.h \
           src/Integrators/Integrator.h \
           src/Integrators/PathIntegrator.h \
           src/Materials/BlinnPhongMaterial.h \
           src/Materials/DiffuseMaterial.h \
           src/Materials/Material.h \
           src/Math/AxisAngle.h \
           src/Math/Color.h \
           src/Math/EulerAngle.h \
           src/Math/Mapper.h \
           src/Math/MathUtils.h \
           src/Math/Matrix4x4.h \
           src/Math/MovingAverage.h \
           src/Math/ONB.h \
           src/Math/Polynomial.h \
           src/Math/Quaternion.h \
           src/Math/Solver.h \
           src/Math/Vector2.h \
           src/Math/Vector3.h \
           src/Math/Vector4.h \
           src/Renderers/CpuRenderer.h \
           src/Renderers/CudaRenderer.h \
           src/Renderers/Renderer.h \
           src/Runners/ConsoleRunner.h \
           src/Runners/WindowRunner.h \
           src/Runners/WindowRunnerRenderState.h \
           src/TestScenes/TestScene.h \
           src/Textures/CheckerTexture.h \
           src/Textures/ImageTexture.h \
           src/Textures/Texture.h \
           src/Tonemappers/LinearTonemapper.h \
           src/Tonemappers/PassthroughTonemapper.h \
           src/Tonemappers/ReinhardTonemapper.h \
           src/Tonemappers/SimpleTonemapper.h \
           src/Tonemappers/Tonemapper.h \
           src/Utils/FilmQuad.h \
           src/Utils/FpsCounter.h \
           src/Utils/GLHelper.h \
           src/Utils/ImagePool.h \
           src/Utils/InfoPanel.h \
           src/Utils/Log.h \
           src/Utils/ModelLoader.h \
           src/Utils/Random.h \
           src/Utils/Settings.h \
           src/Utils/StringUtils.h \
           src/Utils/SysUtils.h \
           src/Utils/Timer.h

SOURCES += src/Main.cpp \
           src/BVH/BVH.cpp \
           src/BVH/BVH1.cpp \
           src/BVH/BVH4.cpp \
           src/Core/AABB.cpp \
           src/Core/App.cpp \
           src/Core/Camera.cpp \
           src/Core/Film.cpp \
           src/Core/Image.cpp \
           src/Core/Ray.cpp \
           src/Core/Scene.cpp \
           src/Core/Triangle.cpp \
           src/External/gl3w.cpp \
           src/External/nanovg.cpp \
           src/External/stb.cpp \
           src/Filters/BellFilter.cpp \
           src/Filters/BoxFilter.cpp \
           src/Filters/Filter.cpp \
           src/Filters/GaussianFilter.cpp \
           src/Filters/LanczosSincFilter.cpp \
           src/Filters/MitchellFilter.cpp \
           src/Filters/TentFilter.cpp \
           src/Integrators/DotIntegrator.cpp \
           src/Integrators/Integrator.cpp \
           src/Integrators/PathIntegrator.cpp \
           src/Materials/BlinnPhongMaterial.cpp \
           src/Materials/DiffuseMaterial.cpp \
           src/Materials/Material.cpp \
           src/Math/AxisAngle.cpp \
           src/Math/Color.cpp \
           src/Math/EulerAngle.cpp \
           src/Math/Mapper.cpp \
           src/Math/MathUtils.cpp \
           src/Math/Matrix4x4.cpp \
           src/Math/MovingAverage.cpp \
           src/Math/ONB.cpp \
           src/Math/Quaternion.cpp \
           src/Math/Solver.cpp \
           src/Math/Vector2.cpp \
           src/Math/Vector3.cpp \
           src/Math/Vector4.cpp \
           src/Renderers/CpuRenderer.cpp \
           src/Renderers/CudaRenderer.cpp \
           src/Renderers/Renderer.cpp \
           src/Runners/ConsoleRunner.cpp \
           src/Runners/WindowRunner.cpp \
           src/Runners/WindowRunnerRenderState.cpp \
           src/Tests/EulerAngleTest.cpp \
           src/Tests/FilterTest.cpp \
           src/Tests/ImageTest.cpp \
           src/Tests/MathUtilsTest.cpp \
           src/Tests/Matrix4x4Test.cpp \
           src/Tests/ModelLoaderTest.cpp \
           src/Tests/OnbTest.cpp \
           src/Tests/PolynomialTest.cpp \
           src/Tests/SolverTest.cpp \
           src/Tests/TestScenesTest.cpp \
           src/Tests/Vector3Test.cpp \
           src/TestScenes/TestScene.cpp \
           src/TestScenes/TestScene1.cpp \
           src/TestScenes/TestScene2.cpp \
           src/TestScenes/TestScene3.cpp \
           src/TestScenes/TestScene4.cpp \
           src/TestScenes/TestScene5.cpp \
           src/TestScenes/TestScene6.cpp \
           src/TestScenes/TestScene7.cpp \
           src/TestScenes/TestScene8.cpp \
           src/TestScenes/TestScene9.cpp \
           src/Textures/CheckerTexture.cpp \
           src/Textures/ImageTexture.cpp \
           src/Textures/Texture.cpp \
           src/Tonemappers/LinearTonemapper.cpp \
           src/Tonemappers/PassthroughTonemapper.cpp \
           src/Tonemappers/ReinhardTonemapper.cpp \
           src/Tonemappers/SimpleTonemapper.cpp \
           src/Tonemappers/Tonemapper.cpp \
           src/Utils/FilmQuad.cpp \
           src/Utils/FpsCounter.cpp \
           src/Utils/GLHelper.cpp \
           src/Utils/ImagePool.cpp \
           src/Utils/InfoPanel.cpp \
           src/Utils/Log.cpp \
           src/Utils/ModelLoader.cpp \
           src/Utils/Random.cpp \
           src/Utils/Settings.cpp \
           src/Utils/StringUtils.cpp \
           src/Utils/SysUtils.cpp \
           src/Utils/Timer.cpp
