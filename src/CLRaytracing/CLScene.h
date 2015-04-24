// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <vector>

#include "CLRaytracing/CLStructs.h"

namespace Raycer
{
	class Scene;

	class CLScene
	{
	public:

		void readScene(const Scene& scene);

		OpenCL::Info info;
		OpenCL::Camera camera;
		std::vector<OpenCL::Light> lights;
		std::vector<OpenCL::Plane> planes;
		std::vector<OpenCL::Sphere> spheres;
	};
}
