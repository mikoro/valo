// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <string>
#include <vector>

#include "Raytracing/Primitives/Primitive.h"
#include "Raytracing/Primitives/Triangle.h"
#include "Math/Vector3.h"
#include "Math/EulerAngle.h"

namespace Raycer
{
	class Ray;
	struct Intersection;
	class AABB;

	class Mesh : public Primitive
	{
	public:

		void initialize();
		void intersect(const Ray& ray, Intersection& intersection) const;
		AABB getAABB() const;

		std::string meshFilePath;
		Vector3 position = Vector3(0.0, 0.0, 0.0);
		Vector3 scale = Vector3(1.0, 1.0, 1.0);
		EulerAngle orientation = EulerAngle(0.0, 0.0, 0.0);

		std::vector<Triangle> triangles;
	};
}