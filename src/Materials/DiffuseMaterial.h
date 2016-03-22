// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Core/Common.h"

namespace Raycer
{
	class Vector3;
	class Intersection;
	class Random;
	class Color;
	class Material;

	class DiffuseMaterial
	{
	public:

		CUDA_CALLABLE Vector3 getDirection(const Material& material, const Intersection& intersection, Random& random);
		CUDA_CALLABLE Color getBrdf(const Material& material, const Intersection& intersection, const Vector3& in, const Vector3& out);
		CUDA_CALLABLE float getPdf(const Material& material, const Intersection& intersection, const Vector3& out);
	};
}
