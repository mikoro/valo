// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <limits>

#include "Math/Vector3.h"
#include "Math/Vector2.h"
#include "Math/Color.h"
#include "Math/ONB.h"

namespace Raycer
{
	class Material;

	class Intersection
	{
	public:

		bool wasFound = false;
		bool isBehind = false;
		bool hasColor = false;

		float distance = std::numeric_limits<float>::max();

		Vector3 position;
		Vector3 normal;
		Vector2 texcoord;
		Vector3 rayDirection;
		Color color;

		ONB onb;

		Material* material = nullptr;
	};
}
