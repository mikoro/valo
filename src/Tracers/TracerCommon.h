// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

namespace Raycer
{
	class Intersection;
	class Color;
	class Vector3;

	class TracerCommon
	{
	public:

		static void calculateNormalMapping(Intersection& intersection);
		static Color calculateNormalColor(const Vector3& normal);
	};
}
