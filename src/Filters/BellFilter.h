// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

namespace Raycer
{
	class Vector2;

	class BellFilter
	{
	public:

		float getWeight(float s);
		float getWeight(const Vector2& point);

		Vector2 getRadius();
	};
}
