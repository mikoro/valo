// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Tracing/Ray.h"

using namespace Raycer;

void Ray::precalculate()
{
	inverseDirection = direction.inversed();

	directionIsNegative[0] = direction.x < 0.0;
	directionIsNegative[1] = direction.y < 0.0;
	directionIsNegative[2] = direction.z < 0.0;
}
