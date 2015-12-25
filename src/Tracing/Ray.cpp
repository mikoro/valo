// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Tracing/Ray.h"

using namespace Raycer;

void Ray::precalculate()
{
	inverseDirection = direction.inversed();
}
