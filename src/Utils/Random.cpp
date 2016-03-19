// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

#include "Math/Color.h"
#include "Math/Vector2.h"
#include "Math/Vector3.h"
#include "Utils/Random.h"

using namespace Raycer;

Random::Random()
{
	initialize();
}

Random::Random(uint32_t seed)
{
	initialize(seed);
}

void Random::initialize()
{
	std::random_device rd;
	initialize(rd());
}

void Random::initialize(uint32_t seed)
{
	generator.seed(uint32_t(seed));
}

int32_t Random::getInt32(int32_t min, int32_t max)
{
	return std::uniform_int_distribution<int32_t>(min, max)(generator);
}

uint32_t Random::getUint32(uint32_t min, uint32_t max)
{
	return std::uniform_int_distribution<uint32_t>(min, max)(generator);
}

float Random::getFloat(float min, float max)
{
	return std::uniform_real_distribution<float>(min, max)(generator);
}

double Random::getDouble(double min, double max)
{
	return std::uniform_real_distribution<double>(min, max)(generator);
}

Color Random::getColor(bool randomAlpha)
{
	Color c;

	c.r = getFloat();
	c.g = getFloat();
	c.b = getFloat();
	c.a = randomAlpha ? getFloat() : 1.0f;

	return c;
}

Vector2 Random::getVector2()
{
	Vector2 v;

	v.x = getFloat();
	v.y = getFloat();

	return v;
}

Vector3 Random::getVector3()
{
	Vector3 v;

	v.x = getFloat();
	v.y = getFloat();
	v.z = getFloat();

	return v;
}

Random::result_type Random::operator()()
{
	return std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint32_t>::max())(generator);
}
