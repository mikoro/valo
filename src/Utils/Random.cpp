// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

#include "Math/Color.h"
#include "Math/Vector2.h"
#include "Math/Vector3.h"
#include "Utils/Random.h"

using namespace Raycer;

RandomGeneratorPCG::RandomGeneratorPCG()
{
	state = 0x853c49e6748fea9bULL;
	inc = 0xda3e39cb94b95bdbULL;
}

RandomGeneratorPCG::RandomGeneratorPCG(uint64_t seed_)
{
	seed(seed_);
}

void RandomGeneratorPCG::seed(uint64_t seed_)
{
	state = seed_;
	inc = reinterpret_cast<uint64_t>(this);
}

#ifdef _MSC_VER
#pragma warning (push)
#pragma warning (disable : 4146)
#endif
RandomGeneratorPCG::result_type RandomGeneratorPCG::operator()()
{
	uint64_t oldstate = state;
	state = oldstate * 6364136223846793005ULL + inc;
	uint32_t xorshifted = static_cast<uint32_t>(((oldstate >> 18u) ^ oldstate) >> 27u);
	uint32_t rot = static_cast<uint32_t>(oldstate >> 59u);

	return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}
#ifdef _MSC_VER
#pragma warning (pop)
#endif

Random::Random()
{
}

Random::Random(uint64_t seed_)
{
	generator.seed(seed_);
}

void Random::seed(uint64_t seed_)
{
	generator.seed(seed_);
}

int32_t Random::getInt32(int32_t min, int32_t max)
{
	return std::uniform_int_distribution<int32_t>(min, max)(generator);
}

uint32_t Random::getUint32(uint32_t min, uint32_t max)
{
	return std::uniform_int_distribution<uint32_t>(min, max)(generator);
}

float Random::getFloat()
{
	//return std::uniform_real_distribution<float>(0.0f, 1.0f)(generator);
	return float(ldexp(generator(), -32));
}

double Random::getDouble()
{
	//return std::uniform_real_distribution<double>(0.0f, 1.0f)(generator);
	return ldexp(generator(), -32);
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
