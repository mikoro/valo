// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Math/Color.h"
#include "Math/Vector2.h"
#include "Math/Vector3.h"
#include "Math/Random.h"

#ifdef _MSC_VER
#pragma warning (push)
#pragma warning (disable : 4146)
#endif

using namespace Raycer;

CUDA_CALLABLE RandomGeneratorPCG::RandomGeneratorPCG()
{
	state = 0x853c49e6748fea9bULL;
	inc = 0xda3e39cb94b95bdbULL;
}

CUDA_CALLABLE RandomGeneratorPCG::RandomGeneratorPCG(uint64_t seed_)
{
	seed(seed_);
}

CUDA_CALLABLE void RandomGeneratorPCG::seed(uint64_t seed_)
{
	state = seed_;
	inc = reinterpret_cast<uint64_t>(this);
}

// https://github.com/imneme/pcg-c-basic/blob/master/pcg_basic.c
CUDA_CALLABLE RandomGeneratorPCG::result_type RandomGeneratorPCG::operator()()
{
	uint64_t oldstate = state;
	state = oldstate * 6364136223846793005ULL + inc;
	uint32_t xorshifted = static_cast<uint32_t>(((oldstate >> 18u) ^ oldstate) >> 27u);
	uint32_t rot = static_cast<uint32_t>(oldstate >> 59u);

	return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

CUDA_CALLABLE Random::Random()
{
}

CUDA_CALLABLE Random::Random(uint64_t seed_)
{
	generator.seed(seed_);
}

CUDA_CALLABLE void Random::seed(uint64_t seed_)
{
	generator.seed(seed_);
}

CUDA_CALLABLE uint32_t Random::getUint32()
{
	return generator();
}

CUDA_CALLABLE uint32_t Random::getUint32(uint32_t max)
{
	uint32_t threshold = -max % max;

	for (;;)
	{
		uint32_t value = generator();

		if (value >= threshold)
			return value % max;
	}
}

CUDA_CALLABLE uint32_t Random::getUint32(uint32_t min, uint32_t max)
{
	return getUint32((max - min) + 1) + min;
}

// http://mumble.net/~campbell/tmp/random_real.c
CUDA_CALLABLE float Random::getFloat()
{
#ifdef __CUDA_ARCH__
	return float(generator()) / float(0xFFFFFFFF);
#else
	return float(ldexp(generator(), -32));
#endif
}

CUDA_CALLABLE Color Random::getColor(bool randomAlpha)
{
	Color c;

	c.r = getFloat();
	c.g = getFloat();
	c.b = getFloat();
	c.a = randomAlpha ? getFloat() : 1.0f;

	return c;
}

CUDA_CALLABLE Vector2 Random::getVector2()
{
	Vector2 v;

	v.x = getFloat();
	v.y = getFloat();

	return v;
}

CUDA_CALLABLE Vector3 Random::getVector3()
{
	Vector3 v;

	v.x = getFloat();
	v.y = getFloat();
	v.z = getFloat();

	return v;
}

#ifdef _MSC_VER
#pragma warning (pop)
#endif
