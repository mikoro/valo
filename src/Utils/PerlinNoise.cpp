// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Utils/PerlinNoise.h"
#include "Utils/Random.h"

using namespace Raycer;

PerlinNoise::PerlinNoise()
{
	std::random_device rd;
	initialize(rd());
}

PerlinNoise::PerlinNoise(uint64_t seed)
{
	initialize(seed);
}

void PerlinNoise::initialize(uint64_t seed)
{
	permutations.clear();
	permutations.resize(256);
	std::iota(permutations.begin(), permutations.end(), 0);
	Random random(seed);
	std::shuffle(permutations.begin(), permutations.end(), random);
	std::vector<uint64_t> duplicate = permutations;
	permutations.insert(permutations.end(), duplicate.begin(), duplicate.end());
}

float PerlinNoise::getNoise(float x, float y, float z) const
{
	uint64_t X = uint64_t(floor(x)) & 255;
	uint64_t Y = uint64_t(floor(y)) & 255;
	uint64_t Z = uint64_t(floor(z)) & 255;

	x -= floor(x);
	y -= floor(y);
	z -= floor(z);

	float u = fade(x);
	float v = fade(y);
	float w = fade(z);

	uint64_t A = permutations[X] + Y;
	uint64_t AA = permutations[A] + Z;
	uint64_t AB = permutations[A + 1] + Z;
	uint64_t B = permutations[X + 1] + Y;
	uint64_t BA = permutations[B] + Z;
	uint64_t BB = permutations[B + 1] + Z;

	float n = lerp(w, lerp(v, lerp(u, grad(permutations[AA], x, y, z),
		grad(permutations[BA], x - 1, y, z)),
		lerp(u, grad(permutations[AB], x, y - 1, z),
		grad(permutations[BB], x - 1, y - 1, z))),
		lerp(v, lerp(u, grad(permutations[AA + 1], x, y, z - 1),
		grad(permutations[BA + 1], x - 1, y, z - 1)),
		lerp(u, grad(permutations[AB + 1], x, y - 1, z - 1),
		grad(permutations[BB + 1], x - 1, y - 1, z - 1))));

	return std::max(0.0f, std::min(0.5f + n / 2.0f, 1.0f)); // move and clamp to 0.0-1.0 range
}

float PerlinNoise::getFbmNoise(uint64_t octaves, float lacunarity, float persistence, float x, float y, float z) const
{
	float result = 0.0f;
	float frequency = 1.0f;
	float amplitude = 1.0f;

	for (uint64_t i = 0; i < octaves; ++i)
	{
		result += getNoise(x * frequency, y * frequency, z * frequency) * amplitude;
		frequency *= lacunarity;
		amplitude *= persistence;
	}

	return result;
}

float PerlinNoise::fade(float t) const
{
	return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

float PerlinNoise::lerp(float t, float a, float b) const
{
	return a + t * (b - a);
}

float PerlinNoise::grad(uint64_t hash, float x, float y, float z) const
{
	uint64_t h = hash & 15;
	float u = (h < 8) ? x : y;
	float v = (h < 4) ? y : ((h == 12 || h == 14) ? x : z);

	return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}
