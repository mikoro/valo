// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Utils/PerlinNoise.h"
#include "Math/Vector3.h"

using namespace Raycer;

PerlinNoise::PerlinNoise() : permutationsAlloc(false)
{
}

void PerlinNoise::initialize(uint32_t seed)
{
	std::vector<uint32_t> tempPermutations(256);
	std::iota(tempPermutations.begin(), tempPermutations.end(), 0);
	std::mt19937 mt(seed);
	std::shuffle(tempPermutations.begin(), tempPermutations.end(), mt);
	std::vector<uint32_t> tempPermutationsDuplicate = tempPermutations;
	tempPermutations.insert(tempPermutations.end(), tempPermutationsDuplicate.begin(), tempPermutationsDuplicate.end());

	permutationsAlloc.resize(tempPermutations.size());
	permutationsAlloc.write(tempPermutations.data(), tempPermutations.size());
}

CUDA_CALLABLE float PerlinNoise::getNoise(const Vector3& position) const
{
	return getNoise(position.x, position.y, position.z);
}

CUDA_CALLABLE float PerlinNoise::getNoise(float x, float y, float z) const
{
	uint32_t X = uint32_t(floor(x)) & 255;
	uint32_t Y = uint32_t(floor(y)) & 255;
	uint32_t Z = uint32_t(floor(z)) & 255;

	x -= floor(x);
	y -= floor(y);
	z -= floor(z);

	float u = fade(x);
	float v = fade(y);
	float w = fade(z);

	uint32_t* permutations = permutationsAlloc.getPtr();

	uint32_t A = permutations[X] + Y;
	uint32_t AA = permutations[A] + Z;
	uint32_t AB = permutations[A + 1] + Z;
	uint32_t B = permutations[X + 1] + Y;
	uint32_t BA = permutations[B] + Z;
	uint32_t BB = permutations[B + 1] + Z;

	float n = lerp(w, lerp(v, lerp(u, grad(permutations[AA], x, y, z),
		grad(permutations[BA], x - 1, y, z)),
		lerp(u, grad(permutations[AB], x, y - 1, z),
		grad(permutations[BB], x - 1, y - 1, z))),
		lerp(v, lerp(u, grad(permutations[AA + 1], x, y, z - 1),
		grad(permutations[BA + 1], x - 1, y, z - 1)),
		lerp(u, grad(permutations[AB + 1], x, y - 1, z - 1),
		grad(permutations[BB + 1], x - 1, y - 1, z - 1))));

	return MAX(0.0f, MIN(0.5f + n / 2.0f, 1.0f)); // move and clamp to 0.0-1.0 range
}

CUDA_CALLABLE float PerlinNoise::getFbmNoise(uint32_t octaves, float lacunarity, float persistence, const Vector3& position) const
{
	return getFbmNoise(octaves, lacunarity, persistence, position.x, position.y, position.z);
}

CUDA_CALLABLE float PerlinNoise::getFbmNoise(uint32_t octaves, float lacunarity, float persistence, float x, float y, float z) const
{
	float result = 0.0f;
	float frequency = 1.0f;
	float amplitude = 1.0f;

	for (uint32_t i = 0; i < octaves; ++i)
	{
		result += getNoise(x * frequency, y * frequency, z * frequency) * amplitude;
		frequency *= lacunarity;
		amplitude *= persistence;
	}

	return result;
}

CUDA_CALLABLE float PerlinNoise::fade(float t) const
{
	return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

CUDA_CALLABLE float PerlinNoise::lerp(float t, float a, float b) const
{
	return a + t * (b - a);
}

CUDA_CALLABLE float PerlinNoise::grad(uint32_t hash, float x, float y, float z) const
{
	uint32_t h = hash & 15;
	float u = (h < 8) ? x : y;
	float v = (h < 4) ? y : ((h == 12 || h == 14) ? x : z);

	return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}
