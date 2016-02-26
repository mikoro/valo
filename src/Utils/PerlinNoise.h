// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <vector>

/*

http://mrl.nyu.edu/~perlin/noise/
http://mrl.nyu.edu/~perlin/paper445.pdf

getNoise returns values between 0.0 - 1.0
getFbmNoise return values between 0.0 - inf

*/

namespace Raycer
{
	class PerlinNoise
	{
	public:

		PerlinNoise();
		PerlinNoise(uint64_t seed);

		void initialize(uint64_t seed);
		float getNoise(float x, float y, float z) const;
		float getFbmNoise(uint64_t octaves, float lacunarity, float persistence, float x, float y, float z) const;

	private:

		float fade(float t) const;
		float lerp(float t, float a, float b) const;
		float grad(uint64_t hash, float x, float y, float z) const;

		std::vector<uint64_t> permutations;
	};
}
