// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <vector>

#include "Math/Vector2.h"
#include "Utils/Random.h"

/*

Samplers work in procedual and/or pre-generated fashion.
getSample generates a new sample on the fly
generateSamples fills internal buffer with getSample or some other way
getNextSample loops through the internal buffer and returns false when one loop through the samples is completed

*/

namespace Raycer
{
	class Vector3;
	class ONB;

	enum class SamplerType { CENTER, RANDOM, REGULAR, JITTERED, CMJ, POISSON_DISC };

	class Sampler
	{
	public:

		virtual ~Sampler() {}

		virtual float getSample(uint64_t x, uint64_t n, uint64_t permutation, Random& random) = 0;
		virtual Vector2 getSquareSample(uint64_t x, uint64_t y, uint64_t nx, uint64_t ny, uint64_t permutation, Random& random) = 0;
		Vector2 getDiscSample(uint64_t x, uint64_t y, uint64_t nx, uint64_t ny, uint64_t permutation, Random& random);
		Vector3 getCosineHemisphereSample(const ONB& onb, uint64_t x, uint64_t y, uint64_t nx, uint64_t ny, uint64_t permutation, Random& random);
		Vector3 getUniformHemisphereSample(const ONB& onb, uint64_t x, uint64_t y, uint64_t nx, uint64_t ny, uint64_t permutation, Random& random);

		virtual void generateSamples1D(uint64_t sampleCount, Random& random);
		virtual void generateSamples2D(uint64_t sampleCountSqrt, Random& random);

		bool getNextSample(float& result);
		bool getNextSquareSample(Vector2& result);
		bool getNextDiscSample(Vector2& result);
		bool getNextCosineHemisphereSample(const ONB& onb, Vector3& result);
		bool getNextUniformHemisphereSample(const ONB& onb, Vector3& result);
		void reset();

		static std::unique_ptr<Sampler> getSampler(SamplerType type);

	protected:

		std::vector<float> samples1D;
		std::vector<Vector2> samples2D;

	private:

		uint64_t nextSampleIndex1D = 0;
		uint64_t nextSampleIndex2D = 0;
	};
}
