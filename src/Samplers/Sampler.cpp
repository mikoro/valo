// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Samplers/Sampler.h"
#include "Samplers/CenterSampler.h"
#include "Samplers/RandomSampler.h"
#include "Samplers/RegularSampler.h"
#include "Samplers/JitteredSampler.h"
#include "Samplers/CMJSampler.h"
#include "Samplers/PoissonDiscSampler.h"
#include "Math/Vector2.h"
#include "Math/Vector3.h"
#include "Tracing/ONB.h"

using namespace Raycer;

std::unique_ptr<Sampler> Sampler::getSampler(SamplerType type)
{
	switch (type)
	{
		case SamplerType::CENTER: return std::make_unique<CenterSampler>();
		case SamplerType::RANDOM: return std::make_unique<RandomSampler>();
		case SamplerType::REGULAR: return std::make_unique<RegularSampler>();
		case SamplerType::JITTERED: return std::make_unique<JitteredSampler>();
		case SamplerType::CMJ: return std::make_unique<CMJSampler>();
		case SamplerType::POISSON_DISC: return std::make_unique<PoissonDiscSampler>();
		default: throw std::runtime_error("Unknown sampler type");
	}
}

namespace
{
	// concentric square -> disc mapping
	Vector2 mapSquareToDisc(const Vector2& point)
	{
		float phi, r;
		float a = 2.0f * point.x - 1.0f;
		float b = 2.0f * point.y - 1.0f;

		if (a > -b)
		{
			if (a > b)
			{
				r = a;
				phi = (float(M_PI) / 4.0f) * (b / a);
			}
			else
			{
				r = b;
				phi = (float(M_PI) / 4.0f) * (2.0f - (a / b));
			}
		}
		else
		{
			if (a < b)
			{
				r = -a;
				phi = (float(M_PI) / 4.0f) * (4.0f + (b / a));
			}
			else
			{
				r = -b;

				if (b != 0.0f)
					phi = (float(M_PI) / 4.0f) * (6.0f - (a / b));
				else
					phi = 0.0f;
			}
		}

		float u = r * std::cos(phi);
		float v = r * std::sin(phi);

		return Vector2(u, v);
	}

	Vector3 mapDiscToCosineHemisphere(const ONB& onb, const Vector2& point)
	{
		float r2 = point.x * point.x + point.y * point.y;

		if (r2 > 1.0f)
			r2 = 1.0f;

		float x = point.x;
		float y = point.y;
		float z = std::sqrt(1.0f - r2);

		return x * onb.u + y * onb.v + z * onb.w;
	}

	Vector3 mapDiscToUniformHemisphere(const ONB& onb, const Vector2& point)
	{
		float r2 = point.x * point.x + point.y * point.y;
		float a = std::sqrt(2.0f - r2);

		float x = point.x * a;
		float y = point.y * a;
		float z = 1.0f - r2;

		return x * onb.u + y * onb.v + z * onb.w;
	}
}

Vector2 Sampler::getDiscSample(uint64_t x, uint64_t y, uint64_t nx, uint64_t ny, uint64_t permutation, Random& random)
{
	return mapSquareToDisc(getSquareSample(x, y, nx, ny, permutation, random));
}

Vector3 Sampler::getCosineHemisphereSample(const ONB& onb, uint64_t x, uint64_t y, uint64_t nx, uint64_t ny, uint64_t permutation, Random& random)
{
	return mapDiscToCosineHemisphere(onb, getDiscSample(x, y, nx, ny, permutation, random));
}

Vector3 Sampler::getUniformHemisphereSample(const ONB& onb, uint64_t x, uint64_t y, uint64_t nx, uint64_t ny, uint64_t permutation, Random& random)
{
	return mapDiscToUniformHemisphere(onb, getDiscSample(x, y, nx, ny, permutation, random));
}

void Sampler::generateSamples1D(uint64_t sampleCount, Random& random)
{
	samples1D.resize(sampleCount);
	uint64_t permutation = random.getUint64();

	for (uint64_t i = 0; i < sampleCount; ++i)
		samples1D[i] = getSample(i, sampleCount, permutation, random);

	nextSampleIndex1D = 0;
}

void Sampler::generateSamples2D(uint64_t sampleCountSqrt, Random& random)
{
	samples2D.resize(sampleCountSqrt * sampleCountSqrt);
	uint64_t permutation = random.getUint64();

	for (uint64_t y = 0; y < sampleCountSqrt; ++y)
	{
		for (uint64_t x = 0; x < sampleCountSqrt; ++x)
		{
			samples2D[y * sampleCountSqrt + x] = getSquareSample(x, y, sampleCountSqrt, sampleCountSqrt, permutation, random);
		}
	}

	nextSampleIndex2D = 0;
}

bool Sampler::getNextSample(float& result)
{
	if (nextSampleIndex1D >= samples1D.size())
	{
		nextSampleIndex1D = 0;
		return false;
	}

	result = samples1D[nextSampleIndex1D++];
	return true;
}

bool Sampler::getNextSquareSample(Vector2& result)
{
	if (nextSampleIndex2D >= samples2D.size())
	{
		nextSampleIndex2D = 0;
		return false;
	}

	result = samples2D[nextSampleIndex2D++];
	return true;
}

bool Sampler::getNextDiscSample(Vector2& result)
{
	if (nextSampleIndex2D >= samples2D.size())
	{
		nextSampleIndex2D = 0;
		return false;
	}

	result = mapSquareToDisc(samples2D[nextSampleIndex2D++]);
	return true;
}

bool Sampler::getNextCosineHemisphereSample(const ONB& onb, Vector3& result)
{
	if (nextSampleIndex2D >= samples2D.size())
	{
		nextSampleIndex2D = 0;
		return false;
	}

	result = mapDiscToCosineHemisphere(onb, mapSquareToDisc(samples2D[nextSampleIndex2D++]));
	return true;
}

bool Sampler::getNextUniformHemisphereSample(const ONB& onb, Vector3& result)
{
	if (nextSampleIndex2D >= samples2D.size())
	{
		nextSampleIndex2D = 0;
		return false;
	}

	result = mapDiscToUniformHemisphere(onb, mapSquareToDisc(samples2D[nextSampleIndex2D++]));
	return true;
}

void Sampler::reset()
{
	nextSampleIndex1D = 0;
	nextSampleIndex2D = 0;
}
