// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#ifdef RUN_UNIT_TESTS

#include "catch/catch.hpp"

#include "Samplers/Sampler.h"
#include "Samplers/RandomSampler.h"
#include "Samplers/RegularSampler.h"
#include "Samplers/JitteredSampler.h"
#include "Samplers/CMJSampler.h"
#include "Samplers/PoissonDiscSampler.h"
#include "Rendering/Image.h"
#include "Tracing/ONB.h"
#include "Rendering/Color.h"
#include "Math/Vector2.h"
#include "Math/Vector3.h"
#include "Utils/Random.h"
#include "Tracing/Triangle.h"
#include "Tracing/Intersection.h"
#include "Materials/DiffuseSpecularMaterial.h"

using namespace Raycer;

TEST_CASE("Sampler functionality", "[sampler]")
{
	RandomSampler randomSampler;
	RegularSampler regularSampler;
	JitteredSampler jitteredSampler;
	CMJSampler cmjSampler;
	PoissonDiscSampler poissonDiscSampler;

	std::map<std::string, Sampler*> samplers;
	samplers["random"] = &randomSampler;
	samplers["regular"] = &regularSampler;
	samplers["jittered"] = &jitteredSampler;
	samplers["cmj"] = &cmjSampler;
	samplers["poisson_disc"] = &poissonDiscSampler;

	for (const auto &sampler : samplers)
	{
		uint64_t sampleCount = 32;
		Vector2 size = Vector2(99.0, 99.0);

		Image image1(100, 100);
		Image image2(100, 100);
		Image image3(100, 100);
		
		std::ofstream file1(tfm::format("sampler_%s_cosine_hemisphere.txt", sampler.first));
		std::ofstream file2(tfm::format("sampler_%s_uniform_hemisphere.txt", sampler.first));

		Random random;

		sampler.second->generateSamples1D(sampleCount, random);
		sampler.second->generateSamples2D(sampleCount, random);

		double sample1D;

		while (sampler.second->getNextSample(sample1D))
		{
			sample1D *= size.x;
			image1.setPixel(uint64_t(sample1D + 0.5), uint64_t(size.y / 2.0 + 0.5), Color(255, 255, 255));
		}

		Vector2 sample2D;

		while (sampler.second->getNextSquareSample(sample2D))
		{
			sample2D *= size;
			image2.setPixel(uint64_t(sample2D.x + 0.5), uint64_t(sample2D.y + 0.5), Color(255, 255, 255));
		}

		while (sampler.second->getNextDiscSample(sample2D))
		{
			sample2D = (sample2D / 2.0 + Vector2(0.5, 0.5)) * size;
			image3.setPixel(uint64_t(sample2D.x + 0.5), uint64_t(sample2D.y + 0.5), Color(255, 255, 255));
		}

		Vector3 sample3D;
		
		while (sampler.second->getNextCosineHemisphereSample(ONB::UP, sample3D))
			file1 << tfm::format("%f %f %f\n", sample3D.x, sample3D.y, sample3D.z);

		while (sampler.second->getNextUniformHemisphereSample(ONB::UP, sample3D))
			file2 << tfm::format("%f %f %f\n", sample3D.x, sample3D.y, sample3D.z);

		image1.save(tfm::format("sampler_%s_1D.png", sampler.first));
		image2.save(tfm::format("sampler_%s_2D.png", sampler.first));
		image3.save(tfm::format("sampler_%s_disc.png", sampler.first));

		file1.close();
		file2.close();
	}
}

TEST_CASE("Triangle sampler functionality", "[sampler]")
{
	Triangle triangle;
	DiffuseSpecularMaterial material;

	triangle.material = &material;

	triangle.vertices[0] = Vector3(0.1, 0.1, 0.0);
	triangle.vertices[1] = Vector3(0.9, 0.1, 0.0);
	triangle.vertices[2] = Vector3(0.5, 0.9, 0.0);

	Random random;
	Image image1(100, 100);

	for (uint64_t i = 0; i < 1000; ++i)
	{
		Intersection intersection = triangle.getRandomIntersection(random);

		uint64_t x = uint64_t(intersection.position.x * 99.0 + 0.5);
		uint64_t y = uint64_t(intersection.position.y * 99.0 + 0.5);

		image1.setPixel(x, y, Color(255, 255, 255));
	}

	image1.save("sampler_triangle.png");
}

#endif
