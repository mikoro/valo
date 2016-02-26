// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <atomic>

namespace Raycer
{
	class Scene;
	class Film;

	struct TracerState
	{
		TracerState() : sampleCount(0), pixelSampleCount(0), pixelCount(0), rayCount(0), pathCount(0) {};

		Scene* scene = nullptr;
		Film* film = nullptr;
		
		uint64_t filmWidth = 0;
		uint64_t filmHeight = 0;
		uint64_t filmPixelOffset = 0;
		uint64_t filmPixelCount = 0;

		std::atomic<uint64_t> sampleCount;
		std::atomic<uint64_t> pixelSampleCount;
		std::atomic<uint64_t> pixelCount;
		std::atomic<uint64_t> rayCount;
		std::atomic<uint64_t> pathCount;
	};
}
