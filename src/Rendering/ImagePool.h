// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <map>
#include <vector>

#include "cereal/cereal.hpp"

#include "Rendering/Image.h"

namespace Raycer
{
	class ImagePool
	{
	public:

		const Imagef* getImage(const std::string& fileName, bool applyGamma);
		uint64_t getImageIndex(const std::string& fileName);
		const std::vector<Imagef>& getImages();
		void clear();

	private:

		bool initialized = false;
		std::map<std::string, uint64_t> imageIndexMap;
		std::vector<Imagef> images;

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(initialized),
				CEREAL_NVP(imageIndexMap),
				CEREAL_NVP(images));
		}
	};
}
