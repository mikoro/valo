// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <vector>

#include "cereal/cereal.hpp"

#include "Tracing/Triangle.h"
#include "Textures/ImageTexture.h"
#include "Materials/DiffuseSpecularMaterial.h"
#include "Math/Vector3.h"
#include "Math/EulerAngle.h"

/*

Only OBJ (and MTL) files are supported.

Restrictions:
 - numbers cannot have scientific notation

*/

namespace Raycer
{
	struct ModelLoaderInfo
	{
		std::string modelFilePath;
		Vector3 scale = Vector3(1.0f, 1.0f, 1.0f);
		EulerAngle rotate = EulerAngle(0.0f, 0.0f, 0.0f);
		Vector3 translate = Vector3(0.0f, 0.0f, 0.0f);
		uint64_t defaultMaterialId = 0;
		uint64_t idStartOffset = 0;
		uint64_t triangleCountEstimate = 0;
		
		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(modelFilePath),
				CEREAL_NVP(scale),
				CEREAL_NVP(rotate),
				CEREAL_NVP(translate),
				CEREAL_NVP(defaultMaterialId),
				CEREAL_NVP(idStartOffset));
		}
	};

	struct ModelLoaderResult
	{
		std::vector<Triangle> triangles;
		std::vector<DiffuseSpecularMaterial> diffuseSpecularMaterials;
		std::vector<ImageTexture> textures;
	};

	class ModelLoader
	{
	public:

		static ModelLoaderResult load(const ModelLoaderInfo& info);
	};
}
