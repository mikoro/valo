// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <vector>

#include "cereal/cereal.hpp"

#include "Core/Triangle.h"
#include "Materials/Material.h"
#include "Math/EulerAngle.h"
#include "Math/Vector3.h"
#include "Textures/Texture.h"

/*

Only OBJ (and MTL) files are supported.

Restrictions:
 - numbers cannot have scientific notation

*/

namespace Raycer
{
	struct ModelLoaderInfo
	{
		std::string modelFileName;
		Vector3 scale = Vector3(1.0f, 1.0f, 1.0f);
		EulerAngle rotate = EulerAngle(0.0f, 0.0f, 0.0f);
		Vector3 translate = Vector3(0.0f, 0.0f, 0.0f);
		uint64_t defaultMaterialId = 0;
		uint64_t triangleCountEstimate = 0;
		bool loadOnlyMaterials = false;
		
		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(modelFileName),
				CEREAL_NVP(scale),
				CEREAL_NVP(rotate),
				CEREAL_NVP(translate),
				CEREAL_NVP(defaultMaterialId),
				CEREAL_NVP(triangleCountEstimate),
				CEREAL_NVP(loadOnlyMaterials));
		}
	};

	struct ModelLoaderResult
	{
		std::vector<Texture> textures;
		std::vector<Material> materials;
		std::vector<Triangle> triangles;
	};

	class ModelLoader
	{
	public:

		ModelLoaderResult load(const ModelLoaderInfo& info);

	private:

		void processMaterialFile(const std::string& rootDirectory, const std::string& mtlFileName, ModelLoaderResult& result);
		bool processFace(const char* buffer, uint64_t lineStartIndex, uint64_t lineEndIndex, uint64_t lineNumber, ModelLoaderResult& result);

		uint64_t currentMaterialId = 0;
		uint64_t materialIdCounter = 1000;
		uint64_t currentTextureId = 1000;

		std::vector<Vector3> vertices;
		std::vector<Vector3> normals;
		std::vector<Vector2> texcoords;

		std::map<std::string, uint64_t> materialsMap;
		std::map<std::string, uint64_t> externalMaterialsMap;
	};
}
