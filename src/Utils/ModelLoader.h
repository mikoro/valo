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
		uint64_t triangleCountEstimate = 0;
		
		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(modelFilePath),
				CEREAL_NVP(scale),
				CEREAL_NVP(rotate),
				CEREAL_NVP(translate),
				CEREAL_NVP(defaultMaterialId),
				CEREAL_NVP(triangleCountEstimate));
		}
	};

	struct ModelLoaderResult
	{
		std::vector<Triangle> triangles;
		std::vector<DiffuseSpecularMaterial> diffuseSpecularMaterials;
		std::vector<ImageTexture> imageTextures;
	};

	class ModelLoader
	{
	public:

		ModelLoaderResult loadAll(const ModelLoaderInfo& info);
		ModelLoaderResult loadMaterials(const ModelLoaderInfo& info);

	private:

		void processMaterialFile(const std::string& rootDirectory, const std::string& mtlFilePath, ModelLoaderResult& result);
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
