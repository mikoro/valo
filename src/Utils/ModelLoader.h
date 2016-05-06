// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <map>
#include <vector>

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
		uint32_t defaultMaterialId = 0;
		uint32_t triangleCountEstimate = 0;
		bool loadOnlyMaterials = false;
		bool substituteMaterial = false;
		std::string substituteMaterialFileName;
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
		bool processFace(const char* buffer, uint32_t lineStartIndex, uint32_t lineEndIndex, uint32_t lineNumber, ModelLoaderResult& result);

		uint32_t currentMaterialId = 0;
		uint32_t materialIdCounter = 1000;
		uint32_t currentTextureId = 1000;

		std::vector<Vector3> vertices;
		std::vector<Vector3> normals;
		std::vector<Vector2> texcoords;

		std::map<std::string, uint32_t> materialsMap;
		std::map<std::string, uint32_t> externalMaterialsMap;
	};
}
