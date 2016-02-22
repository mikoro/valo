// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Utils/ModelLoader.h"
#include "App.h"
#include "Utils/Log.h"
#include "Utils/StringUtils.h"
#include "Utils/Timer.h"
#include "Math/Vector2.h"
#include "Math/Vector3.h"
#include "Rendering/Color.h"
#include "Math/Matrix4x4.h"

using namespace Raycer;
using namespace boost::filesystem;

namespace
{
	std::string getAbsolutePath(const std::string& rootDirectory, const std::string& relativeFilePath)
	{
		path tempPath(rootDirectory);
		tempPath.append(relativeFilePath.begin(), relativeFilePath.end());
		std::string tempPathString = tempPath.string();
		std::replace(tempPathString.begin(), tempPathString.end(), '\\', '/');
		return tempPathString;
	}

	float readFloat(const std::string& input, uint64_t& startIndex, std::string& result)
	{
		StringUtils::readUntilSpace(input, startIndex, result);
		return StringUtils::parseFloat(result);
	}

	void processMaterialFile(const std::string& objFileDirectory, const std::string& mtlFilePath, ModelLoaderResult& result, std::map<std::string, uint64_t>& materialsMap, std::map<std::string, uint64_t>& externalMaterialsMap, uint64_t& currentId)
	{
		std::string absoluteMtlFilePath = getAbsolutePath(objFileDirectory, mtlFilePath);
		App::getLog().logInfo("Reading MTL file (%s)", absoluteMtlFilePath);
		std::ifstream file(absoluteMtlFilePath);

		if (!file.good())
			throw std::runtime_error("Could not open the MTL file");

		DiffuseSpecularMaterial currentMaterial;
		bool materialPending = false;

		std::string line;
		std::string part;
		std::string currentMaterialName;

		while (std::getline(file, line))
		{
			std::stringstream ss(line);
			ss >> part;

			if (part == "newmtl") // new material
			{
				if (materialPending)
					result.diffuseSpecularMaterials.push_back(currentMaterial);

				materialPending = true;

				currentMaterial = DiffuseSpecularMaterial();
				currentMaterial.id = ++currentId;

				ss >> currentMaterialName;
				materialsMap[currentMaterialName] = currentMaterial.id;
			}
			else if (part == "materialId")
				ss >> externalMaterialsMap[currentMaterialName];
			else if (part == "skipLighting")
				ss >> currentMaterial.skipLighting;
			else if (part == "nonShadowing")
				ss >> currentMaterial.nonShadowing;
			else if (part == "normalInterpolation")
				ss >> currentMaterial.normalInterpolation;
			else if (part == "autoInvertNormal")
				ss >> currentMaterial.autoInvertNormal;
			else if (part == "invertNormal")
				ss >> currentMaterial.invertNormal;
			else if (part == "fresnelReflection")
				ss >> currentMaterial.fresnelReflection;
			else if (part == "attenuating")
				ss >> currentMaterial.attenuating;
			else if (part == "shininess" || part == "Ns")
				ss >> currentMaterial.shininess;
			else if (part == "refractiveIndex" || part == "Ni")
				ss >> currentMaterial.refractiveIndex;
			else if (part == "rayReflectance")
				ss >> currentMaterial.rayReflectance;
			else if (part == "rayTransmittance")
				ss >> currentMaterial.rayTransmittance;
			else if (part == "attenuationFactor")
				ss >> currentMaterial.attenuationFactor;
			else if (part == "attenuationColor")
			{
				ss >> currentMaterial.attenuationColor.r;
				ss >> currentMaterial.attenuationColor.g;
				ss >> currentMaterial.attenuationColor.b;
			}
			else if (part == "texcoordScale")
			{
				ss >> currentMaterial.texcoordScale.x;
				ss >> currentMaterial.texcoordScale.y;
			}
			else if (part == "reflectance" || part == "Kr")
			{
				ss >> currentMaterial.reflectance.r;
				ss >> currentMaterial.reflectance.g;
				ss >> currentMaterial.reflectance.b;
			}
			else if ((part == "reflectanceMap" || part == "map_Kr") && currentMaterial.reflectanceMapTextureId == 0)
			{
				ImageTexture imageTexture;
				imageTexture.id = ++currentId;
				currentMaterial.reflectanceMapTextureId = imageTexture.id;

				ss >> part;
				imageTexture.imageFilePath = getAbsolutePath(objFileDirectory, part);
				imageTexture.applyGamma = !StringUtils::endsWith(imageTexture.imageFilePath, ".hdr");

				result.textures.push_back(imageTexture);
			}
			else if (part == "emittance" || part == "Ke")
			{
				ss >> currentMaterial.emittance.r;
				ss >> currentMaterial.emittance.g;
				ss >> currentMaterial.emittance.b;
			}
			else if ((part == "emittanceMap" || part == "map_Ke") && currentMaterial.emittanceMapTextureId == 0)
			{
				ImageTexture imageTexture;
				imageTexture.id = ++currentId;
				currentMaterial.emittanceMapTextureId = imageTexture.id;

				ss >> part;
				imageTexture.imageFilePath = getAbsolutePath(objFileDirectory, part);
				imageTexture.applyGamma = !StringUtils::endsWith(imageTexture.imageFilePath, ".hdr");

				result.textures.push_back(imageTexture);
			}
			else if (part == "ambientReflectance" || part == "Ka")
			{
				ss >> currentMaterial.ambientReflectance.r;
				ss >> currentMaterial.ambientReflectance.g;
				ss >> currentMaterial.ambientReflectance.b;
			}
			else if ((part == "ambientMap" || part == "map_Ka") && currentMaterial.ambientMapTextureId == 0)
			{
				ImageTexture imageTexture;
				imageTexture.id = ++currentId;
				currentMaterial.ambientMapTextureId = imageTexture.id;

				ss >> part;
				imageTexture.imageFilePath = getAbsolutePath(objFileDirectory, part);
				imageTexture.applyGamma = !StringUtils::endsWith(imageTexture.imageFilePath, ".hdr");

				result.textures.push_back(imageTexture);
			}
			else if (part == "diffuseReflectance" || part == "Kd")
			{
				ss >> currentMaterial.diffuseReflectance.r;
				ss >> currentMaterial.diffuseReflectance.g;
				ss >> currentMaterial.diffuseReflectance.b;

				// for compatability
				currentMaterial.reflectance.r = currentMaterial.diffuseReflectance.r;
				currentMaterial.reflectance.g = currentMaterial.diffuseReflectance.g;
				currentMaterial.reflectance.b = currentMaterial.diffuseReflectance.b;
			}
			else if ((part == "diffuseMap" || part == "map_Kd") && currentMaterial.diffuseMapTextureId == 0)
			{
				ImageTexture imageTexture;
				imageTexture.id = ++currentId;
				currentMaterial.diffuseMapTextureId = imageTexture.id;
				currentMaterial.reflectanceMapTextureId = imageTexture.id;

				ss >> part;
				imageTexture.imageFilePath = getAbsolutePath(objFileDirectory, part);
				imageTexture.applyGamma = !StringUtils::endsWith(imageTexture.imageFilePath, ".hdr");

				result.textures.push_back(imageTexture);
			}
			else if (part == "specularReflectance" || part == "Ks")
			{
				ss >> currentMaterial.specularReflectance.r;
				ss >> currentMaterial.specularReflectance.g;
				ss >> currentMaterial.specularReflectance.b;
			}
			else if ((part == "specularMap" || part == "map_Ks") && currentMaterial.specularMapTextureId == 0)
			{
				ImageTexture imageTexture;
				imageTexture.id = ++currentId;
				currentMaterial.specularMapTextureId = imageTexture.id;

				ss >> part;
				imageTexture.imageFilePath = getAbsolutePath(objFileDirectory, part);
				imageTexture.applyGamma = !StringUtils::endsWith(imageTexture.imageFilePath, ".hdr");

				result.textures.push_back(imageTexture);
			}
			else if ((part == "normalMap" || part == "map_normal") && currentMaterial.normalMapTextureId == 0)
			{
				ImageTexture imageTexture;
				imageTexture.id = ++currentId;
				currentMaterial.normalMapTextureId = imageTexture.id;

				ss >> part;
				imageTexture.imageFilePath = getAbsolutePath(objFileDirectory, part);
				imageTexture.applyGamma = false;

				result.textures.push_back(imageTexture);
			}
			else if ((part == "maskMap" || part == "map_d") && currentMaterial.maskMapTextureId == 0)
			{
				ImageTexture imageTexture;
				imageTexture.id = ++currentId;
				currentMaterial.maskMapTextureId = imageTexture.id;

				ss >> part;
				imageTexture.imageFilePath = getAbsolutePath(objFileDirectory, part);
				imageTexture.applyGamma = false;

				result.textures.push_back(imageTexture);
			}
		}

		file.close();

		if (materialPending)
			result.diffuseSpecularMaterials.push_back(currentMaterial);
	}

	bool processFace(const std::string& line, std::vector<Vector3>& vertices, std::vector<Vector3>& normals, std::vector<Vector2>& texcoords, ModelLoaderResult& result, uint64_t& currentId, uint64_t currentMaterialId)
	{
		Log& log = App::getLog();

		uint64_t vertexIndices[4];
		uint64_t normalIndices[4];
		uint64_t texcoordIndices[4];

		std::string part1;
		uint64_t lineIndex1 = 1;
		uint64_t vertexCount = 0;

		StringUtils::readUntilSpace(line, lineIndex1, part1);

		uint64_t slashCount = std::count(part1.begin(), part1.end(), '/');
		bool doubleSlash = (part1.find("//") != std::string::npos);
		bool hasTexcoords = (slashCount > 0 && !doubleSlash);
		bool hasNormals = (slashCount > 1);

		lineIndex1 = 1;

		for (uint64_t i = 0; i < 4; ++i)
		{
			if (!StringUtils::readUntilSpace(line, lineIndex1, part1))
				break;

			vertexCount++;

			std::replace(part1.begin(), part1.end(), '/', ' ');

			std::string part2;
			uint64_t lineIndex2 = 0;

			StringUtils::readUntilSpace(part1, lineIndex2, part2);
			int64_t vertexIndex = strtoll(part2.c_str(), nullptr, 10);

			if (vertexIndex < 0)
				vertexIndex = int64_t(vertices.size()) + vertexIndex;
			else
				vertexIndex--;

			if (vertexIndex < 0 || vertexIndex >= int64_t(vertices.size()))
			{
				log.logWarning("Vertex index (%s) was out of bounds", vertexIndex);
				return false;
			}

			vertexIndices[i] = uint64_t(vertexIndex);

			if (hasTexcoords)
			{
				StringUtils::readUntilSpace(part1, lineIndex2, part2);
				int64_t texcoordIndex = strtoll(part2.c_str(), nullptr, 10);

				if (texcoordIndex < 0)
					texcoordIndex = int64_t(texcoords.size()) + texcoordIndex;
				else
					texcoordIndex--;

				if (texcoordIndex < 0 || texcoordIndex >= int64_t(texcoords.size()))
				{
					log.logWarning("Texcoord index (%s) was out of bounds", texcoordIndex);
					return false;
				}

				texcoordIndices[i] = uint64_t(texcoordIndex);
			}

			if (hasNormals)
			{
				StringUtils::readUntilSpace(part1, lineIndex2, part2);
				int64_t normalIndex = strtoll(part2.c_str(), nullptr, 10);

				if (normalIndex < 0)
					normalIndex = int64_t(normals.size()) + normalIndex;
				else
					normalIndex--;

				if (normalIndex < 0 || normalIndex >= int64_t(normals.size()))
				{
					log.logWarning("Normal index (%s) was out of bounds", normalIndex);
					return false;
				}

				normalIndices[i] = uint64_t(normalIndex);
			}
		}

		if (vertexCount < 3)
		{
			log.logWarning("Too few vertices (%s) in a face", vertexCount);
			return false;
		}

		// triangulate
		for (uint64_t i = 2; i < vertexCount; ++i)
		{
			Triangle triangle;
			triangle.id = ++currentId;
			triangle.materialId = currentMaterialId;

			triangle.vertices[0] = vertices[vertexIndices[0]];
			triangle.vertices[1] = vertices[vertexIndices[i - 1]];
			triangle.vertices[2] = vertices[vertexIndices[i]];

			if (hasNormals)
			{
				triangle.normals[0] = normals[normalIndices[0]];
				triangle.normals[1] = normals[normalIndices[i - 1]];
				triangle.normals[2] = normals[normalIndices[i]];
			}
			else
			{
				Vector3 v0tov1 = triangle.vertices[1] - triangle.vertices[0];
				Vector3 v0tov2 = triangle.vertices[2] - triangle.vertices[0];
				Vector3 normal = v0tov1.cross(v0tov2).normalized();

				triangle.normals[0] = triangle.normals[1] = triangle.normals[2] = normal;
			}

			if (hasTexcoords)
			{
				triangle.texcoords[0] = texcoords[texcoordIndices[0]];
				triangle.texcoords[1] = texcoords[texcoordIndices[i - 1]];
				triangle.texcoords[2] = texcoords[texcoordIndices[i]];
			}

			result.triangles.push_back(triangle);
		}

		return true;
	}
}

ModelLoaderResult ModelLoader::load(const ModelLoaderInfo& info)
{
	Log& log = App::getLog();

	log.logInfo("Reading OBJ file (%s)", info.modelFilePath);

	Timer timer;
	ModelLoaderResult result;

	uint64_t currentId = info.idStartOffset;
	uint64_t currentMaterialId = info.defaultMaterialId;

	std::map<std::string, uint64_t> materialsMap;
	std::map<std::string, uint64_t> externalMaterialsMap;
	std::string objFileDirectory = boost::filesystem::absolute(info.modelFilePath).parent_path().string();

	Matrix4x4 scaling = Matrix4x4::scale(info.scale);
	Matrix4x4 rotation = Matrix4x4::rotateXYZ(info.rotate);
	Matrix4x4 translation = Matrix4x4::translate(info.translate);
	Matrix4x4 transformation = translation * rotation * scaling;
	Matrix4x4 transformationInvT = transformation.inverted().transposed();

	std::vector<Vector3> vertices;
	std::vector<Vector3> normals;
	std::vector<Vector2> texcoords;

	result.triangles.reserve(info.triangleCountEstimate);
	vertices.reserve(info.triangleCountEstimate / 2);
	normals.reserve(info.triangleCountEstimate / 2);
	texcoords.reserve(info.triangleCountEstimate / 2);

	FILE* file = fopen(info.modelFilePath.c_str(), "r");

	if (file == nullptr)
		throw std::runtime_error("Could not open the OBJ file");

	char buffer[1024];
	std::string line;
	std::string part;
	uint64_t lineIndex = 0;

	while (fgets(buffer, sizeof(buffer), file) != nullptr)
	{
		line.assign(buffer);
		part.clear();
		lineIndex = 0;
		StringUtils::readUntilSpace(line, lineIndex, part);

		if (part.size() == 0)
			continue;

		if (part == "mtllib") // new material file
		{
			StringUtils::readUntilSpace(line, lineIndex, part);
			processMaterialFile(objFileDirectory, part, result, materialsMap, externalMaterialsMap, currentId);
		}
		else if (part == "usemtl") // select material
		{
			StringUtils::readUntilSpace(line, lineIndex, part);

			if (externalMaterialsMap.count(part))
				currentMaterialId = externalMaterialsMap[part];
			else if (materialsMap.count(part))
				currentMaterialId = materialsMap[part];
			else
			{
				log.logWarning("Could not find material named \"%s\"", part);
				currentMaterialId = info.defaultMaterialId;
			}
		}
		else if (part == "v") // vertex
		{
			Vector3 vertex;

			vertex.x = readFloat(line, lineIndex, part);
			vertex.y = readFloat(line, lineIndex, part);
			vertex.z = readFloat(line, lineIndex, part);

			vertices.push_back(transformation.transformPosition(vertex));
		}
		else if (part == "vn") // normal
		{
			Vector3 normal;

			normal.x = readFloat(line, lineIndex, part);
			normal.y = readFloat(line, lineIndex, part);
			normal.z = readFloat(line, lineIndex, part);

			normals.push_back(transformationInvT.transformDirection(normal).normalized());
		}
		else if (part == "vt") // texcoord
		{
			Vector2 texcoord;

			texcoord.x = readFloat(line, lineIndex, part);
			texcoord.y = readFloat(line, lineIndex, part);

			texcoords.push_back(texcoord);
		}
		else if (part == "f") // face
		{
			if (!processFace(line, vertices, normals, texcoords, result, currentId, currentMaterialId))
				break;
		}
	}

	fclose(file);

	log.logInfo("OBJ file reading finished (time: %s, vertices: %s, normals: %s, texcoords: %s, triangles: %s, materials: %s, textures: %s)", timer.getElapsed().getString(true), vertices.size(), normals.size(), texcoords.size(), result.triangles.size(), result.diffuseSpecularMaterials.size(), result.textures.size());

	return result;
}
