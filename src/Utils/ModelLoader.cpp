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
		uint64_t lineIndex = 0;

		while (std::getline(file, line))
		{
			part.clear();
			lineIndex = 0;
			StringUtils::readUntilSpace(line, lineIndex, part);

			if (part.size() == 0)
				continue;

			if (part == "newmtl") // new material
			{
				if (materialPending)
					result.diffuseSpecularMaterials.push_back(currentMaterial);

				materialPending = true;

				currentMaterial = DiffuseSpecularMaterial();
				currentMaterial.id = ++currentId;

				StringUtils::readUntilSpace(line, lineIndex, currentMaterialName);
				materialsMap[currentMaterialName] = currentMaterial.id;
			}
			else if (part == "materialId")
			{
				externalMaterialsMap[currentMaterialName] = uint64_t(readFloat(line, lineIndex, part));
			}
			else if (part == "skipLighting")
			{
				currentMaterial.skipLighting = readFloat(line, lineIndex, part) != 0.0;
			}
			else if (part == "nonShadowing")
			{
				currentMaterial.nonShadowing = readFloat(line, lineIndex, part) != 0.0;
			}
			else if (part == "normalInterpolation")
			{
				currentMaterial.normalInterpolation = readFloat(line, lineIndex, part) != 0.0;
			}
			else if (part == "autoInvertNormal")
			{
				currentMaterial.autoInvertNormal = readFloat(line, lineIndex, part) != 0.0;
			}
			else if (part == "invertNormal")
			{
				currentMaterial.invertNormal = readFloat(line, lineIndex, part) != 0.0;
			}
			else if (part == "fresnelReflection")
			{
				currentMaterial.fresnelReflection = readFloat(line, lineIndex, part) != 0.0;
			}
			else if (part == "attenuating")
			{
				currentMaterial.attenuating = readFloat(line, lineIndex, part) != 0.0;
			}
			else if (part == "shininess" || part == "Ns")
			{
				currentMaterial.shininess = readFloat(line, lineIndex, part);
			}
			else if (part == "refractiveIndex" || part == "Ni")
			{
				currentMaterial.refractiveIndex = readFloat(line, lineIndex, part);
			}
			else if (part == "rayReflectance")
			{
				currentMaterial.rayReflectance = readFloat(line, lineIndex, part);
			}
			else if (part == "rayTransmittance")
			{
				currentMaterial.rayTransmittance = readFloat(line, lineIndex, part);
			}
			else if (part == "attenuationFactor")
			{
				currentMaterial.attenuationFactor = readFloat(line, lineIndex, part);
			}
			else if (part == "attenuationColor")
			{
				currentMaterial.attenuationColor.r = readFloat(line, lineIndex, part);
				currentMaterial.attenuationColor.g = readFloat(line, lineIndex, part);
				currentMaterial.attenuationColor.b = readFloat(line, lineIndex, part);
			}
			else if (part == "texcoordScale")
			{
				currentMaterial.texcoordScale.x = readFloat(line, lineIndex, part);
				currentMaterial.texcoordScale.y = readFloat(line, lineIndex, part);
			}
			else if (part == "reflectance" || part == "Kr")
			{
				currentMaterial.reflectance.r = readFloat(line, lineIndex, part);
				currentMaterial.reflectance.g = readFloat(line, lineIndex, part);
				currentMaterial.reflectance.b = readFloat(line, lineIndex, part);
			}
			else if ((part == "reflectanceMap" || part == "map_Kr") && currentMaterial.reflectanceMapTextureId == 0)
			{
				ImageTexture imageTexture;
				imageTexture.id = ++currentId;
				currentMaterial.reflectanceMapTextureId = imageTexture.id;

				StringUtils::readUntilSpace(line, lineIndex, part);
				imageTexture.imageFilePath = getAbsolutePath(objFileDirectory, part);
				imageTexture.applyGamma = !StringUtils::endsWith(imageTexture.imageFilePath, ".hdr");

				result.textures.push_back(imageTexture);
			}
			else if (part == "emittance" || part == "Ke")
			{
				currentMaterial.emittance.r = readFloat(line, lineIndex, part);
				currentMaterial.emittance.g = readFloat(line, lineIndex, part);
				currentMaterial.emittance.b = readFloat(line, lineIndex, part);
			}
			else if ((part == "emittanceMap" || part == "map_Ke") && currentMaterial.emittanceMapTextureId == 0)
			{
				ImageTexture imageTexture;
				imageTexture.id = ++currentId;
				currentMaterial.emittanceMapTextureId = imageTexture.id;

				StringUtils::readUntilSpace(line, lineIndex, part);
				imageTexture.imageFilePath = getAbsolutePath(objFileDirectory, part);
				imageTexture.applyGamma = !StringUtils::endsWith(imageTexture.imageFilePath, ".hdr");

				result.textures.push_back(imageTexture);
			}
			else if (part == "ambientReflectance" || part == "Ka")
			{
				currentMaterial.ambientReflectance.r = readFloat(line, lineIndex, part);
				currentMaterial.ambientReflectance.g = readFloat(line, lineIndex, part);
				currentMaterial.ambientReflectance.b = readFloat(line, lineIndex, part);
			}
			else if ((part == "ambientMap" || part == "map_Ka") && currentMaterial.ambientMapTextureId == 0)
			{
				ImageTexture imageTexture;
				imageTexture.id = ++currentId;
				currentMaterial.ambientMapTextureId = imageTexture.id;

				StringUtils::readUntilSpace(line, lineIndex, part);
				imageTexture.imageFilePath = getAbsolutePath(objFileDirectory, part);
				imageTexture.applyGamma = !StringUtils::endsWith(imageTexture.imageFilePath, ".hdr");

				result.textures.push_back(imageTexture);
			}
			else if (part == "diffuseReflectance" || part == "Kd")
			{
				currentMaterial.diffuseReflectance.r = readFloat(line, lineIndex, part);
				currentMaterial.diffuseReflectance.g = readFloat(line, lineIndex, part);
				currentMaterial.diffuseReflectance.b = readFloat(line, lineIndex, part);

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

				StringUtils::readUntilSpace(line, lineIndex, part);
				imageTexture.imageFilePath = getAbsolutePath(objFileDirectory, part);
				imageTexture.applyGamma = !StringUtils::endsWith(imageTexture.imageFilePath, ".hdr");

				result.textures.push_back(imageTexture);
			}
			else if (part == "specularReflectance" || part == "Ks")
			{
				currentMaterial.specularReflectance.r = readFloat(line, lineIndex, part);
				currentMaterial.specularReflectance.g = readFloat(line, lineIndex, part);
				currentMaterial.specularReflectance.b = readFloat(line, lineIndex, part);
			}
			else if ((part == "specularMap" || part == "map_Ks") && currentMaterial.specularMapTextureId == 0)
			{
				ImageTexture imageTexture;
				imageTexture.id = ++currentId;
				currentMaterial.specularMapTextureId = imageTexture.id;

				StringUtils::readUntilSpace(line, lineIndex, part);
				imageTexture.imageFilePath = getAbsolutePath(objFileDirectory, part);
				imageTexture.applyGamma = !StringUtils::endsWith(imageTexture.imageFilePath, ".hdr");

				result.textures.push_back(imageTexture);
			}
			else if ((part == "normalMap" || part == "map_normal") && currentMaterial.normalMapTextureId == 0)
			{
				ImageTexture imageTexture;
				imageTexture.id = ++currentId;
				currentMaterial.normalMapTextureId = imageTexture.id;

				StringUtils::readUntilSpace(line, lineIndex, part);
				imageTexture.imageFilePath = getAbsolutePath(objFileDirectory, part);
				imageTexture.applyGamma = false;

				result.textures.push_back(imageTexture);
			}
			else if ((part == "maskMap" || part == "map_d") && currentMaterial.maskMapTextureId == 0)
			{
				ImageTexture imageTexture;
				imageTexture.id = ++currentId;
				currentMaterial.maskMapTextureId = imageTexture.id;

				StringUtils::readUntilSpace(line, lineIndex, part);
				imageTexture.imageFilePath = getAbsolutePath(objFileDirectory, part);
				imageTexture.applyGamma = false;

				result.textures.push_back(imageTexture);
			}
		}

		file.close();

		if (materialPending)
			result.diffuseSpecularMaterials.push_back(currentMaterial);
	}

	void processFace(const std::string& line, std::vector<Vector3>& vertices, std::vector<Vector3>& normals, std::vector<Vector2>& texcoords, ModelLoaderResult& result, uint64_t& currentId, uint64_t currentMaterialId)
	{
		Log& log = App::getLog();

		std::vector<uint64_t> vertexIndices;
		std::vector<uint64_t> normalIndices;
		std::vector<uint64_t> texcoordIndices;

		std::string part1;
		uint64_t lineIndex1 = 0;

		StringUtils::readUntilSpace(line, lineIndex1, part1);

		// determine what indices are available from the slash count
		uint64_t slashCount = std::count(part1.begin(), part1.end(), '/');
		bool doubleSlash = (part1.find("//") != std::string::npos);
		bool hasTexcoords = (slashCount > 0 && !doubleSlash);
		bool hasNormals = (slashCount > 1);

		lineIndex1 = 0;

		while (StringUtils::readUntilSpace(line, lineIndex1, part1))
		{
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
				return;
			}

			vertexIndices.push_back(uint64_t(vertexIndex));

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
					return;
				}

				texcoordIndices.push_back(uint64_t(texcoordIndex));
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
					return;
				}

				normalIndices.push_back(uint64_t(normalIndex));
			}
		}

		if (vertexIndices.size() < 3)
		{
			log.logWarning("Too few vertices (%s) in a face", vertexIndices.size());
			return;
		}

		// triangulate
		for (uint64_t i = 2; i < vertexIndices.size(); ++i)
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
			processFace(line.substr(lineIndex), vertices, normals, texcoords, result, currentId, currentMaterialId);
	}

	fclose(file);

	log.logInfo("OBJ file reading finished (time: %s, triangles: %s, materials: %s, textures: %s)", timer.getElapsed().getString(true), result.triangles.size(), result.diffuseSpecularMaterials.size(), result.textures.size());

	return result;
}
