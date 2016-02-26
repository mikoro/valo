// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
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

	bool getLine(const char* buffer, uint64_t bufferLength, uint64_t& lineStartIndex, uint64_t& lineEndIndex)
	{
		while (lineStartIndex < bufferLength)
		{
			char c = buffer[lineStartIndex];

			if (c != '\r' && c != '\n')
				break;

			lineStartIndex++;
		}

		if (lineStartIndex >= bufferLength)
			return false;

		lineEndIndex = lineStartIndex;

		while (lineEndIndex < bufferLength)
		{
			char c = buffer[lineEndIndex];

			if (c == '\r' || c == '\n')
				break;

			lineEndIndex++;
		}

		return true;
	}

	bool getWord(const char* buffer, uint64_t lineEndIndex, uint64_t& wordStartIndex, uint64_t& wordEndIndex)
	{
		while (wordStartIndex < lineEndIndex)
		{
			char c = buffer[wordStartIndex];

			if (c != ' ')
				break;

			wordStartIndex++;
		}

		if (wordStartIndex >= lineEndIndex)
			return false;

		wordEndIndex = wordStartIndex;

		while (wordEndIndex < lineEndIndex)
		{
			char c = buffer[wordEndIndex];

			if (c == ' ')
				break;

			wordEndIndex++;
		}

		return true;
	}

	int64_t getInt(const char* buffer, uint64_t& startIndex, uint64_t endIndex)
	{
		char c = 0;

		while (startIndex < endIndex)
		{
			c = buffer[startIndex];

			if (c != ' ' && c != '/')
				break;

			startIndex++;
		}

		if (startIndex >= endIndex)
			return 0;

		int64_t sign = 1;
		int64_t accumulator = 0;

		if (c == '-')
		{
			sign = -1;

			if (++startIndex >= endIndex)
				return 0;
		}

		c = buffer[startIndex];

		while (c >= '0' && c <= '9')
		{
			accumulator = accumulator * 10 + c - '0';

			if (++startIndex >= endIndex)
				return sign * accumulator;

			c = buffer[startIndex];
		}

		return sign * accumulator;
	}

	float getFloat(const char* buffer, uint64_t& startIndex, uint64_t endIndex)
	{
		char c = 0;

		while (startIndex < endIndex)
		{
			c = buffer[startIndex];

			if (c != ' ')
				break;

			startIndex++;
		}

		if (startIndex >= endIndex)
			return 0.0f;

		float sign = 1.0f;
		float accumulator = 0.0f;

		if (c == '-')
		{
			sign = -1.0f;

			if (++startIndex >= endIndex)
				return 0.0f;
		}

		c = buffer[startIndex];

		while (c >= '0' && c <= '9')
		{
			accumulator = accumulator * 10.0f + c - '0';

			if (++startIndex >= endIndex)
				return sign * accumulator;

			c = buffer[startIndex];
		}

		if (c == '.')
		{
			if (++startIndex >= endIndex)
				return sign * accumulator;

			float k = 0.1f;
			c = buffer[startIndex];

			while (c >= '0' && c <= '9')
			{
				accumulator += (c - '0') * k;
				k *= 0.1f;

				if (++startIndex >= endIndex)
					return sign * accumulator;

				c = buffer[startIndex];
			}
		}

		return sign * accumulator;
	}

	bool compareWord(const char* buffer, uint64_t wordStartIndex, uint64_t wordEndIndex, const char* otherWord)
	{
		uint64_t wordLength = wordEndIndex - wordStartIndex;
		uint64_t otherWordLength = strlen(otherWord);

		if (wordLength != otherWordLength)
			return false;

		for (uint64_t i = wordStartIndex; i < wordEndIndex; ++i)
		{
			if (buffer[i] != otherWord[i - wordStartIndex])
				return false;
		}

		return true;
	}

	void checkIndices(const char* buffer, uint64_t wordStartIndex, uint64_t wordEndIndex, bool& hasNormals, bool& hasTexcoords)
	{
		uint64_t slashCount = 0;
		uint64_t doubleSlashCount = 0;

		for (uint64_t i = wordStartIndex; i < wordEndIndex; ++i)
		{
			if (buffer[i] == '/')
			{
				slashCount++;

				if (i < wordEndIndex - 1)
				{
					if (buffer[i + 1] == '/')
						doubleSlashCount++;
				}
			}
		}

		hasNormals = (slashCount == 2);
		hasTexcoords = (slashCount > 0 && doubleSlashCount == 0);
	}

	void getIndices(const char* buffer, uint64_t wordStartIndex, uint64_t wordEndIndex, bool hasNormals, bool hasTexcoords, int64_t& vertexIndex, int64_t& normalIndex, int64_t& texcoordIndex)
	{
		vertexIndex = getInt(buffer, wordStartIndex, wordEndIndex);

		if (hasTexcoords)
			texcoordIndex = getInt(buffer, wordStartIndex, wordEndIndex);
		
		if (hasNormals)
			normalIndex = getInt(buffer, wordStartIndex, wordEndIndex);
	}

	bool processFace(const char* buffer, uint64_t lineStartIndex, uint64_t lineEndIndex, uint64_t lineNumber, std::vector<Vector3>& vertices, std::vector<Vector3>& normals, std::vector<Vector2>& texcoords, ModelLoaderResult& result, uint64_t& currentId, uint64_t currentMaterialId)
	{
		Log& log = App::getLog();

		uint64_t vertexIndices[4];
		uint64_t normalIndices[4];
		uint64_t texcoordIndices[4];
		uint64_t vertexCount = 0;

		bool hasNormals = false;
		bool hasTexcoords = false;

		uint64_t wordStartIndex = lineStartIndex;
		uint64_t wordEndIndex = 0;
		
		for (uint64_t i = 0; i < 4; ++i)
		{
			if (!getWord(buffer, lineEndIndex, wordStartIndex, wordEndIndex))
				break;

			if (i == 0)
				checkIndices(buffer, wordStartIndex, wordEndIndex, hasNormals, hasTexcoords);

			vertexCount++;

			int64_t vertexIndex;
			int64_t texcoordIndex;
			int64_t normalIndex;

			getIndices(buffer, wordStartIndex, wordEndIndex, hasNormals, hasTexcoords, vertexIndex, normalIndex, texcoordIndex);

			if (vertexIndex < 0)
				vertexIndex = int64_t(vertices.size()) + vertexIndex;
			else
				vertexIndex--;

			if (vertexIndex < 0 || vertexIndex >= int64_t(vertices.size()))
			{
				log.logWarning("Vertex index (%s) was out of bounds (line: %s)", vertexIndex, lineNumber);
				return false;
			}

			vertexIndices[i] = uint64_t(vertexIndex);

			if (hasTexcoords)
			{
				if (texcoordIndex < 0)
					texcoordIndex = int64_t(texcoords.size()) + texcoordIndex;
				else
					texcoordIndex--;

				if (texcoordIndex < 0 || texcoordIndex >= int64_t(texcoords.size()))
				{
					log.logWarning("Texcoord index (%s) was out of bounds (line: %s)", texcoordIndex, lineNumber);
					return false;
				}

				texcoordIndices[i] = uint64_t(texcoordIndex);
			}

			if (hasNormals)
			{
				if (normalIndex < 0)
					normalIndex = int64_t(normals.size()) + normalIndex;
				else
					normalIndex--;

				if (normalIndex < 0 || normalIndex >= int64_t(normals.size()))
				{
					log.logWarning("Normal index (%s) was out of bounds (line: %s)", normalIndex, lineNumber);
					return false;
				}

				normalIndices[i] = uint64_t(normalIndex);
			}

			wordStartIndex = wordEndIndex;
		}

		if (vertexCount < 3)
		{
			log.logWarning("Too few vertices (%s) in a face (line: %s)", vertexCount, lineNumber);
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

	std::ifstream file(info.modelFilePath, std::ios::in | std::ios::binary | std::ios::ate);

	if (!file.good())
		throw std::runtime_error(tfm::format("Could not open the OBJ file: %s", info.modelFilePath));

	auto size = file.tellg();
	file.seekg(0, std::ios::beg);

	std::vector<char> fileBuffer(size);
	file.read(&fileBuffer[0], size);

	file.close();

	char* fileBufferPtr = &fileBuffer[0];
	uint64_t fileBufferLength = fileBuffer.size();
	uint64_t lineStartIndex = 0;
	uint64_t lineEndIndex = 0;
	uint64_t lineNumber = 0;

	while (getLine(fileBufferPtr, fileBufferLength, lineStartIndex, lineEndIndex))
	{
		lineNumber++;

		uint64_t wordStartIndex = lineStartIndex;
		uint64_t wordEndIndex = 0;

		getWord(fileBufferPtr, lineEndIndex, wordStartIndex, wordEndIndex);

		if (compareWord(fileBufferPtr, wordStartIndex, wordEndIndex, "f")) // face
		{
			if (!processFace(fileBufferPtr, lineStartIndex + 2, lineEndIndex, lineNumber, vertices, normals, texcoords, result, currentId, currentMaterialId))
				break;
		}
		else if (compareWord(fileBufferPtr, wordStartIndex, wordEndIndex, "v")) // vertex
		{
			Vector3 vertex;
			wordStartIndex += 2;

			vertex.x = getFloat(fileBufferPtr, wordStartIndex, lineEndIndex);
			vertex.y = getFloat(fileBufferPtr, wordStartIndex, lineEndIndex);
			vertex.z = getFloat(fileBufferPtr, wordStartIndex, lineEndIndex);

			vertices.push_back(transformation.transformPosition(vertex));
		}
		else if (compareWord(fileBufferPtr, wordStartIndex, wordEndIndex, "vn")) // normal
		{
			Vector3 normal;
			wordStartIndex += 3;

			normal.x = getFloat(fileBufferPtr, wordStartIndex, lineEndIndex);
			normal.y = getFloat(fileBufferPtr, wordStartIndex, lineEndIndex);
			normal.z = getFloat(fileBufferPtr, wordStartIndex, lineEndIndex);

			normals.push_back(transformationInvT.transformDirection(normal).normalized());
		}
		else if (compareWord(fileBufferPtr, wordStartIndex, wordEndIndex, "vt")) // texcoord
		{
			Vector2 texcoord;
			wordStartIndex += 3;

			texcoord.x = getFloat(fileBufferPtr, wordStartIndex, lineEndIndex);
			texcoord.y = getFloat(fileBufferPtr, wordStartIndex, lineEndIndex);

			texcoords.push_back(texcoord);
		}
		else if (compareWord(fileBufferPtr, wordStartIndex, wordEndIndex, "mtllib")) // new material file
		{
			wordStartIndex = wordEndIndex;
			getWord(fileBufferPtr, lineEndIndex, wordStartIndex, wordEndIndex);
			std::string mtlFilePath(fileBufferPtr + wordStartIndex, wordEndIndex - wordStartIndex);

			processMaterialFile(objFileDirectory, mtlFilePath, result, materialsMap, externalMaterialsMap, currentId);
		}
		else if (compareWord(fileBufferPtr, wordStartIndex, wordEndIndex, "usemtl")) // select material
		{
			wordStartIndex = wordEndIndex;
			getWord(fileBufferPtr, lineEndIndex, wordStartIndex, wordEndIndex);
			std::string materialName(fileBufferPtr + wordStartIndex, wordEndIndex - wordStartIndex);

			if (externalMaterialsMap.count(materialName))
				currentMaterialId = externalMaterialsMap[materialName];
			else if (materialsMap.count(materialName))
				currentMaterialId = materialsMap[materialName];
			else
			{
				log.logWarning("Could not find material named \"%s\"", materialName);
				currentMaterialId = info.defaultMaterialId;
			}
		}

		lineStartIndex = lineEndIndex;
	}

	log.logInfo("OBJ file reading finished (time: %s, vertices: %s, normals: %s, texcoords: %s, triangles: %s, materials: %s, textures: %s)", timer.getElapsed().getString(true), vertices.size(), normals.size(), texcoords.size(), result.triangles.size(), result.diffuseSpecularMaterials.size(), result.textures.size());

	return result;
}
