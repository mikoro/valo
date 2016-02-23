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

	// read one line from filebuffer to linebuffer (separators: \r \n)
	bool getLine(const char* fileBuffer, uint64_t& fileBufferIndex, uint64_t fileBufferLength, char* lineBuffer, uint64_t& lineBufferLength)
	{
		if (fileBufferIndex >= fileBufferLength)
			return false;

		while (fileBufferIndex < fileBufferLength)
		{
			char c = fileBuffer[fileBufferIndex];

			if (c != '\r' && c != '\n')
				break;

			fileBufferIndex++;
		}

		uint64_t startFileBufferIndex = fileBufferIndex;

		while (fileBufferIndex < fileBufferLength)
		{
			char c = fileBuffer[fileBufferIndex];

			if (c == '\r' || c == '\n')
				break;

			fileBufferIndex++;
		}

		lineBufferLength = fileBufferIndex - startFileBufferIndex;

		for (uint64_t i = 0; i < lineBufferLength && i < 128; ++i)
			lineBuffer[i] = fileBuffer[startFileBufferIndex + i];

		return true;
	}

	// read one word from linebuffer to wordbuffer (separator: space)
	bool getWord(const char* lineBuffer, uint64_t& lineBufferIndex, uint64_t lineBufferLength, char* wordBuffer, uint64_t& wordBufferLength)
	{
		if (lineBufferIndex >= lineBufferLength)
			return false;

		while (lineBufferIndex < lineBufferLength)
		{
			char c = lineBuffer[lineBufferIndex];

			if (c != ' ')
				break;

			lineBufferIndex++;
		}

		uint64_t startLineBufferIndex = lineBufferIndex;

		while (lineBufferIndex < lineBufferLength)
		{
			char c = lineBuffer[lineBufferIndex];

			if (c == ' ')
				break;

			lineBufferIndex++;
		}

		wordBufferLength = lineBufferIndex - startLineBufferIndex;

		if (wordBufferLength == 0)
			return false;

		for (uint64_t i = 0; i < wordBufferLength && i < 128; ++i)
			wordBuffer[i] = lineBuffer[startLineBufferIndex + i];

		return true;
	}

	// simple word comparison character by character
	bool compareWord(const char* wordBuffer, uint64_t wordBufferLength, const char* otherWord)
	{
		uint64_t otherWordLength = strlen(otherWord);

		if (wordBufferLength != otherWordLength)
			return false;

		for (uint64_t i = 0; i < wordBufferLength; ++i)
		{
			if (wordBuffer[i] != otherWord[i])
				return false;
		}

		return true;
	}

	// simplified ascii -> float conversion (no scientific notation)
	float getFloat(const char* buffer, uint64_t& bufferIndex, uint64_t bufferLength)
	{
		if (bufferIndex >= bufferLength)
			return 0.0f;

		char c = 0;

		while (bufferIndex < bufferLength)
		{
			c = buffer[bufferIndex];

			if (c != ' ')
				break;

			bufferIndex++;
		}

		if (bufferIndex >= bufferLength)
			return 0.0f;

		float sign = 1.0f;
		float accumulator = 0.0f;

		if (c == '-')
		{
			sign = -1.0f;

			if (++bufferIndex >= bufferLength)
				return 0.0f;
		}

		c = buffer[bufferIndex];

		while (c >= '0' && c <= '9')
		{
			accumulator = accumulator * 10.0f + c - '0';

			if (++bufferIndex >= bufferLength)
				return sign * accumulator;

			c = buffer[bufferIndex];
		}

		if (c == '.')
		{
			if (++bufferIndex >= bufferLength)
				return sign * accumulator;

			float k = 0.1f;
			c = buffer[bufferIndex];

			while (c >= '0' && c <= '9')
			{
				accumulator += (c - '0') * k;
				k *= 0.1f;

				if (++bufferIndex >= bufferLength)
					return sign * accumulator;

				c = buffer[bufferIndex];
			}
		}

		return sign * accumulator;
	}

	// simplified ascii -> int conversion
	int64_t getInt(const char* buffer, uint64_t& bufferIndex, uint64_t bufferLength)
	{
		if (bufferIndex >= bufferLength)
			return 0;

		char c = 0;

		while (bufferIndex < bufferLength)
		{
			c = buffer[bufferIndex];

			if (c != ' ' && c != '/')
				break;

			bufferIndex++;
		}

		if (bufferIndex >= bufferLength)
			return 0;

		int64_t sign = 1;
		int64_t accumulator = 0;

		if (c == '-')
		{
			sign = -1;

			if (++bufferIndex >= bufferLength)
				return 0;
		}

		c = buffer[bufferIndex];

		while (c >= '0' && c <= '9')
		{
			accumulator = accumulator * 10 + c - '0';

			if (++bufferIndex >= bufferLength)
				return sign * accumulator;

			c = buffer[bufferIndex];
		}

		return sign * accumulator;
	}

	// what indices are included in a face
	void checkIndices(const char* wordBuffer, uint64_t wordBufferLength, bool& hasNormals, bool& hasTexcoords)
	{
		uint64_t slashCount = 0;
		uint64_t doubleSlashCount = 0;

		for (uint64_t i = 0; i < wordBufferLength; ++i)
		{
			if (wordBuffer[i] == '/')
			{
				slashCount++;

				if (i < wordBufferLength - 1)
				{
					if (wordBuffer[i + 1] == '/')
						doubleSlashCount++;
				}
			}
		}

		hasNormals = (slashCount == 2);
		hasTexcoords = (slashCount > 0 && doubleSlashCount == 0);
	}

	// extract indices from a face
	void getIndices(char* wordBuffer, uint64_t& wordBufferIndex, uint64_t wordBufferLength, bool hasNormals, bool hasTexcoords, int64_t& vertexIndex, int64_t& normalIndex, int64_t& texcoordIndex)
	{
		vertexIndex = getInt(wordBuffer, wordBufferIndex, wordBufferLength);

		if (hasTexcoords)
			texcoordIndex = getInt(wordBuffer, wordBufferIndex, wordBufferLength);
		
		if (hasNormals)
			normalIndex = getInt(wordBuffer, wordBufferIndex, wordBufferLength);
	}

	bool processFace(const char* lineBuffer, uint64_t& lineBufferIndex, uint64_t lineBufferLength, uint64_t lineNumber, std::vector<Vector3>& vertices, std::vector<Vector3>& normals, std::vector<Vector2>& texcoords, ModelLoaderResult& result, uint64_t& currentId, uint64_t currentMaterialId)
	{
		Log& log = App::getLog();

		uint64_t vertexIndices[4];
		uint64_t normalIndices[4];
		uint64_t texcoordIndices[4];

		char wordBuffer[128];
		uint64_t wordBufferLength = 0;
		uint64_t wordBufferIndex = 0;
		uint64_t vertexCount = 0;

		bool hasNormals = false;
		bool hasTexcoords = false;

		for (uint64_t i = 0; i < 4; ++i)
		{
			if (!getWord(lineBuffer, lineBufferIndex, lineBufferLength, wordBuffer, wordBufferLength))
				break;

			vertexCount++;

			if (i == 0)
				checkIndices(wordBuffer, wordBufferLength, hasNormals, hasTexcoords);

			int64_t vertexIndex = 0;
			int64_t texcoordIndex = 0;
			int64_t normalIndex = 0;

			wordBufferIndex = 0;
			getIndices(wordBuffer, wordBufferIndex, wordBufferLength, hasNormals, hasTexcoords, vertexIndex, normalIndex, texcoordIndex);

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

	uint64_t fileBufferIndex = 0;
	uint64_t lineBufferIndex = 0;
	uint64_t fileBufferLength = fileBuffer.size();
	uint64_t lineBufferLength = 0;
	uint64_t wordBufferLength = 0;
	uint64_t lineNumber = 0;
	char lineBuffer[128];
	char wordBuffer[128];

	while (getLine(&fileBuffer[0], fileBufferIndex, fileBufferLength, lineBuffer, lineBufferLength))
	{
		lineNumber++;

		if (lineBufferLength == 0)
			continue;

		lineBufferIndex = 0;
		getWord(lineBuffer, lineBufferIndex, lineBufferLength, wordBuffer, wordBufferLength);

		if (compareWord(wordBuffer, wordBufferLength, "f")) // face
		{
			if (!processFace(lineBuffer, lineBufferIndex, lineBufferLength, lineNumber, vertices, normals, texcoords, result, currentId, currentMaterialId))
				break;
		}
		else if (compareWord(wordBuffer, wordBufferLength, "v")) // vertex
		{
			Vector3 vertex;

			vertex.x = getFloat(lineBuffer, lineBufferIndex, lineBufferLength);
			vertex.y = getFloat(lineBuffer, lineBufferIndex, lineBufferLength);
			vertex.z = getFloat(lineBuffer, lineBufferIndex, lineBufferLength);

			vertices.push_back(transformation.transformPosition(vertex));
		}
		else if (compareWord(wordBuffer, wordBufferLength, "vn")) // normal
		{
			Vector3 normal;

			normal.x = getFloat(lineBuffer, lineBufferIndex, lineBufferLength);
			normal.y = getFloat(lineBuffer, lineBufferIndex, lineBufferLength);
			normal.z = getFloat(lineBuffer, lineBufferIndex, lineBufferLength);

			normals.push_back(transformationInvT.transformDirection(normal).normalized());
		}
		else if (compareWord(wordBuffer, wordBufferLength, "vt")) // texcoord
		{
			Vector2 texcoord;

			texcoord.x = getFloat(lineBuffer, lineBufferIndex, lineBufferLength);
			texcoord.y = getFloat(lineBuffer, lineBufferIndex, lineBufferLength);

			texcoords.push_back(texcoord);
		}
		else if (compareWord(wordBuffer, wordBufferLength, "mtllib")) // new material file
		{
			getWord(lineBuffer, lineBufferIndex, lineBufferLength, wordBuffer, wordBufferLength);
			std::string mtlFilePath(wordBuffer, wordBufferLength);
			processMaterialFile(objFileDirectory, mtlFilePath, result, materialsMap, externalMaterialsMap, currentId);
		}
		else if (compareWord(wordBuffer, wordBufferLength, "usemtl")) // select material
		{
			getWord(lineBuffer, lineBufferIndex, lineBufferLength, wordBuffer, wordBufferLength);
			std::string materialName(wordBuffer, wordBufferLength);

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
	}

	log.logInfo("OBJ file reading finished (time: %s, vertices: %s, normals: %s, texcoords: %s, triangles: %s, materials: %s, textures: %s)", timer.getElapsed().getString(true), vertices.size(), normals.size(), texcoords.size(), result.triangles.size(), result.diffuseSpecularMaterials.size(), result.textures.size());

	return result;
}
