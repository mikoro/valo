// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "cereal/cereal.hpp"

#include "Math/Vector3.h"
#include "Math/EulerAngle.h"

namespace Raycer
{
	class Ray;
	class Vector2;
	class Scene;
	class ONB;

	enum class CameraProjectionType { PERSPECTIVE, ORTHOGRAPHIC, FISHEYE };

	class Camera
	{
	public:

		void initialize();
		void setImagePlaneSize(uint64_t width, uint64_t height);
		void update(float timeStep);
		void reset();
		bool isMoving() const;
		void saveState(const std::string& fileName) const;

		Ray getRay(const Vector2& pixel, bool& isOffLens) const;

		Vector3 getRight() const;
		Vector3 getUp() const;
		Vector3 getForward() const;

		Vector3 position;
		EulerAngle orientation;
		CameraProjectionType projectionType = CameraProjectionType::PERSPECTIVE;

		float fov = 75.0f;
		float orthoSize = 10.0f;
		float fishEyeAngle = 180.0f;
		float apertureSize = 0.1f;
		float focalDistance = 10.0f;

	private:

		float aspectRatio = 1.0f;
		float imagePlaneDistance = 1.0f;
		float imagePlaneWidth = 0.0f;
		float imagePlaneHeight = 0.0f;

		Vector3 right;
		Vector3 up;
		Vector3 forward;
		Vector3 imagePlaneCenter;

		Vector3 velocity;
		Vector3 smoothVelocity;
		Vector3 smoothAcceleration;
		Vector3 angularVelocity;
		Vector3 smoothAngularVelocity;
		Vector3 smoothAngularAcceleration;

		bool cameraIsMoving = false;

		Vector3 originalPosition;
		EulerAngle originalOrientation;

		float cameraMoveSpeedModifier = 1.0f;

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(position),
				CEREAL_NVP(orientation),
				CEREAL_NVP(projectionType),
				CEREAL_NVP(fov),
				CEREAL_NVP(orthoSize),
				CEREAL_NVP(fishEyeAngle),
				CEREAL_NVP(apertureSize),
				CEREAL_NVP(focalDistance));
		}
	};
}
