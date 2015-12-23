// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
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
	class Primitive;
	class ONB;

	enum class CameraProjectionType { PERSPECTIVE, ORTHOGRAPHIC, FISHEYE };

	class Camera
	{
	public:

		void initialize();
		void setImagePlaneSize(uint64_t width, uint64_t height);
		void update(double timeStep);
		void reset();
		bool isMoving() const;

		bool getRay(const Vector2& pixelCoordinate, Ray& ray) const;

		Vector3 position;
		EulerAngle orientation;
		CameraProjectionType projectionType = CameraProjectionType::PERSPECTIVE;

		double fov = 75.0;
		double orthoSize = 10.0;
		double fishEyeAngle = 180.0;
		double apertureSize = 0.1;
		double focalDistance = 10.0;

		Vector3 right;
		Vector3 up;
		Vector3 forward;
		Vector3 imagePlaneCenter;

	private:

		double aspectRatio = 1.0;
		double imagePlaneDistance = 1.0;
		double imagePlaneWidth = 0.0;
		double imagePlaneHeight = 0.0;

		Vector3 velocity;
		Vector3 smoothVelocity;
		Vector3 smoothAcceleration;
		Vector3 angularVelocity;
		Vector3 smoothAngularVelocity;
		Vector3 smoothAngularAcceleration;

		bool cameraIsMoving = false;

		Vector3 originalPosition;
		EulerAngle originalOrientation;

		double cameraMoveSpeedModifier = 1.0;

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
