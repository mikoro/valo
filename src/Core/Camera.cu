// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include <GLFW/glfw3.h>

#include "tinyformat/tinyformat.h"

#include "App.h"
#include "Core/Camera.h"
#include "Core/Intersection.h"
#include "Core/Ray.h"
#include "Math/MathUtils.h"
#include "Math/Vector2.h"
#include "Math/Mapper.h"
#include "Runners/WindowRunner.h"
#include "Utils/Log.h"

using namespace Raycer;

void Camera::initialize()
{
	originalPosition = position;
	originalOrientation = orientation;
	originalFov = fov;
	originalOrthoSize = orthoSize;
	originalFishEyeAngle = fishEyeAngle;
}

void Camera::setImagePlaneSize(uint32_t width, uint32_t height)
{
	imagePlaneWidth = float(width - 1);
	imagePlaneHeight = float(height - 1);
	aspectRatio = float(height) / float(width);
	imageCenter = Vector2(float(width / 2), float(height / 2));
	maxVignetteDistance = imageCenter.lengthSquared();
}

void Camera::reset()
{
	position = originalPosition;
	orientation = originalOrientation;
	fov = originalFov;
	orthoSize = originalOrthoSize;
	fishEyeAngle = originalFishEyeAngle;
	currentSpeedModifier = 1.0f;
	velocity = Vector3(0.0f, 0.0f, 0.0f);
	smoothVelocity = Vector3(0.0f, 0.0f, 0.0f);
	smoothAcceleration = Vector3(0.0f, 0.0f, 0.0f);
	angularVelocity = Vector3(0.0f, 0.0f, 0.0f);
	smoothAngularVelocity = Vector3(0.0f, 0.0f, 0.0f);
	smoothAngularAcceleration = Vector3(0.0f, 0.0f, 0.0f);
}

void Camera::update(float timeStep)
{
	WindowRunner& windowRunner = App::getWindowRunner();
	MouseInfo mouseInfo = windowRunner.getMouseInfo();

	// SPEED MODIFIERS //

	if (windowRunner.keyWasPressed(GLFW_KEY_INSERT))
		currentSpeedModifier *= 2.0f;

	if (windowRunner.keyWasPressed(GLFW_KEY_DELETE))
		currentSpeedModifier *= 0.5f;

	float actualMoveSpeed = moveSpeed * currentSpeedModifier;

	if (windowRunner.keyIsDown(GLFW_KEY_LEFT_CONTROL) || windowRunner.keyIsDown(GLFW_KEY_RIGHT_CONTROL))
		actualMoveSpeed *= slowSpeedModifier;

	if (windowRunner.keyIsDown(GLFW_KEY_LEFT_SHIFT) || windowRunner.keyIsDown(GLFW_KEY_RIGHT_SHIFT))
		actualMoveSpeed *= fastSpeedModifier;

	if (windowRunner.keyIsDown(GLFW_KEY_LEFT_ALT) || windowRunner.keyIsDown(GLFW_KEY_RIGHT_ALT))
		actualMoveSpeed *= veryFastSpeedModifier;

	// ACCELERATIONS AND VELOCITIES //

	velocity = Vector3(0.0f, 0.0f, 0.0f);
	angularVelocity = Vector3(0.0f, 0.0f, 0.0f);
	bool movementKeyIsPressed = false;

	if (windowRunner.mouseIsDown(GLFW_MOUSE_BUTTON_LEFT) || freeLook)
	{
		smoothAngularAcceleration.y -= mouseInfo.deltaX * mouseSpeed;
		smoothAngularAcceleration.x += mouseInfo.deltaY * mouseSpeed;
		angularVelocity.y = -mouseInfo.deltaX * mouseSpeed;
		angularVelocity.x = mouseInfo.deltaY * mouseSpeed;
	}

	if (windowRunner.keyIsDown(GLFW_KEY_W))
	{
		smoothAcceleration += forward * actualMoveSpeed;
		velocity = forward * actualMoveSpeed;
		movementKeyIsPressed = true;
	}

	if (windowRunner.keyIsDown(GLFW_KEY_S))
	{
		smoothAcceleration -= forward * actualMoveSpeed;
		velocity = -forward * actualMoveSpeed;
		movementKeyIsPressed = true;
	}

	if (windowRunner.keyIsDown(GLFW_KEY_D))
	{
		smoothAcceleration += right * actualMoveSpeed;
		velocity = right * actualMoveSpeed;
		movementKeyIsPressed = true;
	}

	if (windowRunner.keyIsDown(GLFW_KEY_A))
	{
		smoothAcceleration -= right * actualMoveSpeed;
		velocity = -right * actualMoveSpeed;
		movementKeyIsPressed = true;
	}

	if (windowRunner.keyIsDown(GLFW_KEY_E))
	{
		smoothAcceleration += up * actualMoveSpeed;
		velocity = up * actualMoveSpeed;
		movementKeyIsPressed = true;
	}

	if (windowRunner.keyIsDown(GLFW_KEY_Q))
	{
		smoothAcceleration -= up * actualMoveSpeed;
		velocity = -up * actualMoveSpeed;
		movementKeyIsPressed = true;
	}

	if (windowRunner.keyIsDown(GLFW_KEY_SPACE) || !enableMovement)
	{
		velocity = Vector3(0.0f, 0.0f, 0.0f);
		smoothVelocity = Vector3(0.0f, 0.0f, 0.0f);
		smoothAcceleration = Vector3(0.0f, 0.0f, 0.0f);
		angularVelocity = Vector3(0.0f, 0.0f, 0.0f);
		smoothAngularVelocity = Vector3(0.0f, 0.0f, 0.0f);
		smoothAngularAcceleration = Vector3(0.0f, 0.0f, 0.0f);
	}

	// EULER INTEGRATION //

	cameraIsMoving = false;

	if (smoothMovement)
	{
		smoothVelocity += smoothAcceleration * timeStep;
		position += smoothVelocity * timeStep;

		smoothAngularVelocity += smoothAngularAcceleration * timeStep;
		orientation.yaw += smoothAngularVelocity.y * timeStep;
		orientation.pitch += smoothAngularVelocity.x * timeStep;

		cameraIsMoving = !smoothVelocity.isZero() || !smoothAngularVelocity.isZero();
	}
	else
	{
		position += velocity * timeStep;
		orientation.yaw += angularVelocity.y * timeStep;
		orientation.pitch += angularVelocity.x * timeStep;

		cameraIsMoving = !velocity.isZero() || !angularVelocity.isZero();
	}

	// DRAG & AUTO STOP //

	float smoothVelocityLength = smoothVelocity.length();
	float smoothAngularVelocityLength = smoothAngularVelocity.length();

	if ((smoothVelocityLength < autoStopSpeed * actualMoveSpeed) && !movementKeyIsPressed)
		smoothVelocity = smoothAcceleration = Vector3(0.0f, 0.0f, 0.0f);
	else if (!smoothVelocity.isZero())
		smoothAcceleration = moveDrag * (-smoothVelocity.normalized() * smoothVelocityLength);
	
	if (smoothAngularVelocityLength < autoStopSpeed * actualMoveSpeed)
		smoothAngularVelocity = smoothAngularAcceleration = Vector3(0.0f, 0.0f, 0.0f);
	else if (!smoothAngularVelocity.isZero())
		smoothAngularAcceleration = mouseDrag * (-smoothAngularVelocity.normalized() * smoothAngularVelocityLength);

	// ORIENTATION & BASIS VECTORS //

	orientation.clampPitch();
	orientation.normalize();

	ONB onb = ONB::fromNormal(orientation.getDirection());
	right = onb.u;
	up = onb.v;
	forward = onb.w;

	// MISC

	fov = MAX(1.0f, MIN(fov, 180.0f));
	orthoSize = MAX(0.0f, orthoSize);
	fishEyeAngle = MAX(1.0f, MIN(fishEyeAngle, 360.0f));

	float imagePlaneDistance = 0.5f / std::tan(MathUtils::degToRad(fov / 2.0f));
	imagePlaneCenter = position + (forward * imagePlaneDistance);
}

bool Camera::isMoving() const
{
	return cameraIsMoving;
}

void Camera::saveState(const std::string& fileName) const
{
	App::getLog().logInfo("Saving camera state to %s", fileName);

	std::ofstream file(fileName);

	if (!file.is_open())
		throw std::runtime_error("Could not open file for writing camera state");

	file << tfm::format("x = %f", position.x) << std::endl;
	file << tfm::format("y = %f", position.y) << std::endl;
	file << tfm::format("z = %f", position.z) << std::endl;
	file << std::endl;
	file << tfm::format("pitch = %f", orientation.pitch) << std::endl;
	file << tfm::format("yaw = %f", orientation.yaw) << std::endl;
	file << tfm::format("roll = %f", orientation.roll) << std::endl;
	file << std::endl;
	file << tfm::format("fov = %f", fov) << std::endl;
	file << tfm::format("orthoSize = %f", orthoSize) << std::endl;
	file << tfm::format("fishEyeAngle = %f", fishEyeAngle) << std::endl;
	file << std::endl;
	file << tfm::format("scene.camera.position = Vector3(%.4ff, %.4ff, %.4ff);", position.x, position.y, position.z) << std::endl;
	file << tfm::format("scene.camera.orientation = EulerAngle(%.4ff, %.4ff, %.4ff);", orientation.pitch, orientation.yaw, orientation.roll) << std::endl;
	file << tfm::format("scene.camera.fov = %.2ff;", fov) << std::endl;
	file << tfm::format("scene.camera.orthoSize = %.2ff;", orthoSize) << std::endl;
	file << tfm::format("scene.camera.fishEyeAngle = %.2ff;", fishEyeAngle) << std::endl;

	file.close();
}

CUDA_CALLABLE CameraRay Camera::getRay(const Vector2& pixel, Random& random) const
{
	Vector3 origin;
	Vector3 direction;
	bool offLens = false;

	switch (type)
	{
		case CameraType::PERSPECTIVE:
		{
			float dx = (pixel.x / imagePlaneWidth) - 0.5f;
			float dy = (pixel.y / imagePlaneHeight) - 0.5f;

			Vector3 imagePlanePixelPosition = imagePlaneCenter + (dx * right) + (dy * aspectRatio * up);

			origin = position;
			direction = (imagePlanePixelPosition - position).normalized();

		} break;

		case CameraType::ORTHOGRAPHIC:
		{
			float dx = (pixel.x / imagePlaneWidth) - 0.5f;
			float dy = (pixel.y / imagePlaneHeight) - 0.5f;

			origin = position + (dx * orthoSize * right) + (dy * orthoSize * aspectRatio * up);
			direction = forward;

		} break;

		// http://paulbourke.net/dome/fisheye/
		case CameraType::FISHEYE:
		{
			float dx = (pixel.x / imagePlaneWidth) * 2.0f - 1.0f;
			float dy = (pixel.y / imagePlaneHeight) * 2.0f - 1.0f;

			dx /= MIN(1.0f, aspectRatio);
			dy *= MAX(1.0f, aspectRatio);

			float r = sqrt(dx * dx + dy * dy);

			if (r > 1.0f)
				offLens = true;

			float phi = std::atan2(dy, dx);
			float theta = r * (MathUtils::degToRad(fishEyeAngle) / 2.0f);

			float u = std::sin(theta) * std::cos(phi);
			float v = std::sin(theta) * std::sin(phi);
			float w = std::cos(theta);

			origin = position;
			direction = u * right + v * up + w * forward;

		} break;

		default: break;
	}

	if (depthOfField)
	{
		Vector3 focalPoint = origin + direction * focalDistance;
		Vector2 originOffset = Mapper::mapToDisc(random.getVector2());

		origin = origin + ((originOffset.x * apertureSize) * right + (originOffset.y * apertureSize) * up);
		direction = (focalPoint - origin).normalized();
	}

	CameraRay cameraRay;
	cameraRay.ray.origin = origin;
	cameraRay.ray.direction = direction;
	cameraRay.offLens = offLens;
	cameraRay.ray.precalculate();

	if (vignette)
	{
		float vignetteDistance = (imageCenter - pixel).lengthSquared();
		float vignetteAmount = std::max(0.0f, std::min(vignetteDistance / maxVignetteDistance + vignetteOffset, 1.0f));
		cameraRay.brightness = 1.0f - std::pow(vignetteAmount, vignettePower);
	}

	return cameraRay;
}

Vector3 Camera::getRight() const
{
	return right;
}

Vector3 Camera::getUp() const
{
	return up;
}

Vector3 Camera::getForward() const
{
	return forward;
}

std::string Camera::getName() const
{
	switch (type)
	{
		case CameraType::PERSPECTIVE: return "perspective";
		case CameraType::ORTHOGRAPHIC: return "orthographic";
		case CameraType::FISHEYE: return "fisheye";
		default: return "unknown";
	}
}
