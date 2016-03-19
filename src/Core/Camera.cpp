// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

#include "Core/App.h"
#include "Core/Camera.h"
#include "Core/Intersection.h"
#include "Core/Ray.h"
#include "Math/MathUtils.h"
#include "Math/Vector2.h"
#include "Runners/WindowRunner.h"
#include "Utils/Log.h"

using namespace Raycer;

void Camera::initialize()
{
	originalPosition = position;
	originalOrientation = orientation;
}

void Camera::setImagePlaneSize(uint64_t width, uint64_t height)
{
	imagePlaneWidth = float(width - 1);
	imagePlaneHeight = float(height - 1);
	aspectRatio = float(height) / float(width);
}

void Camera::reset()
{
	position = originalPosition;
	orientation = originalOrientation;
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

	// LENS STUFF //

	if (windowRunner.keyWasPressed(GLFW_KEY_HOME))
	{
		if (type == CameraType::PERSPECTIVE)
			type = CameraType::ORTHOGRAPHIC;
		else if (type == CameraType::ORTHOGRAPHIC)
			type = CameraType::FISHEYE;
		else if (type == CameraType::FISHEYE)
			type = CameraType::PERSPECTIVE;
	}

	if (!windowRunner.keyIsDown(GLFW_KEY_LEFT_CONTROL) && !windowRunner.keyIsDown(GLFW_KEY_RIGHT_CONTROL) &&
		!windowRunner.keyIsDown(GLFW_KEY_LEFT_SHIFT) && !windowRunner.keyIsDown(GLFW_KEY_RIGHT_SHIFT))
	{
		if (windowRunner.keyIsDown(GLFW_KEY_PAGE_DOWN))
		{
			if (type == CameraType::PERSPECTIVE)
				fov -= 50.0f * timeStep;
			else if (type == CameraType::ORTHOGRAPHIC)
				orthoSize -= 10.0f * timeStep;
			else if (type == CameraType::FISHEYE)
				fishEyeAngle -= 50.0f * timeStep;
		}

		if (windowRunner.keyIsDown(GLFW_KEY_PAGE_UP))
		{
			if (type == CameraType::PERSPECTIVE)
				fov += 50.0f * timeStep;
			else if (type == CameraType::ORTHOGRAPHIC)
				orthoSize += 10.0f * timeStep;
			else if (type == CameraType::FISHEYE)
				fishEyeAngle += 50.0f * timeStep;
		}
	}

	fov = std::max(1.0f, std::min(fov, 180.0f));
	orthoSize = std::max(0.0f, orthoSize);
	fishEyeAngle = std::max(1.0f, std::min(fishEyeAngle, 360.0f));
	imagePlaneDistance = 0.5f / std::tan(MathUtils::degToRad(fov / 2.0f));

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

Ray Camera::getRay(const Vector2& pixel, bool& isOffLens) const
{
	Ray ray;
	isOffLens = false;

	switch (type)
	{
		case CameraType::PERSPECTIVE:
		{
			float dx = (pixel.x / imagePlaneWidth) - 0.5f;
			float dy = (pixel.y / imagePlaneHeight) - 0.5f;

			Vector3 imagePlanePixelPosition = imagePlaneCenter + (dx * right) + (dy * aspectRatio * up);

			ray.origin = position;
			ray.direction = (imagePlanePixelPosition - position).normalized();

		} break;

		case CameraType::ORTHOGRAPHIC:
		{
			float dx = (pixel.x / imagePlaneWidth) - 0.5f;
			float dy = (pixel.y / imagePlaneHeight) - 0.5f;

			ray.origin = position + (dx * orthoSize * right) + (dy * orthoSize * aspectRatio * up);
			ray.direction = forward;

		} break;

		// http://paulbourke.net/dome/fisheye/
		case CameraType::FISHEYE:
		{
			float dx = (pixel.x / imagePlaneWidth) * 2.0f - 1.0f;
			float dy = (pixel.y / imagePlaneHeight) * 2.0f - 1.0f;

			dx /= std::min(1.0f, aspectRatio);
			dy *= std::max(1.0f, aspectRatio);

			float r = sqrt(dx * dx + dy * dy);

			if (r > 1.0f)
				isOffLens = true;

			float phi = std::atan2(dy, dx);
			float theta = r * (MathUtils::degToRad(fishEyeAngle) / 2.0f);

			float u = std::sin(theta) * std::cos(phi);
			float v = std::sin(theta) * std::sin(phi);
			float w = std::cos(theta);

			ray.origin = position;
			ray.direction = u * right + v * up + w * forward;

		} break;

		default: break;
	}

	ray.precalculate();
	return ray;
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
