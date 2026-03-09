/*
 * Copyright 2025 Hillbot Inc.
 * Copyright 2020-2024 UCSD SU Lab
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "sapien/physx/physx_system.h"
#include "../logger.h"
#include "./filter_shader.hpp"
#include "sapien/math/conversion.h"
#include "sapien/physx/articulation.h"
#include "sapien/physx/articulation_link_component.h"
#include "sapien/physx/material.h"
#include "sapien/physx/physx_default.h"
#include "sapien/physx/rigid_component.h"
#include "sapien/profiler.h"
#include <extensions/PxExtensionsAPI.h>

#ifdef SAPIEN_CUDA
#include "./physx_system.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <PxDirectGPUAPI.h>
#endif

using namespace physx;
namespace sapien {
namespace physx {

struct SapienBodyDataTest {
  Pose pose;
  Vec3 v;
  Vec3 w;
};

static_assert(sizeof(SapienBodyDataTest) == 52);

PhysxSystem::PhysxSystem()
    : mSceneConfig(PhysxDefault::getSceneConfig()), mEngine(PhysxEngine::Get()) {}

PhysxSystemCpu::PhysxSystemCpu() {
  if (PhysxDefault::GetGPUEnabled()) {
    logger::warn(
        "A PhysX CPU system is being created while PhysX GPU is enabled. You can safely ignore "
        "this message if it is intended. To use GPU PhysX, create a sapien.physx.PhysxGpuSystem "
        "explicitly and pass it to sapien.Scene constructor.");
  }

  auto &config = mSceneConfig;
  PxSceneDesc sceneDesc(mEngine->getPxPhysics()->getTolerancesScale());
  sceneDesc.gravity = Vec3ToPxVec3(config.gravity);
  sceneDesc.filterShader = TypeAffinityIgnoreFilterShader;
  sceneDesc.solverType = config.enableTGS ? PxSolverType::eTGS : PxSolverType::ePGS;
  sceneDesc.bounceThresholdVelocity = config.bounceThreshold;

  PxSceneFlags sceneFlags;
  if (config.enableEnhancedDeterminism) {
    sceneFlags |= PxSceneFlag::eENABLE_ENHANCED_DETERMINISM;
  }
  if (config.enablePCM) {
    sceneFlags |= PxSceneFlag::eENABLE_PCM;
  }
  if (config.enableCCD) {
    sceneFlags |= PxSceneFlag::eENABLE_CCD;
  }
  if (config.enableFrictionEveryIteration) {
    sceneFlags |= PxSceneFlag::eENABLE_FRICTION_EVERY_ITERATION;
  }

  sceneDesc.flags = sceneFlags;

  mPxCPUDispatcher = PxDefaultCpuDispatcherCreate(config.cpuWorkers);
  if (!mPxCPUDispatcher) {
    throw std::runtime_error("PhysX system creation failed: failed to create CPU dispatcher");
  }
  sceneDesc.cpuDispatcher = mPxCPUDispatcher;
  mPxScene = mEngine->getPxPhysics()->createScene(sceneDesc);
  mPxScene->setSimulationEventCallback(&mSimulationCallback);
}

#ifdef SAPIEN_CUDA
PhysxSystemGpu::PhysxSystemGpu(std::shared_ptr<Device> device) {
  if (!PhysxDefault::GetGPUEnabled()) {
    throw std::runtime_error(
        "sapien.physx.enable_gpu() must be called before creating a PhysX GPU system.");
  }

  if (!device) {
    device = findDevice("cuda");
    if (!device) {
      throw std::runtime_error("failed to find a CUDA device for PhysX GPU");
    }
  } else if (!device->isCuda()) {
    throw std::runtime_error(
        "failed to create PhysX GPU system: device provided does not support CUDA");
  }
  mDevice = device;

  auto &config = mSceneConfig;
  PxSceneDesc sceneDesc(mEngine->getPxPhysics()->getTolerancesScale());
  sceneDesc.gravity = Vec3ToPxVec3(config.gravity);
  sceneDesc.filterShader = TypeAffinityIgnoreFilterShaderGpu;
  sceneDesc.solverType = config.enableTGS ? PxSolverType::eTGS : PxSolverType::ePGS;
  sceneDesc.bounceThresholdVelocity = config.bounceThreshold;

  sceneDesc.gpuDynamicsConfig = PhysxDefault::getGpuMemoryConfig();

  PxSceneFlags sceneFlags;
  if (config.enableEnhancedDeterminism) {
    sceneFlags |= PxSceneFlag::eENABLE_ENHANCED_DETERMINISM;
  }
  if (config.enablePCM) {
    sceneFlags |= PxSceneFlag::eENABLE_PCM;
  }
  if (config.enableCCD) {
    sceneFlags |= PxSceneFlag::eENABLE_CCD;
  }
  if (config.enableFrictionEveryIteration) {
    sceneFlags |= PxSceneFlag::eENABLE_FRICTION_EVERY_ITERATION;
  }

  sceneFlags |= PxSceneFlag::eENABLE_GPU_DYNAMICS;
  sceneFlags |= PxSceneFlag::eENABLE_DIRECT_GPU_API;
  sceneDesc.broadPhaseType = PxBroadPhaseType::eGPU;
  sceneDesc.cudaContextManager = mEngine->getCudaContextManager(device->cudaId);
  if (!config.enablePCM) {
    logger::warn("PCM must be enabled when using GPU.");
    sceneFlags |= PxSceneFlag::eENABLE_PCM;
  }

  sceneDesc.flags = sceneFlags;

  mPxCPUDispatcher = PxDefaultCpuDispatcherCreate(config.cpuWorkers);
  if (!mPxCPUDispatcher) {
    throw std::runtime_error("PhysX system creation failed: failed to create CPU dispatcher");
  }
  sceneDesc.cpuDispatcher = mPxCPUDispatcher;
  mPxScene = mEngine->getPxPhysics()->createScene(sceneDesc);
}
#else
PhysxSystemGpu::PhysxSystemGpu(std::shared_ptr<Device> device) {
  throw std::runtime_error(
        "Does not support PhysX GPU system.");
}
#endif

void PhysxSystemCpu::registerComponent(std::shared_ptr<PhysxRigidDynamicComponent> component) {
  mRigidDynamicComponents.insert(component);
}
void PhysxSystemCpu::registerComponent(std::shared_ptr<PhysxRigidStaticComponent> component) {
  mRigidStaticComponents.insert(component);
}
void PhysxSystemCpu::registerComponent(std::shared_ptr<PhysxArticulationLinkComponent> component) {
  mArticulationLinkComponents.insert(component);
}
void PhysxSystemCpu::unregisterComponent(std::shared_ptr<PhysxRigidDynamicComponent> component) {
  mRigidDynamicComponents.erase(component);
}
void PhysxSystemCpu::unregisterComponent(std::shared_ptr<PhysxRigidStaticComponent> component) {
  mRigidStaticComponents.erase(component);
}
void PhysxSystemCpu::unregisterComponent(
    std::shared_ptr<PhysxArticulationLinkComponent> component) {
  mArticulationLinkComponents.erase(component);
}
std::vector<std::shared_ptr<PhysxRigidDynamicComponent>>
PhysxSystemCpu::getRigidDynamicComponents() const {
  return {mRigidDynamicComponents.begin(), mRigidDynamicComponents.end()};
}
std::vector<std::shared_ptr<PhysxRigidStaticComponent>>
PhysxSystemCpu::getRigidStaticComponents() const {
  return {mRigidStaticComponents.begin(), mRigidStaticComponents.end()};
}
std::vector<std::shared_ptr<PhysxArticulationLinkComponent>>
PhysxSystemCpu::getArticulationLinkComponents() const {
  return {mArticulationLinkComponents.begin(), mArticulationLinkComponents.end()};
}

#ifdef SAPIEN_CUDA
void PhysxSystemGpu::registerComponent(std::shared_ptr<PhysxRigidDynamicComponent> component) {
  mRigidDynamicComponents.insert(component);
  mGpuInitialized = false;
}
void PhysxSystemGpu::registerComponent(std::shared_ptr<PhysxRigidStaticComponent> component) {
  mRigidStaticComponents.insert(component);
  mGpuInitialized = false;
}
void PhysxSystemGpu::registerComponent(std::shared_ptr<PhysxArticulationLinkComponent> component) {
  mArticulationLinkComponents.insert(component);
  mGpuInitialized = false;
}
void PhysxSystemGpu::unregisterComponent(std::shared_ptr<PhysxRigidDynamicComponent> component) {
  mRigidDynamicComponents.erase(component);
  mGpuInitialized = false;
}
void PhysxSystemGpu::unregisterComponent(std::shared_ptr<PhysxRigidStaticComponent> component) {
  mRigidStaticComponents.erase(component);
  mGpuInitialized = false;
}
void PhysxSystemGpu::unregisterComponent(
    std::shared_ptr<PhysxArticulationLinkComponent> component) {
  mArticulationLinkComponents.erase(component);
  mGpuInitialized = false;
}
std::vector<std::shared_ptr<PhysxRigidDynamicComponent>>
PhysxSystemGpu::getRigidDynamicComponents() const {
  return {mRigidDynamicComponents.begin(), mRigidDynamicComponents.end()};
}
std::vector<std::shared_ptr<PhysxRigidStaticComponent>>
PhysxSystemGpu::getRigidStaticComponents() const {
  return {mRigidStaticComponents.begin(), mRigidStaticComponents.end()};
}
std::vector<std::shared_ptr<PhysxArticulationLinkComponent>>
PhysxSystemGpu::getArticulationLinkComponents() const {
  return {mArticulationLinkComponents.begin(), mArticulationLinkComponents.end()};
}
#endif

std::unique_ptr<PhysxHitInfo> PhysxSystemCpu::raycast(Vec3 const &origin, Vec3 const &direction,
                                                      float distance) {
  PxRaycastBuffer hit;
  bool status = mPxScene->raycast(Vec3ToPxVec3(origin), Vec3ToPxVec3(direction), distance, hit);
  if (status) {
    return std::make_unique<PhysxHitInfo>(
        PxVec3ToVec3(hit.block.position), PxVec3ToVec3(hit.block.normal), hit.block.distance,
        static_cast<PhysxCollisionShape *>(hit.block.shape->userData),
        static_cast<PhysxRigidBaseComponent *>(hit.block.actor->userData));
  }
  return nullptr;
}

void PhysxSystemCpu::step() {
  mPxScene->simulate(mTimestep);
  mPxScene->fetchResults(true);
  for (auto c : mRigidStaticComponents) {
    c->syncPoseToEntity();
  }
  for (auto c : mRigidDynamicComponents) {
    c->syncPoseToEntity();
  }
  for (auto c : mArticulationLinkComponents) {
    c->syncPoseToEntity();
  }
}

#ifdef SAPIEN_CUDA
void PhysxSystemGpu::step() {
  if (!mGpuInitialized) {
    throw std::runtime_error("failed to step: gpu simulation is not initialized.");
  }

  mContactUpToDate = false;

  ++mTotalSteps;
  mPxScene->simulate(mTimestep);
  mPxScene->fetchResults(true);

  // TODO: does the GPU API require fetch results?
}

void PhysxSystemGpu::stepStart() {
  if (!mGpuInitialized) {
    throw std::runtime_error("failed to step: gpu simulation is not initialized.");
  }

  mContactUpToDate = false;

  ++mTotalSteps;
  mPxScene->simulate(mTimestep);
}

void PhysxSystemGpu::stepFinish() { mPxScene->fetchResults(true); }
#endif

std::string PhysxSystemCpu::packState() const {
  std::ostringstream ss;
  for (auto &actor : mRigidDynamicComponents) {
    Pose pose = actor->getPose();
    Vec3 v = actor->getLinearVelocity();
    Vec3 w = actor->getAngularVelocity();
    ss.write(reinterpret_cast<const char *>(&pose), sizeof(Pose));
    ss.write(reinterpret_cast<const char *>(&v), sizeof(Vec3));
    ss.write(reinterpret_cast<const char *>(&w), sizeof(Vec3));
  }
  for (auto &link : mArticulationLinkComponents) {
    if (link->isRoot()) {
      auto art = link->getArticulation();

      Pose pose = art->getRootPose();
      Vec3 v = art->getRootLinearVelocity();
      Vec3 w = art->getRootAngularVelocity();
      ss.write(reinterpret_cast<const char *>(&pose), sizeof(Pose));
      ss.write(reinterpret_cast<const char *>(&v), sizeof(Vec3));
      ss.write(reinterpret_cast<const char *>(&w), sizeof(Vec3));

      auto qpos = art->getQpos();
      auto qvel = art->getQvel();

      ss.write(reinterpret_cast<const char *>(qpos.data()), qpos.size() * sizeof(float));
      ss.write(reinterpret_cast<const char *>(qvel.data()), qvel.size() * sizeof(float));

      for (auto j : art->getActiveJoints()) {
        auto pos = j->getDriveTargetPosition();
        auto vel = j->getDriveTargetVelocity();

        ss.write(reinterpret_cast<const char *>(pos.data()), pos.size() * sizeof(float));
        ss.write(reinterpret_cast<const char *>(vel.data()), vel.size() * sizeof(float));
      }
    }
  }
  return ss.str();
}

void PhysxSystemCpu::unpackState(std::string const &data) {
  std::istringstream ss(data);
  for (auto &actor : mRigidDynamicComponents) {
    Pose pose;
    Vec3 v, w;
    ss.read(reinterpret_cast<char *>(&pose), sizeof(Pose));
    ss.read(reinterpret_cast<char *>(&v), sizeof(Vec3));
    ss.read(reinterpret_cast<char *>(&w), sizeof(Vec3));
    actor->setPose(pose);
    if (!actor->isKinematic()) {
      actor->setLinearVelocity(v);
      actor->setAngularVelocity(w);
    }
  }
  for (auto &link : mArticulationLinkComponents) {
    if (link->isRoot()) {
      Pose pose;
      Vec3 v, w;
      ss.read(reinterpret_cast<char *>(&pose), sizeof(Pose));
      ss.read(reinterpret_cast<char *>(&v), sizeof(Vec3));
      ss.read(reinterpret_cast<char *>(&w), sizeof(Vec3));
      auto art = link->getArticulation();
      art->setRootPose(pose);
      art->setRootLinearVelocity(v);
      art->setRootAngularVelocity(w);

      Eigen::VectorXf qpos;
      Eigen::VectorXf qvel;
      qpos.resize(art->getDof());
      qvel.resize(art->getDof());
      ss.read(reinterpret_cast<char *>(qpos.data()), qpos.size() * sizeof(float));
      ss.read(reinterpret_cast<char *>(qvel.data()), qvel.size() * sizeof(float));
      art->setQpos(qpos);
      art->setQvel(qvel);

      for (auto j : art->getActiveJoints()) {
        Eigen::VectorXf pos, vel;
        pos.resize(j->getDof());
        vel.resize(j->getDof());
        ss.read(reinterpret_cast<char *>(pos.data()), pos.size() * sizeof(float));
        ss.read(reinterpret_cast<char *>(vel.data()), vel.size() * sizeof(float));
        j->setDriveTargetPosition(pos);
        j->setDriveTargetVelocity(vel);
      }
    }
  }
}

int PhysxSystem::getArticulationCount() const {
  // TODO: ensure this count matches registered articulations
  return getPxScene()->getNbArticulations();
}

int PhysxSystem::computeArticulationMaxDof() const {
  int result = 0;
  uint32_t count = getPxScene()->getNbArticulations();
  std::vector<PxArticulationReducedCoordinate *> articulations(count);
  getPxScene()->getArticulations(articulations.data(), count);
  for (auto a : articulations) {
    result = std::max(result, static_cast<int>(a->getDofs()));
  }
  return result;
}

int PhysxSystem::computeArticulationMaxLinkCount() const {
  int result = 0;
  uint32_t count = getPxScene()->getNbArticulations();
  std::vector<PxArticulationReducedCoordinate *> articulations(count);
  getPxScene()->getArticulations(articulations.data(), count);
  for (auto a : articulations) {
    result = std::max(result, static_cast<int>(a->getNbLinks()));
  }
  return result;
}

#ifdef SAPIEN_CUDA
void PhysxSystemGpu::gpuInit() {
  ++mTotalSteps;
  ensureCudaDevice();
  mPxScene->simulate(mTimestep);
  while (!mPxScene->fetchResults(true)) {
  }

  mGpuArticulationCount = getArticulationCount();
  mGpuArticulationMaxDof = computeArticulationMaxDof();
  mGpuArticulationMaxLinkCount = computeArticulationMaxLinkCount();

  allocateCudaBuffers();

  // ensureCudaDevice();
  // mCudaEventRecord.init();
  // mCudaEventWait.init();

  mGpuInitialized = true;
}

void PhysxSystemGpu::checkGpuInitialized() const {
  if (!isInitialized()) {
    throw std::runtime_error("GPU PhysX is not initialized.");
  }
}

void PhysxSystemGpu::gpuSetCudaStream(uintptr_t stream) { mCudaStream = (cudaStream_t)stream; }

std::shared_ptr<PhysxGpuContactPairImpulseQuery> PhysxSystemGpu::gpuCreateContactPairImpulseQuery(
    std::vector<std::pair<std::shared_ptr<PhysxRigidBaseComponent>,
                          std::shared_ptr<PhysxRigidBaseComponent>>> const &bodyPairs) {
  if (bodyPairs.empty()) {
    throw std::runtime_error("failed to create contact query: empty body pairs");
  }
  std::vector<ActorPairQuery> pairs;
  for (uint32_t i = 0; i < bodyPairs.size(); ++i) {
    auto &[b0, b1] = bodyPairs[i];
    if (!b0 || !b1) {
      throw std::runtime_error("failed to create contact query: invalid body");
    }
    int order{0};
    ActorPair pair = makeActorPair(b0->getPxActor(), b1->getPxActor(), order);
    pairs.push_back({pair, i, order});
  }

  std::sort(pairs.begin(), pairs.end(),
            [](ActorPairQuery const &a, ActorPairQuery const &b) { return a.pair < b.pair; });

  static_assert(sizeof(ActorPairQuery) == 24);

  ensureCudaDevice();
  CudaArray query({static_cast<int>(pairs.size()), 6}, "i4");
  checkCudaErrors(cudaMemcpy(query.ptr, pairs.data(), pairs.size() * sizeof(ActorPairQuery),
                             cudaMemcpyHostToDevice));
  CudaArray buffer({static_cast<int>(pairs.size()), 3}, "f4");

  auto res = std::make_shared<PhysxGpuContactPairImpulseQuery>();
  res->query = std::move(query);
  res->buffer = std::move(buffer);
  return res;
}

std::shared_ptr<PhysxGpuContactBodyImpulseQuery> PhysxSystemGpu::gpuCreateContactBodyImpulseQuery(
    std::vector<std::shared_ptr<PhysxRigidBaseComponent>> const &bodies) {
  if (bodies.empty()) {
    throw std::runtime_error("failed to create contact query: empty body list");
  }
  std::vector<ActorQuery> actors;
  for (uint32_t i = 0; i < bodies.size(); ++i) {
    if (!bodies[i]) {
      throw std::runtime_error("failed to create contact actors: invalid body");
    }
    actors.push_back({bodies[i]->getPxActor(), i});
  }

  std::sort(actors.begin(), actors.end(),
            [](ActorQuery const &a, ActorQuery const &b) { return a.actor < b.actor; });
  static_assert(sizeof(ActorQuery) == 16);

  ensureCudaDevice();
  CudaArray query({static_cast<int>(actors.size()), 4}, "i4");
  checkCudaErrors(cudaMemcpy(query.ptr, actors.data(), actors.size() * sizeof(ActorQuery),
                             cudaMemcpyHostToDevice));
  CudaArray buffer({static_cast<int>(actors.size()), 3}, "f4");

  // TODO: use dedicated type, do not reuse contact query
  auto res = std::make_shared<PhysxGpuContactBodyImpulseQuery>();
  res->query = std::move(query);
  res->buffer = std::move(buffer);
  return res;
}
#endif

inline static int upperPowerOf2(int x) {
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x++;
  return x;
}

#ifdef SAPIEN_CUDA
void PhysxSystemGpu::copyContactData() {
  if (mContactUpToDate) {
    return;
  }

  ensureCudaDevice();
  if (!mCudaContactCount.ptr) {
    mCudaContactCount = CudaArray({1}, "u4");
  }

  if (!mCudaContactBuffer.ptr) {
    mCudaContactBuffer = CudaArray({1024, sizeof(PxGpuContactPair)}, "u1");
  }

  SAPIEN_PROFILE_BLOCK_BEGIN("fetch contact count");
  mPxScene->getDirectGPUAPI().copyContactData(
      mCudaContactBuffer.ptr, reinterpret_cast<PxU32 *>(mCudaContactCount.ptr),
      static_cast<PxU32>(mCudaContactBuffer.shape[0]));
  cudaMemcpy(&mContactCount, mCudaContactCount.ptr, sizeof(int), cudaMemcpyDeviceToHost);
  SAPIEN_PROFILE_BLOCK_END;

  int size = upperPowerOf2(mContactCount);
  if (mCudaContactBuffer.shape[0] < size) {
    SAPIEN_PROFILE_BLOCK("re-allocate contact buffer");
    mCudaContactBuffer = CudaArray({size, sizeof(PxGpuContactPair)}, "u1");
  }

  mPxScene->getDirectGPUAPI().copyContactData(
      mCudaContactBuffer.ptr, reinterpret_cast<PxU32 *>(mCudaContactCount.ptr),
      static_cast<PxU32>(size));

  mContactUpToDate = true;
}

void PhysxSystemGpu::gpuQueryContactPairImpulses(PhysxGpuContactPairImpulseQuery const &query) {
  SAPIEN_PROFILE_FUNCTION;
  query.query.handle().checkShape({-1, 6});

  ensureCudaDevice();
  cudaMemsetAsync(query.buffer.ptr, 0, query.query.shape.at(0) * 3 * sizeof(float), mCudaStream);

  copyContactData();

  if (mContactCount) {
    handle_contacts((PxGpuContactPair *)mCudaContactBuffer.ptr, mContactCount,
                  (ActorPairQuery *)query.query.ptr, query.query.shape.at(0),
                  (Vec3 *)query.buffer.ptr, mCudaStream);
  }
  cudaStreamSynchronize(mCudaStream);
}

void PhysxSystemGpu::gpuQueryContactBodyImpulses(PhysxGpuContactBodyImpulseQuery const &query) {
  SAPIEN_PROFILE_FUNCTION;
  query.query.handle().checkShape({-1, 4});

  ensureCudaDevice();
  cudaMemsetAsync(query.buffer.ptr, 0, query.query.shape.at(0) * 3 * sizeof(float), mCudaStream);

  copyContactData();

  if (mContactCount) {
    handle_net_contact_force((PxGpuContactPair *)mCudaContactBuffer.ptr, mContactCount,
                           (ActorQuery *)query.query.ptr, query.query.shape.at(0),
                           (Vec3 *)query.buffer.ptr, mCudaStream);
  }
  cudaStreamSynchronize(mCudaStream);
}

void PhysxSystemGpu::gpuFetchRigidDynamicData() {
  checkGpuInitialized();

  if (mRigidDynamicComponents.empty()) {
    return;
  }

  ensureCudaDevice();
  auto &dapi = mPxScene->getDirectGPUAPI();
  PxU32 n = static_cast<PxU32>(mCudaRigidDynamicIndexBuffer.shape.at(0));
  const auto *gpuIndices =
      reinterpret_cast<const PxRigidDynamicGPUIndex *>(mCudaRigidDynamicGpuIndexBuffer.ptr);
  dapi.getRigidDynamicData(mCudaRigidDynamicPoseBuffer.ptr, gpuIndices,
                          PxRigidDynamicGPUAPIReadType::eGLOBAL_POSE, n, NULL, mCudaEventWait.event);
  dapi.getRigidDynamicData(mCudaRigidDynamicVelBuffer.ptr, gpuIndices,
                          PxRigidDynamicGPUAPIReadType::eLINEAR_VELOCITY, n, NULL,
                          mCudaEventWait.event);
  dapi.getRigidDynamicData(mCudaRigidDynamicAngBuffer.ptr, gpuIndices,
                          PxRigidDynamicGPUAPIReadType::eANGULAR_VELOCITY, n, NULL,
                          mCudaEventWait.event);
  mCudaEventWait.wait(mCudaStream);
  body_data_physx_to_sapien(mCudaRigidDynamicHandle.ptr, mCudaRigidDynamicPoseBuffer.ptr,
                            mCudaRigidDynamicVelBuffer.ptr, mCudaRigidDynamicAngBuffer.ptr,
                            mCudaRigidDynamicOffsetBuffer.ptr, n, mCudaStream);
}

void PhysxSystemGpu::gpuFetchArticulationLinkPose() {
  checkGpuInitialized();

  if (mGpuArticulationCount == 0) {
    return;
  }

  ensureCudaDevice();
  mPxScene->getDirectGPUAPI().getArticulationData(
      mCudaLinkPoseScratch.ptr,
      reinterpret_cast<const PxArticulationGPUIndex *>(mCudaArticulationIndexBuffer.ptr),
      PxArticulationGPUAPIReadType::eLINK_GLOBAL_POSE,
      static_cast<PxU32>(mCudaArticulationIndexBuffer.shape.at(0)), NULL, mCudaEventWait.event);
  mCudaEventWait.wait(mCudaStream);
  link_pose_physx_to_sapien(
      mCudaLinkHandle.ptr, mCudaLinkPoseScratch.ptr, mCudaArticulationOffsetBuffer.ptr,
      mGpuArticulationMaxLinkCount,
      mCudaArticulationIndexBuffer.shape.at(0) * mGpuArticulationMaxLinkCount, mCudaStream);
}

void PhysxSystemGpu::gpuFetchArticulationLinkVel() {
  checkGpuInitialized();

  if (mGpuArticulationCount == 0) {
    return;
  }

  ensureCudaDevice();
  auto &dapi = mPxScene->getDirectGPUAPI();
  const auto *gpuIndices =
      reinterpret_cast<const PxArticulationGPUIndex *>(mCudaArticulationIndexBuffer.ptr);
  PxU32 n = static_cast<PxU32>(mCudaArticulationIndexBuffer.shape.at(0));
  dapi.getArticulationData(mCudaLinkVelScratch.ptr, gpuIndices,
                          PxArticulationGPUAPIReadType::eLINK_LINEAR_VELOCITY, n, NULL,
                          mCudaEventWait.event);
  dapi.getArticulationData(mCudaLinkAngVelScratch.ptr, gpuIndices,
                          PxArticulationGPUAPIReadType::eLINK_ANGULAR_VELOCITY, n, NULL,
                          mCudaEventWait.event);
  mCudaEventWait.wait(mCudaStream);
  link_vel_physx_to_sapien(mCudaLinkHandle.ptr, mCudaLinkVelScratch.ptr,
                           mCudaLinkAngVelScratch.ptr,
                           n * mGpuArticulationMaxLinkCount, mCudaStream);
}

void PhysxSystemGpu::gpuFetchArticulationQpos() {
  checkGpuInitialized();

  if (mGpuArticulationCount == 0) {
    return;
  }

  ensureCudaDevice();

  mPxScene->getDirectGPUAPI().getArticulationData(
      mCudaQposHandle.ptr,
      reinterpret_cast<const PxArticulationGPUIndex *>(mCudaArticulationIndexBuffer.ptr),
      PxArticulationGPUAPIReadType::eJOINT_POSITION,
      static_cast<PxU32>(mCudaArticulationIndexBuffer.shape.at(0)), NULL, mCudaEventWait.event);
  mCudaEventWait.wait(mCudaStream);
}

void PhysxSystemGpu::gpuFetchArticulationQvel() {
  checkGpuInitialized();

  if (mGpuArticulationCount == 0) {
    return;
  }

  ensureCudaDevice();

  mPxScene->getDirectGPUAPI().getArticulationData(
      mCudaQvelHandle.ptr,
      reinterpret_cast<const PxArticulationGPUIndex *>(mCudaArticulationIndexBuffer.ptr),
      PxArticulationGPUAPIReadType::eJOINT_VELOCITY,
      static_cast<PxU32>(mCudaArticulationIndexBuffer.shape.at(0)), NULL, mCudaEventWait.event);
  mCudaEventWait.wait(mCudaStream);
}

void PhysxSystemGpu::gpuFetchArticulationQTargetPos() {
  checkGpuInitialized();

  if (mGpuArticulationCount == 0) {
    return;
  }

  ensureCudaDevice();

  mPxScene->getDirectGPUAPI().getArticulationData(
      mCudaQTargetPosHandle.ptr,
      reinterpret_cast<const PxArticulationGPUIndex *>(mCudaArticulationIndexBuffer.ptr),
      PxArticulationGPUAPIReadType::eJOINT_TARGET_POSITION,
      static_cast<PxU32>(mCudaArticulationIndexBuffer.shape.at(0)), NULL, mCudaEventWait.event);
  mCudaEventWait.wait(mCudaStream);
}

void PhysxSystemGpu::gpuFetchArticulationQTargetVel() {
  checkGpuInitialized();

  if (mGpuArticulationCount == 0) {
    return;
  }

  ensureCudaDevice();

  mPxScene->getDirectGPUAPI().getArticulationData(
      mCudaQTargetVelHandle.ptr,
      reinterpret_cast<const PxArticulationGPUIndex *>(mCudaArticulationIndexBuffer.ptr),
      PxArticulationGPUAPIReadType::eJOINT_TARGET_VELOCITY,
      static_cast<PxU32>(mCudaArticulationIndexBuffer.shape.at(0)), NULL, mCudaEventWait.event);
  mCudaEventWait.wait(mCudaStream);
}

void PhysxSystemGpu::gpuFetchArticulationLinkIncomingJointForce() {
  checkGpuInitialized();

  if (mGpuArticulationCount == 0) {
    return;
  }

  ensureCudaDevice();

  mPxScene->getDirectGPUAPI().getArticulationData(
      mCudaArticulationLinkIncomingJointForceBuffer.ptr,
      reinterpret_cast<const PxArticulationGPUIndex *>(mCudaArticulationIndexBuffer.ptr),
      PxArticulationGPUAPIReadType::eLINK_INCOMING_JOINT_FORCE,
      static_cast<PxU32>(mCudaArticulationIndexBuffer.shape.at(0)), NULL, mCudaEventWait.event);
  mCudaEventWait.wait(mCudaStream);
}

void PhysxSystemGpu::gpuFetchArticulationQacc() {
  checkGpuInitialized();

  if (mGpuArticulationCount == 0) {
    return;
  }
  ensureCudaDevice();

  mPxScene->getDirectGPUAPI().getArticulationData(
      mCudaQaccHandle.ptr,
      reinterpret_cast<const PxArticulationGPUIndex *>(mCudaArticulationIndexBuffer.ptr),
      PxArticulationGPUAPIReadType::eJOINT_ACCELERATION,
      static_cast<PxU32>(mCudaArticulationIndexBuffer.shape.at(0)), NULL, mCudaEventWait.event);
  mCudaEventWait.wait(mCudaStream);
}

void PhysxSystemGpu::gpuUpdateArticulationKinematics() {
  checkGpuInitialized();

  ensureCudaDevice();

  // wait for previous apply to finish
  checkCudaErrors(cudaDeviceSynchronize());

  mPxScene->getDirectGPUAPI().computeArticulationData(
      nullptr,
      reinterpret_cast<const PxArticulationGPUIndex *>(mCudaArticulationIndexBuffer.ptr),
      PxArticulationGPUAPIComputeType::eUPDATE_KINEMATIC,
      static_cast<PxU32>(mCudaArticulationIndexBuffer.shape.at(0)), NULL, NULL);
}

void PhysxSystemGpu::gpuApplyRigidDynamicData() {
  SAPIEN_PROFILE_FUNCTION;
  checkGpuInitialized();

  if (mRigidDynamicComponents.empty()) {
    return;
  }

  ensureCudaDevice();
  body_data_sapien_to_physx(mCudaRigidDynamicScratch.ptr, mCudaRigidDynamicHandle.ptr,
                            mCudaRigidDynamicOffsetBuffer.ptr,
                            mCudaRigidDynamicIndexBuffer.shape.at(0), mCudaStream);

  body_data_split_for_apply(mCudaRigidDynamicPoseBuffer.ptr, mCudaRigidDynamicVelBuffer.ptr,
                            mCudaRigidDynamicAngBuffer.ptr, mCudaRigidDynamicScratch.ptr,
                            mCudaRigidDynamicIndexBuffer.shape.at(0), mCudaStream);
  mCudaEventRecord.record(mCudaStream);
  auto &dapi = mPxScene->getDirectGPUAPI();
  const auto *gpuIndices =
      reinterpret_cast<const PxRigidDynamicGPUIndex *>(mCudaRigidDynamicGpuIndexBuffer.ptr);
  PxU32 n = static_cast<PxU32>(mCudaRigidDynamicIndexBuffer.shape.at(0));
  dapi.setRigidDynamicData(mCudaRigidDynamicPoseBuffer.ptr, gpuIndices,
                           PxRigidDynamicGPUAPIWriteType::eGLOBAL_POSE, n, mCudaEventRecord.event,
                           mCudaEventWait.event);
  dapi.setRigidDynamicData(mCudaRigidDynamicVelBuffer.ptr, gpuIndices,
                           PxRigidDynamicGPUAPIWriteType::eLINEAR_VELOCITY, n, NULL,
                           mCudaEventWait.event);
  dapi.setRigidDynamicData(mCudaRigidDynamicAngBuffer.ptr, gpuIndices,
                           PxRigidDynamicGPUAPIWriteType::eANGULAR_VELOCITY, n, NULL,
                           mCudaEventWait.event);
  mCudaEventWait.wait(mCudaStream);
}

void PhysxSystemGpu::gpuApplyRigidDynamicData(CudaArrayHandle const &indices) {
  SAPIEN_PROFILE_FUNCTION;
  checkGpuInitialized();
  indices.checkCongiguous();
  indices.checkShape({-1});
  indices.checkStride({sizeof(int)});

  if (mRigidDynamicComponents.empty()) {
    return;
  }

  ensureCudaDevice();
  body_data_sapien_to_physx(mCudaRigidDynamicScratch.ptr, mCudaRigidDynamicIndexScratch.ptr,
                            mCudaRigidDynamicHandle.ptr, mCudaRigidDynamicIndexBuffer.ptr,
                            indices.ptr, mCudaRigidDynamicOffsetBuffer.ptr, indices.shape.at(0),
                            mCudaStream);

  body_data_split_for_apply(mCudaRigidDynamicPoseBuffer.ptr, mCudaRigidDynamicVelBuffer.ptr,
                            mCudaRigidDynamicAngBuffer.ptr, mCudaRigidDynamicScratch.ptr,
                            indices.shape.at(0), mCudaStream);
  gather_rigid_dynamic_gpu_indices(mCudaRigidDynamicIndexScratch.ptr,
                                   mCudaRigidDynamicGpuIndexBuffer.ptr, indices.ptr,
                                   indices.shape.at(0), mCudaStream);
  mCudaEventRecord.record(mCudaStream);
  auto &dapi = mPxScene->getDirectGPUAPI();
  const auto *gpuIndices =
      reinterpret_cast<const PxRigidDynamicGPUIndex *>(mCudaRigidDynamicIndexScratch.ptr);
  PxU32 n = static_cast<PxU32>(indices.shape.at(0));
  dapi.setRigidDynamicData(mCudaRigidDynamicPoseBuffer.ptr, gpuIndices,
                           PxRigidDynamicGPUAPIWriteType::eGLOBAL_POSE, n, mCudaEventRecord.event,
                           mCudaEventWait.event);
  dapi.setRigidDynamicData(mCudaRigidDynamicVelBuffer.ptr, gpuIndices,
                           PxRigidDynamicGPUAPIWriteType::eLINEAR_VELOCITY, n, NULL,
                           mCudaEventWait.event);
  dapi.setRigidDynamicData(mCudaRigidDynamicAngBuffer.ptr, gpuIndices,
                           PxRigidDynamicGPUAPIWriteType::eANGULAR_VELOCITY, n, NULL,
                           mCudaEventWait.event);
  mCudaEventWait.wait(mCudaStream);
}

void PhysxSystemGpu::gpuApplyRigidDynamicForce() {
  SAPIEN_PROFILE_FUNCTION;
  checkGpuInitialized();
  if (mRigidDynamicComponents.empty()) {
    return;
  }
  mCudaEventRecord.record(mCudaStream);
  auto &dapi = mPxScene->getDirectGPUAPI();
  const auto *gpuIndices =
      reinterpret_cast<const PxRigidDynamicGPUIndex *>(mCudaRigidDynamicGpuIndexBuffer.ptr);
  PxU32 n = static_cast<PxU32>(mCudaRigidDynamicIndexBuffer.shape.at(0));
  dapi.setRigidDynamicData(mCudaRigidDynamicForceHandle.ptr, gpuIndices,
                           PxRigidDynamicGPUAPIWriteType::eFORCE, n, mCudaEventRecord.event,
                           mCudaEventWait.event);
  mCudaEventWait.wait(mCudaStream);
}

void PhysxSystemGpu::gpuApplyRigidDynamicTorque() {
  SAPIEN_PROFILE_FUNCTION;
  checkGpuInitialized();
  if (mRigidDynamicComponents.empty()) {
    return;
  }
  mCudaEventRecord.record(mCudaStream);
  auto &dapi = mPxScene->getDirectGPUAPI();
  const auto *gpuIndices =
      reinterpret_cast<const PxRigidDynamicGPUIndex *>(mCudaRigidDynamicGpuIndexBuffer.ptr);
  PxU32 n = static_cast<PxU32>(mCudaRigidDynamicIndexBuffer.shape.at(0));
  dapi.setRigidDynamicData(mCudaRigidDynamicTorqueHandle.ptr, gpuIndices,
                           PxRigidDynamicGPUAPIWriteType::eTORQUE, n, mCudaEventRecord.event,
                           mCudaEventWait.event);
  mCudaEventWait.wait(mCudaStream);
}

void PhysxSystemGpu::gpuApplyArticulationRootPose() {
  gpuApplyArticulationRootPose(mCudaArticulationIndexBuffer.handle());
}

void PhysxSystemGpu::gpuApplyArticulationRootPose(CudaArrayHandle const &indices) {
  SAPIEN_PROFILE_FUNCTION;
  checkGpuInitialized();
  indices.checkCongiguous();
  indices.checkShape({-1});
  indices.checkStride({sizeof(int)});

  if (mGpuArticulationCount == 0) {
    return;
  }
  ensureCudaDevice();

  root_pose_sapien_to_physx(mCudaLinkPoseScratch.ptr, mCudaLinkHandle.ptr, indices.ptr,
                            mCudaArticulationOffsetBuffer.ptr, mGpuArticulationMaxLinkCount,
                            indices.shape.at(0), mCudaStream);
  mCudaEventRecord.record(mCudaStream);
  auto &dapi = mPxScene->getDirectGPUAPI();
  const auto *gpuIndices =
      reinterpret_cast<const PxArticulationGPUIndex *>(indices.ptr);
  dapi.setArticulationData(mCudaLinkPoseScratch.ptr, gpuIndices,
                           PxArticulationGPUAPIWriteType::eROOT_GLOBAL_POSE,
                           static_cast<PxU32>(indices.shape.at(0)),
                           mCudaEventRecord.event, mCudaEventWait.event);
  mCudaEventWait.wait(mCudaStream);
}

void PhysxSystemGpu::gpuApplyArticulationRootVel() {
  gpuApplyArticulationRootVel(mCudaArticulationIndexBuffer.handle());
}

void PhysxSystemGpu::gpuApplyArticulationRootVel(CudaArrayHandle const &indices) {
  SAPIEN_PROFILE_FUNCTION;
  checkGpuInitialized();
  indices.checkCongiguous();
  indices.checkShape({-1});
  indices.checkStride({sizeof(int)});

  if (mGpuArticulationCount == 0) {
    return;
  }
  ensureCudaDevice();

  root_vel_sapien_to_physx(mCudaRootLinearVelScratch.ptr, mCudaRootAngVelScratch.ptr,
                           mCudaLinkHandle.ptr, indices.ptr, mGpuArticulationMaxLinkCount,
                           indices.shape.at(0), mCudaStream);
  mCudaEventRecord.record(mCudaStream);
  auto &dapi = mPxScene->getDirectGPUAPI();
  const auto *gpuIndices =
      reinterpret_cast<const PxArticulationGPUIndex *>(indices.ptr);
  PxU32 n = static_cast<PxU32>(indices.shape.at(0));
  dapi.setArticulationData(mCudaRootLinearVelScratch.ptr, gpuIndices,
                           PxArticulationGPUAPIWriteType::eROOT_LINEAR_VELOCITY, n,
                           mCudaEventRecord.event, mCudaEventWait.event);
  dapi.setArticulationData(mCudaRootAngVelScratch.ptr, gpuIndices,
                           PxArticulationGPUAPIWriteType::eROOT_ANGULAR_VELOCITY, n, NULL,
                           mCudaEventWait.event);
  mCudaEventWait.wait(mCudaStream);
}

void PhysxSystemGpu::gpuApplyArticulationQpos() {
  gpuApplyArticulationQpos(mCudaArticulationIndexBuffer.handle());
}

void PhysxSystemGpu::gpuApplyArticulationQpos(CudaArrayHandle const &indices) {
  SAPIEN_PROFILE_FUNCTION;
  checkGpuInitialized();
  indices.checkCongiguous();
  indices.checkShape({-1});
  indices.checkStride({sizeof(int)});

  if (mGpuArticulationCount == 0) {
    return;
  }
  ensureCudaDevice();

  mCudaEventRecord.record(mCudaStream);
  gather_articulation_dof_data(mCudaArticulationGatherScratch.ptr, mCudaQposHandle.ptr,
                               indices.ptr, indices.shape.at(0), mGpuArticulationMaxDof,
                               mCudaStream);
  auto &dapi = mPxScene->getDirectGPUAPI();
  const auto *gpuIndices =
      reinterpret_cast<const PxArticulationGPUIndex *>(indices.ptr);
  dapi.setArticulationData(mCudaArticulationGatherScratch.ptr, gpuIndices,
                           PxArticulationGPUAPIWriteType::eJOINT_POSITION,
                           static_cast<PxU32>(indices.shape.at(0)),
                           mCudaEventRecord.event, mCudaEventWait.event);
  mCudaEventWait.wait(mCudaStream);
}

void PhysxSystemGpu::gpuApplyArticulationQvel() {
  gpuApplyArticulationQvel(mCudaArticulationIndexBuffer.handle());
}

void PhysxSystemGpu::gpuApplyArticulationQvel(CudaArrayHandle const &indices) {
  SAPIEN_PROFILE_FUNCTION;
  checkGpuInitialized();
  indices.checkCongiguous();
  indices.checkShape({-1});
  indices.checkStride({sizeof(int)});

  if (mGpuArticulationCount == 0) {
    return;
  }
  ensureCudaDevice();

  mCudaEventRecord.record(mCudaStream);
  gather_articulation_dof_data(mCudaArticulationGatherScratch.ptr, mCudaQvelHandle.ptr,
                               indices.ptr, indices.shape.at(0), mGpuArticulationMaxDof,
                               mCudaStream);
  auto &dapi = mPxScene->getDirectGPUAPI();
  const auto *gpuIndices =
      reinterpret_cast<const PxArticulationGPUIndex *>(indices.ptr);
  dapi.setArticulationData(mCudaArticulationGatherScratch.ptr, gpuIndices,
                           PxArticulationGPUAPIWriteType::eJOINT_VELOCITY,
                           static_cast<PxU32>(indices.shape.at(0)),
                           mCudaEventRecord.event, mCudaEventWait.event);
  mCudaEventWait.wait(mCudaStream);
}

void PhysxSystemGpu::gpuApplyArticulationQf() {
  gpuApplyArticulationQf(mCudaArticulationIndexBuffer.handle());
}

void PhysxSystemGpu::gpuApplyArticulationQf(CudaArrayHandle const &indices) {
  SAPIEN_PROFILE_FUNCTION;
  checkGpuInitialized();
  indices.checkCongiguous();
  indices.checkShape({-1});
  indices.checkStride({sizeof(int)});

  if (mGpuArticulationCount == 0) {
    return;
  }
  ensureCudaDevice();

  mCudaEventRecord.record(mCudaStream);
  gather_articulation_dof_data(mCudaArticulationGatherScratch.ptr, mCudaQfHandle.ptr,
                               indices.ptr, indices.shape.at(0), mGpuArticulationMaxDof,
                               mCudaStream);
  auto &dapi = mPxScene->getDirectGPUAPI();
  const auto *gpuIndices =
      reinterpret_cast<const PxArticulationGPUIndex *>(indices.ptr);
  dapi.setArticulationData(mCudaArticulationGatherScratch.ptr, gpuIndices,
                           PxArticulationGPUAPIWriteType::eJOINT_FORCE,
                           static_cast<PxU32>(indices.shape.at(0)),
                           mCudaEventRecord.event, mCudaEventWait.event);
  mCudaEventWait.wait(mCudaStream);
}

void PhysxSystemGpu::gpuApplyArticulationQTargetPos() {
  gpuApplyArticulationQTargetPos(mCudaArticulationIndexBuffer.handle());
}

void PhysxSystemGpu::gpuApplyArticulationQTargetPos(CudaArrayHandle const &indices) {
  SAPIEN_PROFILE_FUNCTION;
  checkGpuInitialized();
  indices.checkCongiguous();
  indices.checkShape({-1});
  indices.checkStride({sizeof(int)});

  if (mGpuArticulationCount == 0) {
    return;
  }
  ensureCudaDevice();

  mCudaEventRecord.record(mCudaStream);
  gather_articulation_dof_data(mCudaArticulationGatherScratch.ptr, mCudaQTargetPosHandle.ptr,
                               indices.ptr, indices.shape.at(0), mGpuArticulationMaxDof,
                               mCudaStream);
  auto &dapi = mPxScene->getDirectGPUAPI();
  const auto *gpuIndices =
      reinterpret_cast<const PxArticulationGPUIndex *>(indices.ptr);
  dapi.setArticulationData(mCudaArticulationGatherScratch.ptr, gpuIndices,
                           PxArticulationGPUAPIWriteType::eJOINT_TARGET_POSITION,
                           static_cast<PxU32>(indices.shape.at(0)),
                           mCudaEventRecord.event, mCudaEventWait.event);
  mCudaEventWait.wait(mCudaStream);
}

void PhysxSystemGpu::gpuApplyArticulationQTargetVel() {
  gpuApplyArticulationQTargetVel(mCudaArticulationIndexBuffer.handle());
}

void PhysxSystemGpu::gpuApplyArticulationQTargetVel(CudaArrayHandle const &indices) {
  SAPIEN_PROFILE_FUNCTION;
  checkGpuInitialized();
  indices.checkCongiguous();
  indices.checkShape({-1});
  indices.checkStride({sizeof(int)});

  if (mGpuArticulationCount == 0) {
    return;
  }
  ensureCudaDevice();

  mCudaEventRecord.record(mCudaStream);
  gather_articulation_dof_data(mCudaArticulationGatherScratch.ptr, mCudaQTargetVelHandle.ptr,
                               indices.ptr, indices.shape.at(0), mGpuArticulationMaxDof,
                               mCudaStream);
  auto &dapi = mPxScene->getDirectGPUAPI();
  const auto *gpuIndices =
      reinterpret_cast<const PxArticulationGPUIndex *>(indices.ptr);
  dapi.setArticulationData(mCudaArticulationGatherScratch.ptr, gpuIndices,
                           PxArticulationGPUAPIWriteType::eJOINT_TARGET_VELOCITY,
                           static_cast<PxU32>(indices.shape.at(0)),
                           mCudaEventRecord.event, mCudaEventWait.event);
  mCudaEventWait.wait(mCudaStream);
}

void PhysxSystemGpu::syncPosesGpuToCpu() {
  checkGpuInitialized();
  gpuFetchRigidDynamicData();
  gpuFetchArticulationLinkPose();
  if (mCudaHostRigidBodyBuffer.shape != mCudaRigidBodyBuffer.shape) {
    mCudaHostRigidBodyBuffer =
        CudaHostArray(mCudaRigidBodyBuffer.shape, mCudaRigidBodyBuffer.type);
  }
  mCudaHostRigidBodyBuffer.copyFrom(mCudaRigidBodyBuffer);
  auto data = (SapienBodyData *)mCudaHostRigidBodyBuffer.ptr;

  for (auto &body : mRigidDynamicComponents) {
    assert(body->getGpuPoseIndex() >= 0);
    body->getEntity()->internalSyncPose(
        {data[body->getGpuPoseIndex()].p, data[body->getGpuPoseIndex()].q});
  }
  for (auto &body : mArticulationLinkComponents) {
    assert(body->getGpuPoseIndex() >= 0);
    body->getEntity()->internalSyncPose(
        {data[body->getGpuPoseIndex()].p, data[body->getGpuPoseIndex()].q});
  }
}

std::vector<float> PhysxSystemGpu::gpuDownloadArticulationQpos(int index) {
  ensureCudaDevice();
  gpuFetchArticulationQpos();
  cudaStreamSynchronize(mCudaStream);

  if (index < 0 || index >= mGpuArticulationMaxDof) {
    throw std::runtime_error("failed to download articulation qpos: invalid index");
  }

  std::vector<float> buffer(mGpuArticulationMaxDof);

  cudaMemcpy(buffer.data(), &((float *)mCudaQposHandle.ptr)[index * mGpuArticulationMaxDof],
             mGpuArticulationMaxDof * sizeof(float), cudaMemcpyDeviceToHost);
  return buffer;
}

void PhysxSystemGpu::gpuUploadArticulationQpos(int index, Eigen::VectorXf const &q) {
  ensureCudaDevice();
  cudaStreamSynchronize(mCudaStream);
  if (index < 0 || index >= mGpuArticulationMaxDof) {
    throw std::runtime_error("failed to download articulation qpos: invalid index");
  }

  cudaMemcpy(&((float *)mCudaQposHandle.ptr)[index * mGpuArticulationMaxDof], q.data(),
             q.size() * sizeof(float), cudaMemcpyHostToDevice);
  CudaArray cudaIndex({1}, "i4");
  gpuApplyArticulationQpos(cudaIndex.handle());
}

void PhysxSystemGpu::setSceneOffset(std::shared_ptr<Scene> scene, Vec3 offset) {
  // clean up occasionally
  if (mSceneOffset.size() % 1024 == 0) {
    std::erase_if(mSceneOffset, [](const auto &p) { return p.first.expired(); });
  }

  mSceneOffset[scene] = offset;
}

Vec3 PhysxSystemGpu::getSceneOffset(std::shared_ptr<Scene> scene) const {
  if (mSceneOffset.contains(scene)) {
    return mSceneOffset.at(scene);
  }
  return Vec3(0.0f);
}

void PhysxSystemGpu::allocateCudaBuffers() {
  SAPIEN_PROFILE_FUNCTION;
  int rigidDynamicCount = mRigidDynamicComponents.size();
  int rigidBodyCount = rigidDynamicCount + mGpuArticulationCount * mGpuArticulationMaxLinkCount;

  ensureCudaDevice();

  // rigid body data buffer
  mCudaRigidBodyBuffer = CudaArray({rigidBodyCount, 13}, "f4");
  mCudaRigidDynamicHandle = CudaArrayHandle{.shape = {rigidDynamicCount, 13},
                                            .strides = {52, 4},
                                            .type = "f4",
                                            .cudaId = mCudaRigidBodyBuffer.cudaId,
                                            .ptr = (float *)mCudaRigidBodyBuffer.ptr};
  mCudaLinkHandle =
      CudaArrayHandle{.shape = {mGpuArticulationCount, mGpuArticulationMaxLinkCount, 13},
                      .strides = {mGpuArticulationMaxLinkCount * 52, 52, 4},
                      .type = "f4",
                      .cudaId = mCudaRigidBodyBuffer.cudaId,
                      .ptr = (float *)mCudaRigidBodyBuffer.ptr + 13 * rigidDynamicCount};

  mCudaArticulationLinkIncomingJointForceBuffer =
      CudaArray({mGpuArticulationCount, mGpuArticulationMaxLinkCount, 6}, "f4");

  // rigid body force torque buffer
  mCudaRigidBodyForceBuffer = CudaArray({rigidBodyCount, 4}, "f4");
  mCudaRigidDynamicForceHandle = CudaArrayHandle{.shape = {rigidDynamicCount, 4},
                                                 .strides = {16, 4},
                                                 .type = "f4",
                                                 .cudaId = mCudaRigidBodyForceBuffer.cudaId,
                                                 .ptr = (float *)mCudaRigidBodyForceBuffer.ptr};
  // TODO: articulation link handle
  mCudaRigidBodyTorqueBuffer = CudaArray({rigidBodyCount, 4}, "f4");
  mCudaRigidDynamicTorqueHandle = CudaArrayHandle{.shape = {rigidDynamicCount, 4},
                                                  .strides = {16, 4},
                                                  .type = "f4",
                                                  .cudaId = mCudaRigidBodyTorqueBuffer.cudaId,
                                                  .ptr = (float *)mCudaRigidBodyTorqueBuffer.ptr};
  // TODO: articulation link handle

  mCudaArticulationBuffer = CudaArray({mGpuArticulationCount * mGpuArticulationMaxDof * 6}, "f4");

  mCudaQposHandle = CudaArrayHandle{.shape = {mGpuArticulationCount, mGpuArticulationMaxDof},
                                    .strides = {mGpuArticulationMaxDof * 4, 4},
                                    .type = "f4",
                                    .cudaId = mCudaArticulationBuffer.cudaId,
                                    .ptr = mCudaArticulationBuffer.ptr};

  mCudaQvelHandle = CudaArrayHandle{.shape = {mGpuArticulationCount, mGpuArticulationMaxDof},
                                    .strides = {mGpuArticulationMaxDof * 4, 4},
                                    .type = "f4",
                                    .cudaId = mCudaArticulationBuffer.cudaId,
                                    .ptr = (float *)mCudaArticulationBuffer.ptr +
                                           mGpuArticulationCount * mGpuArticulationMaxDof};

  mCudaQfHandle = CudaArrayHandle{.shape = {mGpuArticulationCount, mGpuArticulationMaxDof},
                                  .strides = {mGpuArticulationMaxDof * 4, 4},
                                  .type = "f4",
                                  .cudaId = mCudaArticulationBuffer.cudaId,
                                  .ptr = (float *)mCudaArticulationBuffer.ptr +
                                         mGpuArticulationCount * mGpuArticulationMaxDof * 2};

  mCudaQaccHandle = CudaArrayHandle{.shape = {mGpuArticulationCount, mGpuArticulationMaxDof},
                                    .strides = {mGpuArticulationMaxDof * 4, 4},
                                    .type = "f4",
                                    .cudaId = mCudaArticulationBuffer.cudaId,
                                    .ptr = (float *)mCudaArticulationBuffer.ptr +
                                           mGpuArticulationCount * mGpuArticulationMaxDof * 3};

  mCudaQTargetPosHandle =
      CudaArrayHandle{.shape = {mGpuArticulationCount, mGpuArticulationMaxDof},
                      .strides = {mGpuArticulationMaxDof * 4, 4},
                      .type = "f4",
                      .cudaId = mCudaArticulationBuffer.cudaId,
                      .ptr = (float *)mCudaArticulationBuffer.ptr +
                             mGpuArticulationCount * mGpuArticulationMaxDof * 4};

  mCudaQTargetVelHandle =
      CudaArrayHandle{.shape = {mGpuArticulationCount, mGpuArticulationMaxDof},
                      .strides = {mGpuArticulationMaxDof * 4, 4},
                      .type = "f4",
                      .cudaId = mCudaArticulationBuffer.cudaId,
                      .ptr = (float *)mCudaArticulationBuffer.ptr +
                             mGpuArticulationCount * mGpuArticulationMaxDof * 5};

  {
    mCudaRigidDynamicIndexBuffer = CudaArray({rigidDynamicCount, 4}, "i4");
    mCudaRigidDynamicIndexScratch = CudaArray({rigidDynamicCount, 4}, "i4");
    mCudaRigidDynamicOffsetBuffer = CudaArray({rigidDynamicCount, 3}, "f4");

    std::vector<std::array<float, 3>> host_offset;
    std::vector<PxU32> host_gpu_index;
    auto bodies = getRigidDynamicComponents();
    for (uint32_t i = 0; i < bodies.size(); ++i) {
      Vec3 offset = getSceneOffset(bodies[i]->getScene());
      host_offset.push_back({offset.x, offset.y, offset.z});
      host_gpu_index.push_back(
          static_cast<PxRigidDynamic *>(bodies[i]->getPxActor())->getGPUIndex());
      bodies[i]->internalSetGpuIndex(i);
    }

    mCudaRigidDynamicGpuIndexBuffer =
        CudaArray({rigidDynamicCount}, "u4");
    checkCudaErrors(cudaMemcpy(mCudaRigidDynamicGpuIndexBuffer.ptr, host_gpu_index.data(),
                               host_gpu_index.size() * sizeof(PxU32), cudaMemcpyHostToDevice));
    mCudaRigidDynamicPoseBuffer =
        CudaArray({static_cast<int>(rigidDynamicCount * sizeof(PxTransform))}, "u1");
    mCudaRigidDynamicVelBuffer =
        CudaArray({static_cast<int>(rigidDynamicCount * sizeof(PxVec3))}, "u1");
    mCudaRigidDynamicAngBuffer =
        CudaArray({static_cast<int>(rigidDynamicCount * sizeof(PxVec3))}, "u1");
    checkCudaErrors(cudaMemcpy(mCudaRigidDynamicOffsetBuffer.ptr, host_offset.data(),
                               host_offset.size() * sizeof(float) * 3, cudaMemcpyHostToDevice));
  }

  {
    mCudaArticulationOffsetBuffer = CudaArray({mGpuArticulationCount, 3}, "f4");
    std::vector<std::array<float, 3>> host_offset(mGpuArticulationCount, {0.f, 0.f, 0.f});
    for (auto a : getArticulationLinkComponents()) {
      uint32_t art_idx = a->getArticulation()->getPxArticulation()->getGPUIndex();
      if (a->isRoot()) {
        Vec3 offset = getSceneOffset(a->getScene());
        host_offset.at(art_idx) = {offset.x, offset.y, offset.z};
      }
      a->internalSetGpuPoseIndex(rigidDynamicCount + art_idx * mGpuArticulationMaxLinkCount +
                                 a->getIndex());
    }
    checkCudaErrors(cudaMemcpy(mCudaArticulationOffsetBuffer.ptr, host_offset.data(),
                               host_offset.size() * sizeof(float) * 3, cudaMemcpyHostToDevice));

    // index
    mCudaArticulationIndexBuffer = CudaArray({mGpuArticulationCount}, "i4");
    mCudaArticulationIndexScratch = CudaArray({mGpuArticulationCount}, "i4");
    std::vector<int> host_index;
    for (int i = 0; i < mGpuArticulationCount; ++i) {
      host_index.push_back(i);
    }
    checkCudaErrors(cudaMemcpy(mCudaArticulationIndexBuffer.ptr, host_index.data(),
                               host_index.size() * sizeof(int), cudaMemcpyHostToDevice));
  }

  int bodySize = rigidDynamicCount * 16 * sizeof(float);
  mCudaRigidDynamicScratch = CudaArray({bodySize}, "u1");

  int linkPoseSize = mGpuArticulationCount * mGpuArticulationMaxLinkCount * 7 * sizeof(float);
  mCudaLinkPoseScratch = CudaArray({linkPoseSize}, "u1");

  int linkVelSize = mGpuArticulationCount * mGpuArticulationMaxLinkCount * 6 * sizeof(float);
  mCudaLinkVelScratch = CudaArray({linkVelSize}, "u1");
  mCudaLinkAngVelScratch = CudaArray({linkVelSize}, "u1");
  mCudaRootLinearVelScratch =
      CudaArray({static_cast<int>(mGpuArticulationCount * sizeof(PxVec3))}, "u1");
  mCudaRootAngVelScratch =
      CudaArray({static_cast<int>(mGpuArticulationCount * sizeof(PxVec3))}, "u1");
  mCudaArticulationGatherScratch =
      CudaArray({mGpuArticulationCount * mGpuArticulationMaxDof}, "f4");
}

void PhysxSystemGpu::ensureCudaDevice() { checkCudaErrors(cudaSetDevice(mDevice->cudaId)); }
#endif

PhysxSystem::~PhysxSystem() { logger::info("Deleting PhysxSystem"); }

PhysxSystemCpu::~PhysxSystemCpu() {
  if (mPxScene) {
    mPxScene->release();
  }
  if (mPxCPUDispatcher) {
    mPxCPUDispatcher->release();
  }
}

#ifdef SAPIEN_CUDA
PhysxSystemGpu::~PhysxSystemGpu() {
  if (mPxScene) {
    mPxScene->release();
  }
  if (mPxCPUDispatcher) {
    mPxCPUDispatcher->release();
  }
}
#endif
} // namespace physx
} // namespace sapien
