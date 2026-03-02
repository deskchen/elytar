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
#include "sapien/physx/physx_stage_profiler.h"

#include <PxPhysicsAPI.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace sapien {
namespace physx {
namespace {

enum class StageBucket : uint8_t {
  eBroadphase = 0,
  eNarrowphase = 1,
  eColoring = 2,
  eSolver = 3,
  eUpdate = 4,
  eOther = 5,
};

constexpr size_t kStageBucketCount = 6;
constexpr uintptr_t kPackedStageMask = 0x7;

uint64_t nowNs() {
  using Clock = std::chrono::steady_clock;
  using Ns = std::chrono::nanoseconds;
  return std::chrono::duration_cast<Ns>(Clock::now().time_since_epoch()).count();
}

bool containsCI(std::string_view haystack, std::string_view needle) {
  if (needle.size() > haystack.size())
    return false;
  for (size_t i = 0; i <= haystack.size() - needle.size(); ++i) {
    bool match = true;
    for (size_t j = 0; j < needle.size(); ++j) {
      if (std::tolower(static_cast<unsigned char>(haystack[i + j])) !=
          std::tolower(static_cast<unsigned char>(needle[j]))) {
        match = false;
        break;
      }
    }
    if (match)
      return true;
  }
  return false;
}

StageBucket classifyZone(char const *eventName) {
  if (!eventName || !eventName[0]) {
    return StageBucket::eOther;
  }

  std::string_view name(eventName);

  // --- Coloring / constraint partitioning (check before solver) ---
  if (containsCI(name, "partition") || containsCI(name, "coloring") ||
      containsCI(name, "AccumulateSlabs") || containsCI(name, "Compaction")) {
    return StageBucket::eColoring;
  }

  // --- Broadphase ---
  if (containsCI(name, "broadphase") || containsCI(name, "broad phase") ||
      containsCI(name, "AABBManager") || containsCI(name, "BroadPhaseSap") ||
      containsCI(name, "islandGen") || containsCI(name, "islandInsertion") ||
      containsCI(name, "islandTouches") || containsCI(name, "islandLostTouches") ||
      containsCI(name, "setEdgesConnected") || containsCI(name, "findPaths") ||
      containsCI(name, "wakeIslands") || containsCI(name, "insertNewEdges") ||
      containsCI(name, "removeEdges") || containsCI(name, "processNewEdges") ||
      containsCI(name, "processLostEdges") || containsCI(name, "resetDirtyEdges") ||
      containsCI(name, "DestroyedEdges") || containsCI(name, "DestroyedNodes") ||
      containsCI(name, "FoundPatches") || containsCI(name, "LostPatches") ||
      containsCI(name, "FoundPair") || containsCI(name, "LostPair") ||
      containsCI(name, "processFoundSolverPatches") ||
      containsCI(name, "processLostSolverPatches") ||
      containsCI(name, "deactivation") || containsCI(name, "ActivatedContacts") ||
      containsCI(name, "ActivatedJoints") || containsCI(name, "DeactivatedContacts") ||
      containsCI(name, "DeactivatingJoints") || containsCI(name, "PostThirdPass") ||
      containsCI(name, "postThirdPass")) {
    return StageBucket::eBroadphase;
  }

  // --- Narrowphase ---
  if (containsCI(name, "narrowPhase") || containsCI(name, "narrow phase") ||
      containsCI(name, "NPhase") || containsCI(name, "ContactManager") ||
      containsCI(name, "GpuNarrowPhase") || containsCI(name, "collision") ||
      containsCI(name, "collide") || containsCI(name, "processLostContact") ||
      containsCI(name, "processNpLostTouch") || containsCI(name, "processNPLostTouch") ||
      containsCI(name, "lostTouchReports") || containsCI(name, "processLostTouchPairs") ||
      containsCI(name, "registerInteraction") || containsCI(name, "unregisterInteraction") ||
      containsCI(name, "registerSceneInteraction") ||
      containsCI(name, "contactDistances") || containsCI(name, "InteractionNewTouch") ||
      containsCI(name, "persistentContactEvents") ||
      containsCI(name, "processNarrowPhaseLostTouch") ||
      containsCI(name, "processNarrowPhaseTouchEvents") ||
      containsCI(name, "unblockNarrowPhase") ||
      containsCI(name, "dirtyShaders") || containsCI(name, "dirtyShapes") ||
      containsCI(name, "updateBoundsAndShapes") ||
      containsCI(name, "fireCustomFiltering")) {
    return StageBucket::eNarrowphase;
  }

  // --- Solver (dynamics, constraints, TGS/PGS, integration, DMA, friction) ---
  if (containsCI(name, "solve") || containsCI(name, "solver") ||
      containsCI(name, "dynamics") || containsCI(name, "Tgs") || containsCI(name, "TGS") ||
      containsCI(name, "constraint") || containsCI(name, "Integrate") ||
      containsCI(name, "preIntegrate") || containsCI(name, "friction") ||
      containsCI(name, "GpuDynamics") || containsCI(name, "DMABack") ||
      containsCI(name, "DMAUp") || containsCI(name, "dmaBack") || containsCI(name, "dmaUp") ||
      containsCI(name, "gpuMemDma") || containsCI(name, "JointManager")) {
    return StageBucket::eSolver;
  }

  // --- Update (post-solve bookkeeping, scene queries, callbacks, finalization) ---
  if (containsCI(name, "update") || containsCI(name, "kinematic") ||
      containsCI(name, "fetchResult") || containsCI(name, "afterIntegration") ||
      containsCI(name, "SimulationController") || containsCI(name, "SimController") ||
      containsCI(name, "finalization") || containsCI(name, "completion") ||
      containsCI(name, "sceneQuery") || containsCI(name, "pruner") ||
      containsCI(name, "Pruning") || containsCI(name, "flushShapes") ||
      containsCI(name, "Callback") || containsCI(name, "callback") ||
      containsCI(name, "pvdFrame") || containsCI(name, "visualize") ||
      containsCI(name, "checkResults") || containsCI(name, "buildActiveActors") ||
      containsCI(name, "wakeObjectsUp") || containsCI(name, "putObjectsToSleep") ||
      containsCI(name, "putInteractionsToSleep") ||
      containsCI(name, "wakeInteractions") ||
      containsCI(name, "postSolver") || containsCI(name, "postCallbacks") ||
      containsCI(name, "checkConstraintBreakage") ||
      containsCI(name, "fireOutOfBounds") ||
      containsCI(name, "taskFrameworkSetup") ||
      containsCI(name, "copyToGpu") || containsCI(name, "updateScBody") ||
      containsCI(name, "updateTransformCache") || containsCI(name, "updateBodiesAndShapes") ||
      containsCI(name, "updateGPUJoints") || containsCI(name, "updateForces") ||
      containsCI(name, "clearIslandData") || containsCI(name, "resetDependencies") ||
      containsCI(name, "stepSetup") || containsCI(name, "prepareCollide") ||
      containsCI(name, "updateKinematic") ||
      containsCI(name, "integrateKinematicPose") ||
      containsCI(name, "integrateAndUpdate")) {
    return StageBucket::eUpdate;
  }

  return StageBucket::eOther;
}

void *packProfilerData(uint64_t startNs, StageBucket stage) {
  uint64_t startWithOffset = startNs + 1;
  uint64_t packed = (startWithOffset << 3) | static_cast<uint64_t>(stage);
  return reinterpret_cast<void *>(static_cast<uintptr_t>(packed));
}

bool unpackProfilerData(void *profilerData, uint64_t &startNs, StageBucket &stage) {
  uintptr_t packed = reinterpret_cast<uintptr_t>(profilerData);
  if (packed == 0) {
    return false;
  }

  uint64_t stageValue = static_cast<uint64_t>(packed & kPackedStageMask);
  if (stageValue >= kStageBucketCount) {
    stage = StageBucket::eOther;
  } else {
    stage = static_cast<StageBucket>(stageValue);
  }

  startNs = (static_cast<uint64_t>(packed) >> 3) - 1;
  return true;
}

struct FrameMetrics {
  std::array<uint64_t, kStageBucketCount> stageNs{};
  std::unordered_map<std::string, uint64_t> zoneNs;
};

// Per-thread stack for exclusive (self) time tracking.
// Each entry tracks how much time was spent in child zones.
thread_local std::vector<uint64_t> tl_childNsStack;

class StageProfilerCallback final : public ::physx::PxProfilerCallback {
public:
  void *zoneStart(char const *eventName, bool detached, uint64_t contextId) override {
    if (!mEnabled.load(std::memory_order_relaxed) ||
        !mFrameActive.load(std::memory_order_acquire)) {
      return nullptr;
    }

    uint64_t start = nowNs();
    if (detached) {
      std::lock_guard<std::mutex> lock(mMutex);
      mDetachedStartNs[detachedKey(eventName, contextId)].push_back(start);
      return nullptr;
    }
    tl_childNsStack.push_back(0);
    return packProfilerData(start, classifyZone(eventName));
  }

  void zoneEnd(void *profilerData, char const *eventName, bool detached,
               uint64_t contextId) override {
    if (!mEnabled.load(std::memory_order_relaxed)) {
      return;
    }

    if (detached) {
      uint64_t startNs{0};
      {
        std::lock_guard<std::mutex> lock(mMutex);
        auto key = detachedKey(eventName, contextId);
        auto it = mDetachedStartNs.find(key);
        if (it == mDetachedStartNs.end() || it->second.empty()) {
          return;
        }
        startNs = it->second.back();
        it->second.pop_back();
        if (it->second.empty()) {
          mDetachedStartNs.erase(it);
        }

        if (!mFrameActive.load(std::memory_order_acquire)) {
          return;
        }

        accumulateLocked(classifyZone(eventName), eventName, nowNs() - startNs);
      }
      return;
    }

    uint64_t endNs = nowNs();
    uint64_t startNs{0};
    StageBucket stage = StageBucket::eOther;
    if (!unpackProfilerData(profilerData, startNs, stage)) {
      return;
    }

    uint64_t durationNs = endNs - startNs;

    // Pop child time from thread-local stack to compute exclusive (self) time.
    uint64_t childNs = 0;
    if (!tl_childNsStack.empty()) {
      childNs = tl_childNsStack.back();
      tl_childNsStack.pop_back();
    }
    uint64_t selfNs = (durationNs > childNs) ? (durationNs - childNs) : 0;

    // Report our inclusive duration to parent so it can subtract us.
    if (!tl_childNsStack.empty()) {
      tl_childNsStack.back() += durationNs;
    }

    if (!mFrameActive.load(std::memory_order_acquire)) {
      return;
    }

    std::lock_guard<std::mutex> lock(mMutex);
    if (!mFrameActive.load(std::memory_order_relaxed)) {
      return;
    }
    accumulateLocked(stage, eventName, selfNs);
  }

  void setEnabled(bool enabled) { mEnabled.store(enabled, std::memory_order_release); }

  bool isEnabled() const { return mEnabled.load(std::memory_order_acquire); }

  void beginFrame() {
    std::lock_guard<std::mutex> lock(mMutex);
    mCurrentFrame = FrameMetrics{};
    mDetachedStartNs.clear();
    mFrameActive.store(true, std::memory_order_release);
  }

  void endFrame() {
    mFrameActive.store(false, std::memory_order_release);
    std::lock_guard<std::mutex> lock(mMutex);
    mLastFrame = mCurrentFrame;
    mDetachedStartNs.clear();
  }

  std::map<std::string, double> getLastFrameStageMs() const {
    std::lock_guard<std::mutex> lock(mMutex);
    std::map<std::string, double> out;
    out["broadphase_ms"] = mLastFrame.stageNs[static_cast<size_t>(StageBucket::eBroadphase)] *
                           1e-6;
    out["narrowphase_ms"] = mLastFrame.stageNs[static_cast<size_t>(StageBucket::eNarrowphase)] *
                            1e-6;
    out["coloring_ms"] = mLastFrame.stageNs[static_cast<size_t>(StageBucket::eColoring)] * 1e-6;
    out["solver_ms"] = mLastFrame.stageNs[static_cast<size_t>(StageBucket::eSolver)] * 1e-6;
    out["update_ms"] = mLastFrame.stageNs[static_cast<size_t>(StageBucket::eUpdate)] * 1e-6;
    out["other_ms"] = mLastFrame.stageNs[static_cast<size_t>(StageBucket::eOther)] * 1e-6;

    // total = sum of the 5 known stages (excluding other)
    double total = out["broadphase_ms"] + out["narrowphase_ms"] + out["coloring_ms"] +
                   out["solver_ms"] + out["update_ms"];
    out["total_ms"] = total;
    return out;
  }

  std::map<std::string, double> getLastFrameZoneMs() const {
    std::lock_guard<std::mutex> lock(mMutex);
    std::map<std::string, double> out;
    for (auto const &[name, ns] : mLastFrame.zoneNs) {
      out[name] = ns * 1e-6;
    }
    return out;
  }

private:
  static std::string detachedKey(char const *eventName, uint64_t contextId) {
    std::string key = std::to_string(contextId);
    key.push_back(':');
    key.append(eventName ? eventName : "<null>");
    return key;
  }

  void accumulateLocked(StageBucket stage, char const *eventName, uint64_t durationNs) {
    mCurrentFrame.stageNs[static_cast<size_t>(stage)] += durationNs;
    mCurrentFrame.zoneNs[eventName ? eventName : "<null>"] += durationNs;
  }

  std::atomic<bool> mEnabled{false};
  std::atomic<bool> mFrameActive{false};
  mutable std::mutex mMutex;

  FrameMetrics mCurrentFrame;
  FrameMetrics mLastFrame;
  std::unordered_map<std::string, std::vector<uint64_t>> mDetachedStartNs;
};

StageProfilerCallback &getStageProfiler() {
  static StageProfilerCallback profiler;
  return profiler;
}

void installProfilerCallbacks(::physx::PxProfilerCallback *callback) {
  PxSetProfilerCallback(callback);
#if PX_SUPPORT_GPU_PHYSX
  PxSetPhysXGpuProfilerCallback(callback);
#endif
}

} // namespace

void setStageProfilerEnabled(bool enabled) {
  auto &profiler = getStageProfiler();
  profiler.setEnabled(enabled);
  installProfilerCallbacks(enabled ? &profiler : nullptr);
}

bool isStageProfilerEnabled() { return getStageProfiler().isEnabled(); }

void stageProfilerBeginFrame() { getStageProfiler().beginFrame(); }

void stageProfilerEndFrame() { getStageProfiler().endFrame(); }

std::map<std::string, double> getStageProfilerLastFrameStageMs() {
  return getStageProfiler().getLastFrameStageMs();
}

std::map<std::string, double> getStageProfilerLastFrameZoneMs() {
  return getStageProfiler().getLastFrameZoneMs();
}

} // namespace physx
} // namespace sapien
