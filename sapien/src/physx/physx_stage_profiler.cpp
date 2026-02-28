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

std::string toLower(std::string_view input) {
  std::string out;
  out.reserve(input.size());
  for (char c : input) {
    out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  }
  return out;
}

StageBucket classifyZone(char const *eventName) {
  if (!eventName || !eventName[0]) {
    return StageBucket::eOther;
  }

  std::string lower = toLower(eventName);
  if (lower.find("edge coloring") != std::string::npos ||
      lower.find("constraintpartition") != std::string::npos) {
    return StageBucket::eColoring;
  }
  if (lower.find("broadphase") != std::string::npos ||
      lower.find("broad phase") != std::string::npos) {
    return StageBucket::eBroadphase;
  }
  if (lower.find("narrowphase") != std::string::npos ||
      lower.find("narrow phase") != std::string::npos) {
    return StageBucket::eNarrowphase;
  }
  if (lower.find("solve") != std::string::npos || lower.find("solver") != std::string::npos ||
      lower.find("postsolver") != std::string::npos) {
    return StageBucket::eSolver;
  }
  if (lower.find("integrate") != std::string::npos || lower.find("update") != std::string::npos ||
      lower.find("kinematic") != std::string::npos) {
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

    uint64_t startNs{0};
    StageBucket stage = StageBucket::eOther;
    if (!unpackProfilerData(profilerData, startNs, stage)) {
      return;
    }

    if (!mFrameActive.load(std::memory_order_acquire)) {
      return;
    }

    std::lock_guard<std::mutex> lock(mMutex);
    if (!mFrameActive.load(std::memory_order_relaxed)) {
      return;
    }
    accumulateLocked(stage, eventName, nowNs() - startNs);
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

    uint64_t totalNs = 0;
    for (uint64_t value : mLastFrame.stageNs) {
      totalNs += value;
    }
    out["total_ms"] = totalNs * 1e-6;
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
