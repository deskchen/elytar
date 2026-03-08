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
#include "./physx_system.cuh"

#include <cstdio>

namespace sapien {
namespace physx {

__global__ void body_data_sapien_to_physx_kernel(PhysxBodyData *__restrict__ physx_data,
                                                 SapienBodyData *__restrict__ sapien_data,
                                                 Vec3 *__restrict__ offset, int count) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= count) {
    return;
  }

  SapienBodyData sd = sapien_data[g];

  PhysxBodyData pd;
  pd.pose.q = {sd.q.x, sd.q.y, sd.q.z, sd.q.w};
  pd.pose.p = sd.p + offset[g];
  pd.v = sd.v;
  pd.w = sd.w;

  physx_data[g] = pd;
}

__global__ void body_data_sapien_to_physx_kernel(PhysxBodyData *__restrict__ physx_data,
                                                 int4 *__restrict__ physx_index,
                                                 SapienBodyData *__restrict__ sapien_data,
                                                 int4 *__restrict__ sapien_index,
                                                 int *__restrict__ apply_index,
                                                 Vec3 *__restrict__ offset, int count) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= count) {
    return;
  }

  int i = apply_index[g];

  SapienBodyData sd = sapien_data[i];

  PhysxBodyData pd;
  pd.pose.q = {sd.q.x, sd.q.y, sd.q.z, sd.q.w};
  pd.pose.p = sd.p + offset[i];
  pd.v = sd.v;
  pd.w = sd.w;

  physx_data[g] = pd;
  physx_index[g] = sapien_index[i];
}

__global__ void link_pose_physx_to_sapien_kernel(SapienBodyData *__restrict__ sapien_data,
                                                 PhysxPose *__restrict__ physx_pose,
                                                 Vec3 *__restrict__ offset, int link_count,
                                                 int count) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= count) {
    return;
  }

  int ai = g / link_count;

  sapien_data[g].p = physx_pose[g].p - offset[ai];
  sapien_data[g].q =
      Quat(physx_pose[g].q.w, physx_pose[g].q.x, physx_pose[g].q.y, physx_pose[g].q.z);
}

__global__ void root_pose_sapien_to_physx_kernel(PhysxPose *__restrict__ physx_pose,
                                                 SapienBodyData *__restrict__ sapien_data,
                                                 int *__restrict__ index,
                                                 Vec3 *__restrict__ offset, int link_count,
                                                 int count) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= count) {
    return;
  }

  int ai = index[g]; // ith articulation

  SapienBodyData sd = sapien_data[ai * link_count];

  physx_pose[ai] = {{sd.q.x, sd.q.y, sd.q.z, sd.q.w}, sd.p + offset[ai]};
}

__device__ int binary_search(ActorPairQuery const *__restrict__ arr, int count, ActorPair x) {
  int low = 0;
  int high = count - 1;
  while (low <= high) {
    int mid = low + (high - low) / 2;
    if (arr[mid].pair == x)
      return mid;
    if (arr[mid].pair < x)
      low = mid + 1;
    else
      high = mid - 1;
  }
  return -1;
}

__device__ int binary_search(ActorQuery const *__restrict__ arr, int count, ::physx::PxActor *x) {
  int low = 0;
  int high = count - 1;
  while (low <= high) {
    int mid = low + (high - low) / 2;
    if (arr[mid].actor == x)
      return mid;
    if (arr[mid].actor < x)
      low = mid + 1;
    else
      high = mid - 1;
  }
  return -1;
}

__global__ void handle_contacts_kernel(::physx::PxGpuContactPair *__restrict__ contacts,
                                       int contact_count, ActorPairQuery *__restrict__ query,
                                       int query_count, Vec3 *__restrict__ out_forces) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= contact_count) {
    return;
  }

  int order = 0;
  ActorPair pair = makeActorPair(contacts[g].actor0, contacts[g].actor1, order);

  int index = binary_search(query, query_count, pair);
  if (index < 0) {
    return;
  }
  uint32_t id = query[index].id;

  order *= query[index].order;

  ::physx::PxContactPatch *patches = (::physx::PxContactPatch *)contacts[g].contactPatches;
  ::physx::PxContact *points = (::physx::PxContact *)contacts[g].contactPoints;

  float *forces = contacts[g].contactForces;

  Vec3 force = Vec3(0.f);
  for (int pi = 0; pi < contacts[g].nbPatches; ++pi) {
    Vec3 normal(patches[pi].normal.x, patches[pi].normal.y, patches[pi].normal.z);
    for (int i = 0; i < patches[pi].nbContacts; ++i) {
      int ci = patches[pi].startContactIndex + i;
      float f = forces[ci];
      force += normal * (f * order);
      // printf("normal = %f %f %f, normal length2 = %f, separation = %f, force = %f\n", normal.x,
      //        normal.y, normal.z, normal.dot(normal), points[ci].separation, f);
    }
  }
  atomicAdd(&out_forces[id].x, force.x);
  atomicAdd(&out_forces[id].y, force.y);
  atomicAdd(&out_forces[id].z, force.z);
}

__global__ void handle_net_contact_force_kernel(::physx::PxGpuContactPair *__restrict__ contacts,
                                                int contact_count, ActorQuery *__restrict__ query,
                                                int query_count, Vec3 *__restrict__ out_forces) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= contact_count) {
    return;
  }

  ::physx::PxActor *actor0 = contacts[g].actor0;
  ::physx::PxActor *actor1 = contacts[g].actor1;

  int index0 = binary_search(query, query_count, actor0);
  int index1 = binary_search(query, query_count, actor1);

  if (index0 < 0 && index1 < 0) {
    return;
  }

  ::physx::PxContactPatch *patches = (::physx::PxContactPatch *)contacts[g].contactPatches;
  ::physx::PxContact *points = (::physx::PxContact *)contacts[g].contactPoints;

  float *forces = contacts[g].contactForces;

  Vec3 force = Vec3(0.f);
  for (int pi = 0; pi < contacts[g].nbPatches; ++pi) {
    Vec3 normal(patches[pi].normal.x, patches[pi].normal.y, patches[pi].normal.z);
    for (int i = 0; i < patches[pi].nbContacts; ++i) {
      int ci = patches[pi].startContactIndex + i;
      float f = forces[ci];
      force += normal * f;
    }
  }

  if (index0 >= 0) {
    int id = query[index0].id;
    atomicAdd(&out_forces[id].x, force.x);
    atomicAdd(&out_forces[id].y, force.y);
    atomicAdd(&out_forces[id].z, force.z);
  }
  if (index1 >= 0) {
    int id = query[index1].id;
    atomicAdd(&out_forces[id].x, -force.x);
    atomicAdd(&out_forces[id].y, -force.y);
    atomicAdd(&out_forces[id].z, -force.z);
  }
}

constexpr int BLOCK_SIZE = 128;

struct PxTransformLayout {
  float qx, qy, qz, qw;
  float px, py, pz;
};

__global__ void body_data_physx_to_sapien_kernel(
    SapienBodyData *__restrict__ sapien_data, PxTransformLayout *__restrict__ pose,
    float *__restrict__ vel, float *__restrict__ ang, Vec3 *__restrict__ offset, int count) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= count)
    return;
  sapien_data[g] = {
      Vec3{pose[g].px, pose[g].py, pose[g].pz} - offset[g],
      Quat(pose[g].qw, pose[g].qx, pose[g].qy, pose[g].qz),
      Vec3{vel[g * 3], vel[g * 3 + 1], vel[g * 3 + 2]},
      Vec3{ang[g * 3], ang[g * 3 + 1], ang[g * 3 + 2]},
  };
}

void body_data_physx_to_sapien(void *sapien_data, void *pose_data, void *vel_data,
                              void *ang_data, void *offset, int count, cudaStream_t stream) {
  body_data_physx_to_sapien_kernel<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0,
                                     stream>>>(
      (SapienBodyData *)sapien_data, (PxTransformLayout *)pose_data, (float *)vel_data,
      (float *)ang_data, (Vec3 *)offset, count);
}

void body_data_sapien_to_physx(void *physx_data, void *sapien_data, void *offset, int count,
                               cudaStream_t stream) {
  body_data_sapien_to_physx_kernel<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0,
                                     stream>>>(
      (PhysxBodyData *)physx_data, (SapienBodyData *)sapien_data, (Vec3 *)offset, count);
}

void body_data_sapien_to_physx(void *physx_data, void *physx_index, void *sapien_data,
                               void *sapien_index, void *apply_index, void *offset, int count,
                               cudaStream_t stream) {
  body_data_sapien_to_physx_kernel<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0,
                                     stream>>>((PhysxBodyData *)physx_data, (int4 *)physx_index,
                                               (SapienBodyData *)sapien_data, (int4 *)sapien_index,
                                               (int *)apply_index, (Vec3 *)offset, count);
}

void link_pose_physx_to_sapien(void *sapien_data, void *physx_pose, void *offset, int link_count,
                               int count, cudaStream_t stream) {
  link_pose_physx_to_sapien_kernel<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0,
                                     stream>>>(
      (SapienBodyData *)sapien_data, (PhysxPose *)physx_pose, (Vec3 *)offset, link_count, count);
}

void root_pose_sapien_to_physx(void *physx_pose, void *sapien_data, void *index, void *offset,
                               int link_count, int count, cudaStream_t stream) {
  root_pose_sapien_to_physx_kernel<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0,
                                     stream>>>((PhysxPose *)physx_pose,
                                               (SapienBodyData *)sapien_data, (int *)index,
                                               (Vec3 *)offset, link_count, count);
}

__global__ void link_vel_physx_to_sapien_kernel(SapienBodyData *__restrict__ sapien_data,
                                                float *__restrict__ linear_vel,
                                                float *__restrict__ angular_vel, int count) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= count)
    return;
  sapien_data[g].v = Vec3{linear_vel[g * 3], linear_vel[g * 3 + 1], linear_vel[g * 3 + 2]};
  sapien_data[g].w = Vec3{angular_vel[g * 3], angular_vel[g * 3 + 1], angular_vel[g * 3 + 2]};
}

void link_vel_physx_to_sapien(void *sapien_data, void *linear_vel, void *angular_vel, int count,
                              cudaStream_t stream) {
  link_vel_physx_to_sapien_kernel<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0,
                                    stream>>>((SapienBodyData *)sapien_data, (float *)linear_vel,
                                              (float *)angular_vel, count);
}

__global__ void body_data_split_for_apply_kernel(PhysxPose *__restrict__ pose_data,
                                                 Vec3 *__restrict__ vel_data,
                                                 Vec3 *__restrict__ ang_data,
                                                 PhysxBodyData *__restrict__ physx_data,
                                                 int count) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= count)
    return;
  pose_data[g] = physx_data[g].pose;
  vel_data[g] = physx_data[g].v;
  ang_data[g] = physx_data[g].w;
}

void body_data_split_for_apply(void *pose_data, void *vel_data, void *ang_data, void *physx_data,
                               int count, cudaStream_t stream) {
  body_data_split_for_apply_kernel<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0,
                                    stream>>>((PhysxPose *)pose_data, (Vec3 *)vel_data,
                                              (Vec3 *)ang_data, (PhysxBodyData *)physx_data,
                                              count);
}

__global__ void gather_rigid_dynamic_gpu_indices_kernel(
    unsigned int *__restrict__ out_indices, unsigned int *__restrict__ gpu_index_buffer,
    int *__restrict__ apply_index, int count) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= count)
    return;
  out_indices[g] = gpu_index_buffer[apply_index[g]];
}

void gather_rigid_dynamic_gpu_indices(void *out_indices, void *gpu_index_buffer, void *apply_index,
                                      int count, cudaStream_t stream) {
  gather_rigid_dynamic_gpu_indices_kernel<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0,
                                            stream>>>((unsigned int *)out_indices,
                                                      (unsigned int *)gpu_index_buffer,
                                                      (int *)apply_index, count);
}

__global__ void gather_articulation_gpu_indices_kernel(
    unsigned int *__restrict__ out_indices, unsigned int *__restrict__ gpu_index_buffer,
    int *__restrict__ apply_index, int count) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= count)
    return;
  out_indices[g] = gpu_index_buffer[apply_index[g]];
}

void gather_articulation_gpu_indices(void *out_indices, void *gpu_index_buffer, void *apply_index,
                                     int count, cudaStream_t stream) {
  gather_articulation_gpu_indices_kernel<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0,
                                          stream>>>((unsigned int *)out_indices,
                                                    (unsigned int *)gpu_index_buffer,
                                                    (int *)apply_index, count);
}

__global__ void gather_articulation_dof_data_kernel(float *__restrict__ out_data,
                                                    float *__restrict__ src_data,
                                                    int *__restrict__ apply_index, int count,
                                                    int max_dof) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= count * max_dof)
    return;
  int block = g / max_dof;
  int elem = g % max_dof;
  int src_idx = apply_index[block] * max_dof + elem;
  out_data[g] = src_data[src_idx];
}

void gather_articulation_dof_data(void *out_data, void *src_data, void *apply_index, int count,
                                  int max_dof, cudaStream_t stream) {
  gather_articulation_dof_data_kernel<<<(count * max_dof + BLOCK_SIZE - 1) / BLOCK_SIZE,
                                        BLOCK_SIZE, 0, stream>>>(
      (float *)out_data, (float *)src_data, (int *)apply_index, count, max_dof);
}

__global__ void root_vel_sapien_to_physx_kernel(Vec3 *__restrict__ linear_vel,
                                                 Vec3 *__restrict__ angular_vel,
                                                 SapienBodyData *__restrict__ sapien_data,
                                                 int *__restrict__ index, int link_count,
                                                 int count) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= count)
    return;
  int ai = index[g];
  SapienBodyData sd = sapien_data[ai * link_count];
  linear_vel[g] = sd.v;
  angular_vel[g] = sd.w;
}

void root_vel_sapien_to_physx(void *linear_vel, void *angular_vel, void *sapien_data, void *index,
                              int link_count, int count, cudaStream_t stream) {
  root_vel_sapien_to_physx_kernel<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0,
                                    stream>>>((Vec3 *)linear_vel, (Vec3 *)angular_vel,
                                              (SapienBodyData *)sapien_data, (int *)index,
                                              link_count, count);
}

void handle_contacts(::physx::PxGpuContactPair *contacts, int contact_count, ActorPairQuery *query,
                     int query_count, Vec3 *out_forces, cudaStream_t stream) {
  handle_contacts_kernel<<<(contact_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
      contacts, contact_count, query, query_count, out_forces);
}

void handle_net_contact_force(::physx::PxGpuContactPair *contacts, int contact_count,
                              ActorQuery *query, int query_count, Vec3 *out_forces,
                              cudaStream_t stream) {
  handle_net_contact_force_kernel<<<(contact_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0,
                                    stream>>>(contacts, contact_count, query, query_count,
                                              out_forces);
}

} // namespace physx
} // namespace sapien