if(TARGET physx5)
  return()
endif()

set(PHYSX_VERSION 105.1-physx-5.3.1.patch0)

if (IS_DIRECTORY ${SAPIEN_PHYSX5_DIR})
  set(physx5_SOURCE_DIR ${SAPIEN_PHYSX5_DIR})
else()
  message(FATAL_ERROR
    "SAPIEN_PHYSX5_DIR is not set or does not point to a valid directory.\n"
    "  SAPIEN_PHYSX5_DIR='${SAPIEN_PHYSX5_DIR}'\n"
    "Set SAPIEN_PHYSX5_DIR to the local PhysX source/install directory "
    "(must contain include/ and bin/ subdirectories).")
endif()

add_library(physx5 INTERFACE)

if (APPLE)
  if(CMAKE_SYSTEM_NAME MATCHES ".*Darwin.*" OR CMAKE_SYSTEM_NAME MATCHES ".*MacOS.*")
    target_link_directories(physx5 INTERFACE $<BUILD_INTERFACE:${physx5_SOURCE_DIR}/bin/universal/release>)
  endif()
  
  target_link_libraries(physx5 INTERFACE
    libPhysXCharacterKinematic_static_64.a libPhysXCommon_static_64.a
    libPhysXCooking_static_64.a libPhysXExtensions_static_64.a
    libPhysXFoundation_static_64.a libPhysXPvdSDK_static_64.a
    libPhysX_static_64.a libPhysXVehicle_static_64.a
    )
  target_include_directories(physx5 SYSTEM INTERFACE $<BUILD_INTERFACE:${physx5_SOURCE_DIR}/include>)
elseif(UNIX)

  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_link_directories(physx5 INTERFACE $<BUILD_INTERFACE:${physx5_SOURCE_DIR}/bin/linux.aarch64/release>)
  else()
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
      set(_physx_config_search_order checked debug profile release)
    else()
      set(_physx_config_search_order release profile checked)
    endif()

    # Local PhysX source builds output to linux.clang (5.3) or linux.x86_64 (5.4+).
    # Verify actual library presence, not just directory existence.
    set(_physx_lib_dir "")
    set(_physx_probe_lib "libPhysX_static_64.a")
    foreach(_cfg ${_physx_config_search_order})
      foreach(_arch linux.x86_64 linux.clang)
        if (NOT _physx_lib_dir)
          set(_candidate "${physx5_SOURCE_DIR}/bin/${_arch}/${_cfg}")
          if (EXISTS "${_candidate}/${_physx_probe_lib}")
            set(_physx_lib_dir "${_candidate}")
          endif()
        endif()
      endforeach()
    endforeach()

    if (_physx_lib_dir)
      target_link_directories(physx5 INTERFACE $<BUILD_INTERFACE:${_physx_lib_dir}>)
    else()
      message(FATAL_ERROR
        "Unable to locate PhysX libraries under ${physx5_SOURCE_DIR}/bin/. "
        "Searched configs: ${_physx_config_search_order}")
    endif()
  endif()

  target_link_libraries(physx5 INTERFACE
    -Wl,--start-group
    libPhysXCharacterKinematic_static_64.a libPhysXCommon_static_64.a
    libPhysXCooking_static_64.a libPhysXExtensions_static_64.a
    libPhysXFoundation_static_64.a libPhysXPvdSDK_static_64.a
    libPhysX_static_64.a libPhysXVehicle_static_64.a
    -Wl,--end-group)
  target_include_directories(physx5 SYSTEM INTERFACE $<BUILD_INTERFACE:${physx5_SOURCE_DIR}/include>)
endif()

if (WIN32)
  target_include_directories(physx5 SYSTEM INTERFACE $<BUILD_INTERFACE:${physx5_SOURCE_DIR}/include>)
  target_link_directories(physx5 INTERFACE $<BUILD_INTERFACE:${physx5_SOURCE_DIR}/bin/win.x86_64.vc143.mt/release>)
  target_link_libraries(physx5 INTERFACE
    PhysXVehicle2_static_64.lib PhysXExtensions_static_64.lib
    PhysXVehicle_static_64.lib PhysX_static_64.lib PhysXPvdSDK_static_64.lib
    PhysXCooking_static_64.lib PhysXCommon_static_64.lib
    PhysXCharacterKinematic_static_64.lib PhysXFoundation_static_64.lib)
endif()

target_compile_definitions(physx5 INTERFACE PX_PHYSX_STATIC_LIB)
target_compile_definitions(physx5 INTERFACE PHYSX_VERSION="${PHYSX_VERSION}")
