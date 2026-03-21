## Elytar-only headless snippet benchmark target (Linux).
##
## Builds SnippetIsosurface without RENDER_SNIPPET so snippetMain() executes the
## fixed-step non-interactive path.

set(SNIPPET_NAME Isosurface)
include(${PHYSX_ROOT_DIR}/snippets/${PROJECT_CMAKE_FILES_DIR}/${TARGET_BUILD_PLATFORM}/SnippetTemplate.cmake)

# Remove interactive rendering define for this benchmark target.
list(REMOVE_ITEM SNIPPET_COMPILE_DEFS RENDER_SNIPPET)

string(TOLOWER ${SNIPPET_NAME} SNIPPET_NAME_LOWER)
file(GLOB ElytarSnippetSources ${PHYSX_ROOT_DIR}/snippets/snippet${SNIPPET_NAME_LOWER}/*.cpp)
file(GLOB ElytarSnippetHeaders ${PHYSX_ROOT_DIR}/snippets/snippet${SNIPPET_NAME_LOWER}/*.h)

add_executable(Snippet${SNIPPET_NAME}Headless ${SNIPPET_BUNDLE}
  ${SNIPPET_PLATFORM_SOURCES}
  ${ElytarSnippetSources}
  ${ElytarSnippetHeaders}
)

target_include_directories(Snippet${SNIPPET_NAME}Headless
  PRIVATE ${SNIPPET_PLATFORM_INCLUDES}
  PRIVATE ${PHYSX_ROOT_DIR}/include/
  PRIVATE ${PHYSX_ROOT_DIR}/source/physxextensions/src
)

target_compile_definitions(Snippet${SNIPPET_NAME}Headless
  PRIVATE ${SNIPPET_COMPILE_DEFS}
  PRIVATE ELYTAR_HEADLESS_SNIPPET_BENCH=1
)

set_target_properties(Snippet${SNIPPET_NAME}Headless PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PX_EXE_OUTPUT_DIRECTORY_DEBUG}${EXE_PLATFORM_DIR}
  RUNTIME_OUTPUT_DIRECTORY_PROFILE ${PX_EXE_OUTPUT_DIRECTORY_PROFILE}${EXE_PLATFORM_DIR}
  RUNTIME_OUTPUT_DIRECTORY_CHECKED ${PX_EXE_OUTPUT_DIRECTORY_CHECKED}${EXE_PLATFORM_DIR}
  RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PX_EXE_OUTPUT_DIRECTORY_RELEASE}${EXE_PLATFORM_DIR}
  OUTPUT_NAME Snippet${SNIPPET_NAME}Headless${EXE_SUFFIX}
)

if(PVDRuntimeBuilt)
  set(PVDRuntime_Lib "PVDRuntime")
else()
  set(PVDRuntime_Lib "")
endif()

target_link_libraries(Snippet${SNIPPET_NAME}Headless
  PUBLIC PhysXExtensions PhysXPvdSDK PhysX PhysXVehicle2 PhysXCharacterKinematic PhysXCooking PhysXCommon PhysXFoundation SnippetUtils ${PVDRuntime_Lib}
  PUBLIC ${SNIPPET_PLATFORM_LINKED_LIBS}
)

if(PX_GENERATE_GPU_PROJECTS OR PX_GENERATE_GPU_PROJECTS_ONLY)
  add_dependencies(Snippet${SNIPPET_NAME}Headless PhysXGpu)
endif()

set_property(TARGET Snippet${SNIPPET_NAME}Headless PROPERTY FOLDER "Snippets")

