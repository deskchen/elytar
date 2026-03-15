## Copyright (c) 2025 Elytar Project
## ElytarPtxReplace.cmake
##
## Provides the ELYTAR_REPLACE_CU_WITH_PTX() macro used by all six
## PhysX GPU sub-library cmake files to replace compiled .cu device code
## with pre-generated PTX at build time.
##
## Pipeline per .cu file:
##   1. Pre-generated <stem>.ptx  (by scripts/generate_ptx.sh)
##   2. nvcc -fatbin .ptx -> <stem>.fatbin  (build time, custom command)
##   3. bin2c .fatbin -> <stem>_fatbin.h   (build time, custom command)
##   4. Auto-generated <stem>_host_stub.cpp  (configure time, file(GENERATE))
##   5. Auto-generated <stem>_ptx_register.cpp  (configure time, file(GENERATE))
##      includes <stem>_fatbin.h at compile time, registers module + kernels
##      through PhysX's PxGpuCudaRegisterFatBinary / PxGpuCudaRegisterFunction
##
## Usage in each sub-library .cmake file (BEFORE ADD_LIBRARY):
##
##   IF(PX_USE_PTX_KERNELS)
##       INCLUDE("${CMAKE_CURRENT_LIST_DIR}/ElytarPtxReplace.cmake")
##       SET(ELYTAR_PTX_EXTRA_SOURCES "")
##       SET(_ptx_dir "${MY_MODULE_SOURCE_DIR}/PTX")
##       FOREACH(CU_FILE IN LISTS MY_CUDA_KERNELS_LIST)
##           ELYTAR_REPLACE_CU_WITH_PTX(
##               KERNELS_VAR MY_CUDA_KERNELS_LIST
##               CU_FILE     "${CU_FILE}"
##               PTX_DIR     "${_ptx_dir}")
##       ENDFOREACH()
##   ENDIF()
##
##   ADD_LIBRARY(MyTarget ...
##       ${MY_CUDA_KERNELS_LIST}         # empty in PTX mode
##       ${ELYTAR_PTX_EXTRA_SOURCES}     # generated stubs in PTX mode
##   )
##
##   IF(PX_USE_PTX_KERNELS)
##       TARGET_INCLUDE_DIRECTORIES(MyTarget PRIVATE
##           ${CMAKE_CURRENT_BINARY_DIR}/elytar_ptx)
##   ENDIF()

cmake_minimum_required(VERSION 3.16)
include_guard(GLOBAL)

# ---- Locate bin2c (ships with CUDA toolkit) --------------------------------
if(NOT DEFINED BIN2C_EXECUTABLE)
    find_program(BIN2C_EXECUTABLE bin2c
        HINTS "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../bin"
              "${CUDAToolkit_BIN_DIR}"
    )
    if(NOT BIN2C_EXECUTABLE)
        message(FATAL_ERROR
            "[Elytar] bin2c not found. It ships with the CUDA toolkit.\n"
            "         Expected at: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../bin/bin2c")
    endif()
endif()

# ---- Derive sm_XX from PX_PTX_ARCH (compute_XX) ----------------------------
if(NOT DEFINED ELYTAR_PTX_SM_ARCH)
    string(REPLACE "compute_" "sm_" ELYTAR_PTX_SM_ARCH "${PX_PTX_ARCH}")
    message(STATUS "[Elytar] PTX arch: ${PX_PTX_ARCH} -> fatbin arch: ${ELYTAR_PTX_SM_ARCH}")
endif()

# ---------------------------------------------------------------------------
# ELYTAR_REPLACE_CU_WITH_PTX(
#     KERNELS_VAR <list-variable-name>
#     CU_FILE     <absolute-path-to-.cu>
#     PTX_DIR     <directory-containing-.ptx-files>
# )
#
# Effect on calling scope:
#   - Removes CU_FILE from ${KERNELS_VAR}
#   - Appends generated stub paths to ELYTAR_PTX_EXTRA_SOURCES
# ---------------------------------------------------------------------------
macro(ELYTAR_REPLACE_CU_WITH_PTX)
    cmake_parse_arguments(_EPTX "" "KERNELS_VAR;CU_FILE;PTX_DIR" "" ${ARGN})

    get_filename_component(_eptx_stem "${_EPTX_CU_FILE}" NAME_WE)

    set(_eptx_ptx_file    "${_EPTX_PTX_DIR}/${_eptx_stem}.ptx")
    set(_eptx_gen_dir     "${CMAKE_CURRENT_BINARY_DIR}/elytar_ptx")
    set(_eptx_fatbin_file "${_eptx_gen_dir}/${_eptx_stem}.fatbin")
    set(_eptx_fatbin_hdr  "${_eptx_gen_dir}/${_eptx_stem}_fatbin.h")
    set(_eptx_stub_file   "${_eptx_gen_dir}/${_eptx_stem}_host_stub.cpp")
    set(_eptx_reg_file    "${_eptx_gen_dir}/${_eptx_stem}_ptx_register.cpp")

    # ---- Validate PTX file -----------------------------------------------
    if(NOT EXISTS "${_eptx_ptx_file}")
        message(FATAL_ERROR
            "[Elytar] PTX file not found: ${_eptx_ptx_file}\n"
            "         Run: scripts/generate_ptx.sh --all\n"
            "         (or: scripts/generate_ptx.sh --cu ${_EPTX_CU_FILE})")
    endif()

    # ---- Parse .cu for host init fn name (CPU-side, not in PTX) ------------
    file(STRINGS "${_EPTX_CU_FILE}" _eptx_cu_lines)
    set(_eptx_host_fn "")
    foreach(_eptx_line IN LISTS _eptx_cu_lines)
        if(_eptx_line MATCHES "extern \"C\" __host__ void ([A-Za-z0-9_]+)\\(\\)")
            set(_eptx_host_fn "${CMAKE_MATCH_1}")
            break()
        endif()
    endforeach()

    # ---- Read kernel entry names directly from the PTX file ----------------
    # This is authoritative: the .entry names in PTX are exactly the strings
    # cuModuleGetFunction / PxGpuCudaRegisterFunction must use — regardless of
    # whether the original kernel was extern "C" (plain name) or C++ (mangled).
    file(STRINGS "${_eptx_ptx_file}" _eptx_ptx_lines)
    set(_eptx_kernel_names "")
    foreach(_eptx_line IN LISTS _eptx_ptx_lines)
        # PTX format:  .entry <name>(  or  .entry <name> (
        if(_eptx_line MATCHES "\\.entry[ \t]+([A-Za-z0-9_$]+)")
            list(APPEND _eptx_kernel_names "${CMAKE_MATCH_1}")
        endif()
    endforeach()

    list(REMOVE_DUPLICATES _eptx_kernel_names)
    list(LENGTH _eptx_kernel_names _eptx_nkernels)
    if(_eptx_nkernels EQUAL 0)
        message(WARNING
            "[Elytar] No .entry found in ${_eptx_ptx_file}.\n"
            "         Did generate_ptx.sh run successfully for this file?")
    endif()

    # ---- Generate host stub .cpp ------------------------------------------
    file(MAKE_DIRECTORY "${_eptx_gen_dir}")

    if(_eptx_host_fn)
        string(CONCAT _eptx_stub_content
            "// Auto-generated by ElytarPtxReplace.cmake -- do not edit\n"
            "// Host stub for ${_eptx_stem}.cu (PTX mode)\n"
            "// Provides the empty host init function that PxgSolver.cpp\n"
            "// calls to force static-library linkage.\n"
            "extern \"C\" void ${_eptx_host_fn}() {}\n"
        )
    else()
        string(CONCAT _eptx_stub_content
            "// Auto-generated by ElytarPtxReplace.cmake -- do not edit\n"
            "// No host init function found in ${_eptx_stem}.cu\n"
        )
    endif()

    file(GENERATE OUTPUT "${_eptx_stub_file}" CONTENT "${_eptx_stub_content}")

    # ---- Build kernel registration calls string ----------------------------
    set(_eptx_reg_calls "")
    foreach(_eptx_kname IN LISTS _eptx_kernel_names)
        string(APPEND _eptx_reg_calls
            "        PxGpuCudaRegisterFunction(moduleIndex, \"${_eptx_kname}\");\n")
    endforeach()

    # ---- Generate ptx_register.cpp ----------------------------------------
    # If no kernels: still register the fatbin (so the module loads) but suppress
    # the unused-variable warning with (void)moduleIndex — some .cu files expose
    # only __device__ helpers with no directly-launched __global__ entry points.
    if(_eptx_nkernels EQUAL 0)
        set(_eptx_module_index_use "        (void)moduleIndex; // no __global__ entries in this module\n")
    else()
        set(_eptx_module_index_use "")
    endif()

    string(CONCAT _eptx_reg_content
        "// Auto-generated by ElytarPtxReplace.cmake -- do not edit\n"
        "// Registers the pre-generated PTX fatbin for ${_eptx_stem}.cu\n"
        "// with PhysX's CUDA module system (kernel wrangler).\n"
        "//\n"
        "// At program startup (static init), this registers:\n"
        "//   - The fatbin image via PxGpuCudaRegisterFatBinary\n"
        "//   - Each __global__ kernel name via PxGpuCudaRegisterFunction\n"
        "// Kernel names are read from the .entry lines in the PTX file,\n"
        "// so both plain (extern C) and C++-mangled names are handled correctly.\n"
        "// CudaContextManager then calls cuModuleLoadDataEx + cuModuleGetFunction\n"
        "// to resolve CUfunction handles, exactly as in the original .cu path.\n"
        "#include \"${_eptx_stem}_fatbin.h\"\n"
        "#include <cstddef>\n"
        "\n"
        "extern \"C\" void  PxGpuCudaRegisterFunction(int moduleIndex, const char* functionName);\n"
        "extern \"C\" void** PxGpuCudaRegisterFatBinary(void* fatBin);\n"
        "\n"
        "namespace {\n"
        "struct Elytar_${_eptx_stem}_Registrar {\n"
        "    Elytar_${_eptx_stem}_Registrar() {\n"
        "        void** handle = PxGpuCudaRegisterFatBinary(\n"
        "            static_cast<void*>(${_eptx_stem}_fatbin));\n"
        "        int moduleIndex = static_cast<int>(\n"
        "            reinterpret_cast<std::size_t>(handle));\n"
        "${_eptx_module_index_use}"
        "${_eptx_reg_calls}"
        "    }\n"
        "};\n"
        "static Elytar_${_eptx_stem}_Registrar s_${_eptx_stem}_registrar;\n"
        "} // namespace\n"
    )

    file(GENERATE OUTPUT "${_eptx_reg_file}" CONTENT "${_eptx_reg_content}")

    # ---- Custom command: .ptx -> .fatbin -----------------------------------
    add_custom_command(
        OUTPUT  "${_eptx_fatbin_file}"
        COMMAND "${CMAKE_CUDA_COMPILER}" -fatbin
                "${_eptx_ptx_file}"
                -o "${_eptx_fatbin_file}"
                -arch=${ELYTAR_PTX_SM_ARCH}
        DEPENDS "${_eptx_ptx_file}"
        COMMENT "[Elytar] ${_eptx_stem}.ptx -> fatbin (${ELYTAR_PTX_SM_ARCH})"
        VERBATIM
    )

    # ---- Custom command: .fatbin -> _fatbin.h (C byte array) ---------------
    add_custom_command(
        OUTPUT  "${_eptx_fatbin_hdr}"
        COMMAND "${BIN2C_EXECUTABLE}"
                --name ${_eptx_stem}_fatbin
                "${_eptx_fatbin_file}"
                > "${_eptx_fatbin_hdr}"
        DEPENDS "${_eptx_fatbin_file}"
        COMMENT "[Elytar] ${_eptx_stem}.fatbin -> C header"
        VERBATIM
    )

    # Ensure ptx_register.cpp is not compiled until the fatbin header exists
    set_source_files_properties("${_eptx_reg_file}" PROPERTIES
        OBJECT_DEPENDS "${_eptx_fatbin_hdr}")

    # ---- Update caller-scope variables -------------------------------------
    # Remove the .cu from the kernel list (macro shares caller scope)
    list(REMOVE_ITEM ${_EPTX_KERNELS_VAR} "${_EPTX_CU_FILE}")
    # Append generated sources (caller must have initialized ELYTAR_PTX_EXTRA_SOURCES)
    list(APPEND ELYTAR_PTX_EXTRA_SOURCES "${_eptx_stub_file}" "${_eptx_reg_file}")

    message(STATUS "[Elytar]   ${_eptx_stem}.cu -> PTX (${_eptx_nkernels} kernels)")

endmacro()
