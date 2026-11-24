# Distribution dependencies: never build a second Debug configuration.
set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE dynamic)
set(VCPKG_BUILD_TYPE release)

# Protobuf 5.29.0 generates constant initializers that reference symbols
# from libprotobuf. clang-cl rejects them when those symbols are DLL imports.
if(PORT STREQUAL "protobuf")
    set(VCPKG_LIBRARY_LINKAGE static)
    # Protobuf's generated messages and runtime must agree on which
    # compiler-assisted message constructors are available.
    list(APPEND VCPKG_ENV_PASSTHROUGH RAWPREAMP_VCPKG_PROTOBUF_CLANG_CL)
    if(DEFINED ENV{RAWPREAMP_VCPKG_PROTOBUF_CLANG_CL})
        file(TO_CMAKE_PATH "$ENV{RAWPREAMP_VCPKG_PROTOBUF_CLANG_CL}" _rawpreamp_clang_cl)
        list(APPEND VCPKG_HASH_ADDITIONAL_FILES "${_rawpreamp_clang_cl}")
        # Protobuf's generated version.rc has no includes. Run rc.exe directly
        # so cmcldeps does not forward rc.exe's /c65001 option to clang-cl.
        list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS
            "-DCMAKE_C_COMPILER:FILEPATH=${_rawpreamp_clang_cl}"
            "-DCMAKE_CXX_COMPILER:FILEPATH=${_rawpreamp_clang_cl}"
            "-DCMAKE_NINJA_CMCLDEPS_RC:BOOL=OFF")
    endif()
endif()
