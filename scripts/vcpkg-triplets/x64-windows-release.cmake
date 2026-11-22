# Distribution dependencies: never build a second Debug configuration.
set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE dynamic)
set(VCPKG_BUILD_TYPE release)

# Protobuf 5.29.0 generates constant initializers that reference symbols
# from libprotobuf. clang-cl rejects them when those symbols are DLL imports.
if(PORT STREQUAL "protobuf")
    set(VCPKG_LIBRARY_LINKAGE static)
endif()
