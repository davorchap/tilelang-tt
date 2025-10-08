# FindMetalium.cmake
#
# Finds the TT-Metalium SDK for Tenstorrent hardware programming
#
# This module defines:
#  Metalium_FOUND         - True if Metalium SDK is found
#  Metalium_INCLUDE_DIRS  - Include directories for Metalium headers
#  Metalium_LIBRARIES     - Libraries to link against
#  Metalium_VERSION       - Version of Metalium SDK (if available)
#  TT_METAL_HOME          - Root directory of Metalium SDK
#
# Usage:
#   find_package(Metalium)
#   if(Metalium_FOUND)
#       target_include_directories(mytarget PRIVATE ${Metalium_INCLUDE_DIRS})
#       target_link_libraries(mytarget PRIVATE ${Metalium_LIBRARIES})
#   endif()
#
# Environment Variables:
#   TT_METAL_HOME - Root directory of TT-Metalium SDK installation

# Check for TT_METAL_HOME environment variable
if(NOT DEFINED ENV{TT_METAL_HOME})
    if(NOT Metalium_FIND_QUIETLY)
        message(STATUS "TT_METAL_HOME environment variable not set")
    endif()
    set(Metalium_FOUND FALSE)
    return()
endif()

set(TT_METAL_HOME $ENV{TT_METAL_HOME})

if(NOT Metalium_FIND_QUIETLY)
    message(STATUS "Found TT_METAL_HOME: ${TT_METAL_HOME}")
endif()

# Find include directories
find_path(Metalium_INCLUDE_DIR
    NAMES tt_metal/host_api.hpp
    PATHS ${TT_METAL_HOME}
    PATH_SUFFIXES "" "tt_metal"
    NO_DEFAULT_PATH
)

# Find additional include paths
set(Metalium_INCLUDE_DIRS
    ${TT_METAL_HOME}
    ${TT_METAL_HOME}/tt_metal
    ${TT_METAL_HOME}/tt_metal/impl
)

# Find libraries
find_library(Metalium_LIBRARY_TT_METAL
    NAMES tt_metal
    PATHS ${TT_METAL_HOME}
    PATH_SUFFIXES build/lib lib
    NO_DEFAULT_PATH
)

find_library(Metalium_LIBRARY_DEVICE
    NAMES device
    PATHS ${TT_METAL_HOME}
    PATH_SUFFIXES build/lib lib
    NO_DEFAULT_PATH
)

# Collect all libraries
set(Metalium_LIBRARIES)
if(Metalium_LIBRARY_TT_METAL)
    list(APPEND Metalium_LIBRARIES ${Metalium_LIBRARY_TT_METAL})
endif()
if(Metalium_LIBRARY_DEVICE)
    list(APPEND Metalium_LIBRARIES ${Metalium_LIBRARY_DEVICE})
endif()

# Try to find version from VERSION file or git describe
set(Metalium_VERSION "unknown")
if(EXISTS "${TT_METAL_HOME}/VERSION")
    file(READ "${TT_METAL_HOME}/VERSION" Metalium_VERSION)
    string(STRIP "${Metalium_VERSION}" Metalium_VERSION)
endif()

# Handle standard find_package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Metalium
    FOUND_VAR Metalium_FOUND
    REQUIRED_VARS
        TT_METAL_HOME
        Metalium_INCLUDE_DIR
        Metalium_LIBRARY_TT_METAL
    VERSION_VAR Metalium_VERSION
)

# Mark cache variables as advanced
mark_as_advanced(
    Metalium_INCLUDE_DIR
    Metalium_LIBRARY_TT_METAL
    Metalium_LIBRARY_DEVICE
)

# Create imported target TT::Metalium (matches official package)
if(Metalium_FOUND AND NOT TARGET TT::Metalium)
    add_library(TT::Metalium INTERFACE IMPORTED)
    set_target_properties(TT::Metalium PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${Metalium_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${Metalium_LIBRARIES}"
    )

    if(NOT Metalium_FIND_QUIETLY)
        message(STATUS "Created TT::Metalium imported target")
    endif()
endif()

# Print summary if found
if(Metalium_FOUND AND NOT Metalium_FIND_QUIETLY)
    message(STATUS "Metalium SDK Configuration:")
    message(STATUS "  Version: ${Metalium_VERSION}")
    message(STATUS "  Include dirs: ${Metalium_INCLUDE_DIRS}")
    message(STATUS "  Libraries: ${Metalium_LIBRARIES}")
    message(STATUS "  Target: TT::Metalium")
endif()
