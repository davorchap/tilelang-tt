# TTMetal.cmake
#
# Configures the TileLang Tenstorrent backend with TT-Metalium SDK support
#
# This module provides:
#  - TL_TT_BACKEND option to enable/disable the TT backend
#  - USE_REAL_METALIUM option to link against real TT-Metalium SDK vs mock mode
#  - Conditional compilation definitions for real vs mock builds
#
# Build modes:
#  - Mock mode (default): USE_REAL_METALIUM=OFF
#    - Emits code with mock/stub TT APIs for dry-run testing
#    - No TT_METAL_HOME required
#    - All tests pass in CI without hardware
#
#  - Real mode: USE_REAL_METALIUM=ON
#    - Links against actual TT-Metalium SDK
#    - Requires TT_METAL_HOME environment variable
#    - Generated code uses real Metalium APIs (CreateKernelFromString, etc.)

option(TL_TT_BACKEND "Build TileLang Tenstorrent backend" ON)
option(USE_REAL_METALIUM "Link against real TT-Metalium SDK (requires TT_METAL_HOME)" OFF)

if(TL_TT_BACKEND)
    message(STATUS "TileLang Tenstorrent backend: ENABLED")

    # Define TL_TT_BACKEND for conditional compilation
    add_compile_definitions(TL_TT_BACKEND=1)

    if(USE_REAL_METALIUM)
        message(STATUS "TT-Metalium mode: REAL (linking SDK)")

        # Find the TT-Metalium SDK
        find_package(Metalium QUIET)

        if(NOT Metalium_FOUND)
            message(FATAL_ERROR
                "USE_REAL_METALIUM=ON but TT-Metalium SDK not found. "
                "Please set TT_METAL_HOME environment variable to point to tt-metal installation, "
                "or set USE_REAL_METALIUM=OFF for mock mode."
            )
        endif()

        # Define USE_REAL_METALIUM for conditional compilation
        add_compile_definitions(USE_REAL_METALIUM=1)

        message(STATUS "TT-Metalium SDK found:")
        message(STATUS "  TT_METAL_HOME: ${TT_METAL_HOME}")
        message(STATUS "  Include dirs: ${Metalium_INCLUDE_DIRS}")
        message(STATUS "  Libraries: ${Metalium_LIBRARIES}")

    else()
        message(STATUS "TT-Metalium mode: MOCK (dry-run testing)")
        message(STATUS "  Generated code will use mock TT APIs")
        message(STATUS "  To use real SDK, set USE_REAL_METALIUM=ON and TT_METAL_HOME")
    endif()

else()
    message(STATUS "TileLang Tenstorrent backend: DISABLED")
endif()
