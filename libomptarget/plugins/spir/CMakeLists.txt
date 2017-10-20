if(LIBOMPTARGET_DEP_OPENCL_FOUND)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)$" AND CMAKE_SYSTEM_NAME MATCHES "Linux")

        libomptarget_say("Building SPIR-V offloading plugin.")

        # Define the suffix for the runtime messaging dumps.
        add_definitions(-DTARGET_NAME=SPIR)

        if(LIBOMPTARGET_CMAKE_BUILD_TYPE MATCHES debug)
            add_definitions(-DSPIR_ERROR_REPORT)
        endif()

        include_directories(${LIBOMPTARGET_DEP_LIBELF_INCLUDE_DIRS})

        add_library(omptarget.rtl.spir SHARED src/rtl.cpp)

        # Install plugin under the lib destination folder.
        install(TARGETS omptarget.rtl.spir LIBRARY DESTINATION lib${LIBOMPTARGET_LIBDIR_SUFFIX})
        target_link_libraries(
                omptarget.rtl.spir
                ${LIBOMPTARGET_DEP_OPENCL_LIBRARIES}
                OpenCL
                "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../exports")

        # Report to the parent scope that we are building a plugin for SPIR-V.
        set(LIBOMPTARGET_SYSTEM_TARGETS "${LIBOMPTARGET_SYSTEM_TARGETS} spir64-unknown-unknown" PARENT_SCOPE)
    else()
        libomptarget_say("Not building SPIR-V offloading plugin: only support SPIR-V in Linux x86_64 hosts.")
    endif()
else()
    libomptarget_say("Not building SPIR-V offloading plugin: OpenCL 2.1+ not found in system.")
endif()