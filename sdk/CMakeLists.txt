cmake_minimum_required(VERSION 3.24)
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/VERSION" GUILD_KERNEL_SDK_VERSION LIMIT_COUNT 1)
project(GuildKernelSDK VERSION ${GUILD_KERNEL_SDK_VERSION} LANGUAGES NONE)
include(CMakePackageConfigHelpers)
configure_package_config_file(../cmake/GuildKernelSDKConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/GuildKernelSDKConfig.cmake
  INSTALL_DESTINATION lib/cmake/GuildKernelSDK)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/GuildKernelSDKConfigVersion.cmake
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES ../include/guild/abi/kernel_abi.h ../include/guild/abi/runner_abi.h
  ../include/guild/abi/module_abi.h
  ../include/guild/abi/cpu_expert_abi.h ../include/guild/abi/cpu_backend_abi.h
  ../include/guild/abi/cuda_dense_abi.h ../include/guild/abi/cuda_attn_abi.h
  ../include/guild/abi/cuda_moe_abi.h DESTINATION include/guild/abi)
install(FILES ../schemas/gkmetadata.v1.schema.json DESTINATION share/guild/schemas)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/GuildKernelSDKConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/GuildKernelSDKConfigVersion.cmake DESTINATION lib/cmake/GuildKernelSDK)
set(CPACK_GENERATOR TGZ)
set(CPACK_PACKAGE_NAME guild-kernel-sdk)
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
include(CPack)
