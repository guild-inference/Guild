#pragma once
#include "guild/abi/module_abi.h"
#include <string>

namespace guild::kernels {
guild_status_t validate_operation_query(const guild_operation_query_v1_t&, std::string&);
guild_status_t copy_operation_extension(const guild_kernel_op_desc_v1_t&,
                                       guild_operation_ext_v1_t&, std::string&);
guild_status_t match_operation(const guild_kernel_op_desc_v1_t&,
                              const guild_operation_query_v1_t&,
                              const guild_device_capabilities_v1_t&,
                              guild_scratch_request_t&, std::string&);
}
