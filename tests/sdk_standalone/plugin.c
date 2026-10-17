#include <guild/abi/kernel_abi.h>
#include <guild/abi/runner_abi.h>
#include <guild/abi/cpu_backend_abi.h>
#include <guild/abi/cuda_dense_abi.h>
#include <guild/abi/cuda_attn_abi.h>
typedef char frozen_image_size[(sizeof(guild_cpu_planar_act_v1_t)==3584)?1:-1];
typedef char cuda_extension_prefix[(offsetof(guild_cuda_dense_ext_v1_t,tag)==4)?1:-1];
typedef char cuda_attn_extension_prefix[(offsetof(guild_cuda_attn_ext_v1_t,tag)==4)?1:-1];
static guild_status_t execute(const guild_exec_context_v1_t* context,const guild_tensor_desc_t* in,size_t ni,
    const guild_tensor_desc_t* out,size_t no,const void* params,size_t size,guild_error_info_t* error) {
    size_t i;float sum=0;
    (void)context;(void)params;(void)size;(void)error;
    if(!in || !out || ni!=1 || no!=1 || in[0].dtype!=GUILD_DTYPE_F32 || out[0].dtype!=GUILD_DTYPE_F32 ||
       in[0].n_dims!=1 || in[0].dims[0]!=4 || !in[0].buffer.data || !out[0].buffer.data)
        return GUILD_STATUS_ERR_INVALID_TENSOR;
    for(i=0;i<4;++i)sum+=((const float*)in[0].buffer.data)[i];
    ((float*)out[0].buffer.data)[0]=sum;
    return GUILD_STATUS_OK;
}
static guild_status_t initialize(guild_core_host_t host,const guild_host_api_v1_t* api,guild_error_info_t* error) {
    guild_kernel_op_desc_v1_t op={0};
    if(!api || api->struct_size<sizeof(*api))return GUILD_STATUS_ERR_ABI_MISMATCH;
    op.struct_size=sizeof(op);op.abi_version=GUILD_KERNEL_ABI_VERSION_V1;
    op.op_kind=GUILD_OP_FINITE_CHECK;op.op_name="sdk_contract";op.impl_id="sdk.c.contract";op.package_id="sdk.contract";
    op.backend=GUILD_BACKEND_CPU;op.execute=execute;
    return api->register_kernel_op(host,&op,error);
}
GUILD_PLUGIN_EXPORT guild_status_t guild_plugin_entry_v1(uint32_t abi,guild_plugin_descriptor_v1_t* out,guild_error_info_t* error) {
    (void)error;
    if(!out || GUILD_ABI_VERSION_MAJOR(abi)!=1)return GUILD_STATUS_ERR_ABI_MISMATCH;
    out->struct_size=sizeof(*out);out->abi_version=GUILD_KERNEL_ABI_VERSION_V1;
    out->package_id="sdk.contract";out->package_version="1.0.0";out->publisher_id="test.sdk";
    out->init=initialize;out->shutdown=0;
    return GUILD_STATUS_OK;
}
