#include <guild/abi/cpu_backend_abi.h>
#include <cstdio>
#include <cstdlib>
#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#define CHECK(x) do{if(!(x)){std::fprintf(stderr,"SDK check failed at line %d: %s\n",__LINE__,#x);std::abort();}}while(0)
static guild_kernel_execute_fn execute=nullptr;
int main(int argc,char** argv){
    CHECK(argc==2);
    static_assert(sizeof(guild_cpu_planar_act_v1_t)==3584,"C/C++ layout");
    static_assert(offsetof(guild_cpu_backend_ext_v2_t,native)==0,"frozen prefix");
#if defined(_WIN32)
    auto handle=LoadLibraryA(argv[1]);CHECK(handle);
    auto entry=reinterpret_cast<guild_plugin_entry_fn>(GetProcAddress(handle,GUILD_PLUGIN_ENTRY_SYMBOL_V1));
#else
    auto handle=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);CHECK(handle);
    auto entry=reinterpret_cast<guild_plugin_entry_fn>(dlsym(handle,GUILD_PLUGIN_ENTRY_SYMBOL_V1));
#endif
    CHECK(entry);
    guild_plugin_descriptor_v1_t plugin{};guild_error_info_t error{};
    CHECK(entry(GUILD_KERNEL_ABI_VERSION_V1,&plugin,&error)==GUILD_STATUS_OK);
    guild_host_api_v1_t api{};api.struct_size=sizeof(api);
    api.register_kernel_op=+[](guild_core_host_t,const guild_kernel_op_desc_v1_t* op,guild_error_info_t*)->guild_status_t {
        CHECK(op->struct_size==sizeof(*op));execute=op->execute;return GUILD_STATUS_OK;
    };
    CHECK(plugin.init(nullptr,&api,&error)==GUILD_STATUS_OK && execute);
    float values[]={1,2,3,4},result=0;
    guild_tensor_desc_t input{},output{};input.struct_size=sizeof(input);output.struct_size=sizeof(output);
    input.dtype=output.dtype=GUILD_DTYPE_F32;input.n_dims=output.n_dims=1;input.dims[0]=4;output.dims[0]=1;
    input.buffer.data=values;output.buffer.data=&result;
    CHECK(execute(nullptr,&input,1,&output,1,nullptr,0,&error)==GUILD_STATUS_OK && result==10);
    CHECK(entry(GUILD_MAKE_ABI_VERSION(2,0,0),&plugin,&error)==GUILD_STATUS_ERR_ABI_MISMATCH);
#if defined(_WIN32)
    CHECK(FreeLibrary(handle));
#else
    CHECK(dlclose(handle)==0);
#endif
}
