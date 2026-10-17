
/home/ubuntu/guild-kernels-cpu/build/libguild_cpu_plugin.so:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

000000000001e6b0 <_ZN5guild7kernels3cpu14native_quant_hERK28guild_cpu_expert_format_v1_sPKfPv>:
   1e6b0:	endbr64
   1e6b4:	push   %r12
   1e6b6:	mov    %rdx,%r12
   1e6b9:	push   %rbp
   1e6ba:	mov    %rsi,%rbp
   1e6bd:	push   %rbx
   1e6be:	mov    %rdi,%rbx
   1e6c1:	mov    0xc(%rdi),%edi
   1e6c4:	call   9e6d0 <ggml_get_type_traits_cpu>
   1e6c9:	mov    0x18(%rbx),%rdx
   1e6cd:	mov    %r12,%rsi
   1e6d0:	pop    %rbx
   1e6d1:	mov    %rbp,%rdi
   1e6d4:	mov    (%rax),%rax
   1e6d7:	pop    %rbp
   1e6d8:	pop    %r12
   1e6da:	jmp    *%rax

Disassembly of section .fini:
