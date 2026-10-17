
/home/ubuntu/guild-kernels-cpu/build/libguild_cpu_plugin.so:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

000000000001e680 <_ZN5guild7kernels3cpu16native_quant_actERK28guild_cpu_expert_format_v1_sPKfPv>:
   1e680:	endbr64
   1e684:	push   %r12
   1e686:	mov    %rdx,%r12
   1e689:	push   %rbp
   1e68a:	mov    %rsi,%rbp
   1e68d:	push   %rbx
   1e68e:	mov    %rdi,%rbx
   1e691:	mov    0x8(%rdi),%edi
   1e694:	call   9e6d0 <ggml_get_type_traits_cpu>
   1e699:	mov    0x10(%rbx),%rdx
   1e69d:	mov    %r12,%rsi
   1e6a0:	pop    %rbx
   1e6a1:	mov    %rbp,%rdi
   1e6a4:	mov    (%rax),%rax
   1e6a7:	pop    %rbp
   1e6a8:	pop    %r12
   1e6aa:	jmp    *%rax

Disassembly of section .fini:
