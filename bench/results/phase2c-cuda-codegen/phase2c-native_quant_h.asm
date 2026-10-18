
/home/ubuntu/Guild/build-cuda12/ornith_prefill_bench:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

000000000048de60 <_ZN5guild7kernels3cpu14native_quant_hERK28guild_cpu_expert_format_v1_sPKfPv>:
  48de60:	endbr64
  48de64:	push   %r12
  48de66:	mov    %rdx,%r12
  48de69:	push   %rbp
  48de6a:	mov    %rsi,%rbp
  48de6d:	push   %rbx
  48de6e:	mov    %rdi,%rbx
  48de71:	mov    0xc(%rdi),%edi
  48de74:	call   4f2cb0 <ggml_get_type_traits_cpu>
  48de79:	mov    0x18(%rbx),%rdx
  48de7d:	mov    %r12,%rsi
  48de80:	pop    %rbx
  48de81:	mov    %rbp,%rdi
  48de84:	mov    (%rax),%rax
  48de87:	pop    %rbp
  48de88:	pop    %r12
  48de8a:	jmp    *%rax

Disassembly of section .fini:
