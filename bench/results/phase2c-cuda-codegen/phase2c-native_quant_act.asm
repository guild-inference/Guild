
/home/ubuntu/Guild/build-cuda12/ornith_prefill_bench:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

000000000048de30 <_ZN5guild7kernels3cpu16native_quant_actERK28guild_cpu_expert_format_v1_sPKfPv>:
  48de30:	endbr64
  48de34:	push   %r12
  48de36:	mov    %rdx,%r12
  48de39:	push   %rbp
  48de3a:	mov    %rsi,%rbp
  48de3d:	push   %rbx
  48de3e:	mov    %rdi,%rbx
  48de41:	mov    0x8(%rdi),%edi
  48de44:	call   4f2cb0 <ggml_get_type_traits_cpu>
  48de49:	mov    0x10(%rbx),%rdx
  48de4d:	mov    %r12,%rsi
  48de50:	pop    %rbx
  48de51:	mov    %rbp,%rdi
  48de54:	mov    (%rax),%rax
  48de57:	pop    %rbp
  48de58:	pop    %r12
  48de5a:	jmp    *%rax

Disassembly of section .fini:
