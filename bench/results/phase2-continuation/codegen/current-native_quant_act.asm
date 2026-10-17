
build-cuda12/ornith_prefill_bench:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

0000000000475880 <_ZN5guild7kernels3cpu16native_quant_actERK28guild_cpu_expert_format_v1_sPKfPv>:
  475880:	endbr64
  475884:	push   %r12
  475886:	mov    %rdx,%r12
  475889:	push   %rbp
  47588a:	mov    %rsi,%rbp
  47588d:	push   %rbx
  47588e:	mov    %rdi,%rbx
  475891:	mov    0x8(%rdi),%edi
  475894:	call   4d9ed0 <ggml_get_type_traits_cpu>
  475899:	mov    0x10(%rbx),%rdx
  47589d:	mov    %r12,%rsi
  4758a0:	pop    %rbx
  4758a1:	mov    %rbp,%rdi
  4758a4:	mov    (%rax),%rax
  4758a7:	pop    %rbp
  4758a8:	pop    %r12
  4758aa:	jmp    *%rax

Disassembly of section .fini:
