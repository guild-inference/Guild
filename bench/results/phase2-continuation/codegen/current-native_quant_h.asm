
build-cuda12/ornith_prefill_bench:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

00000000004758b0 <_ZN5guild7kernels3cpu14native_quant_hERK28guild_cpu_expert_format_v1_sPKfPv>:
  4758b0:	endbr64
  4758b4:	push   %r12
  4758b6:	mov    %rdx,%r12
  4758b9:	push   %rbp
  4758ba:	mov    %rsi,%rbp
  4758bd:	push   %rbx
  4758be:	mov    %rdi,%rbx
  4758c1:	mov    0xc(%rdi),%edi
  4758c4:	call   4d9ed0 <ggml_get_type_traits_cpu>
  4758c9:	mov    0x18(%rbx),%rdx
  4758cd:	mov    %r12,%rsi
  4758d0:	pop    %rbx
  4758d1:	mov    %rbp,%rdi
  4758d4:	mov    (%rax),%rax
  4758d7:	pop    %rbp
  4758d8:	pop    %r12
  4758da:	jmp    *%rax

Disassembly of section .fini:
