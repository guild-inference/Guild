
bench/results/external-kernels/phase1/ornith_prefill_bench:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

0000000000474350 <_ZN5guild7kernels3cpu14native_quant_hERKNS1_9NativeFmtEPKfPv>:
  474350:	endbr64
  474354:	push   %r12
  474356:	mov    %rdx,%r12
  474359:	push   %rbp
  47435a:	mov    %rsi,%rbp
  47435d:	push   %rbx
  47435e:	mov    %rdi,%rbx
  474361:	mov    0xc(%rdi),%edi
  474364:	call   4d8970 <ggml_get_type_traits_cpu>
  474369:	mov    0x18(%rbx),%rdx
  47436d:	mov    %r12,%rsi
  474370:	pop    %rbx
  474371:	mov    %rbp,%rdi
  474374:	mov    (%rax),%rax
  474377:	pop    %rbp
  474378:	pop    %r12
  47437a:	jmp    *%rax

Disassembly of section .fini:
