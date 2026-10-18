
/home/ubuntu/Guild/bench/results/external-kernels/phase1/ornith_prefill_bench:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

0000000000474320 <_ZN5guild7kernels3cpu16native_quant_actERKNS1_9NativeFmtEPKfPv>:
  474320:	endbr64
  474324:	push   %r12
  474326:	mov    %rdx,%r12
  474329:	push   %rbp
  47432a:	mov    %rsi,%rbp
  47432d:	push   %rbx
  47432e:	mov    %rdi,%rbx
  474331:	mov    0x8(%rdi),%edi
  474334:	call   4d8970 <ggml_get_type_traits_cpu>
  474339:	mov    0x10(%rbx),%rdx
  47433d:	mov    %r12,%rsi
  474340:	pop    %rbx
  474341:	mov    %rbp,%rdi
  474344:	mov    (%rax),%rax
  474347:	pop    %rbp
  474348:	pop    %r12
  47434a:	jmp    *%rax

Disassembly of section .fini:
