
build-cuda12/ornith_prefill_bench:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

00000000004d81f0 <_ZN5guild7kernels3cpu13kq256_gu_rowsEiPKhmmiPKPKviPKPfii>:
  4d81f0:	endbr64
  4d81f4:	push   %r15
  4d81f6:	push   %r14
  4d81f8:	push   %r13
  4d81fa:	push   %r12
  4d81fc:	push   %rbp
  4d81fd:	push   %rbx
  4d81fe:	mov    %rdx,%rbx
  4d8201:	sub    $0xb8,%rsp
  4d8208:	mov    0xf8(%rsp),%rax
  4d8210:	mov    %edi,0x30(%rsp)
  4d8214:	mov    %rsi,%rdi
  4d8217:	mov    %rcx,0x40(%rsp)
  4d821c:	mov    0x100(%rsp),%esi
  4d8223:	mov    %rax,0x50(%rsp)
  4d8228:	mov    0x108(%rsp),%eax
  4d822f:	mov    %r8d,0x34(%rsp)
  4d8234:	mov    %r9,0x48(%rsp)
  4d8239:	mov    %fs:0x28,%rdx
  4d8242:	mov    %rdx,0xa8(%rsp)
  4d824a:	xor    %edx,%edx
  4d824c:	cmp    %eax,%esi
  4d824e:	jge    4d83b3 <_ZN5guild7kernels3cpu13kq256_gu_rowsEiPKhmmiPKPKviPKPfii+0x1c3>
  4d8254:	mov    0xf0(%rsp),%ecx
  4d825b:	test   %ecx,%ecx
  4d825d:	jle    4d83b3 <_ZN5guild7kernels3cpu13kq256_gu_rowsEiPKhmmiPKPKviPKPfii+0x1c3>
  4d8263:	movslq %esi,%rdx
  4d8266:	sub    %esi,%eax
  4d8268:	mov    %rbx,%rcx
  4d826b:	mov    %rbx,0x58(%rsp)
  4d8270:	add    %rdx,%rax
  4d8273:	imul   %rdx,%rcx
  4d8277:	lea    0x0(,%rdx,4),%r14
  4d827f:	shl    $0x2,%rax
  4d8283:	lea    0x80(%rsp),%r12
  4d828b:	lea    0x60(%rsp),%r15
  4d8290:	mov    %rax,0x38(%rsp)
  4d8295:	mov    0xf0(%rsp),%eax
  4d829c:	add    %rdi,%rcx
  4d829f:	sub    $0x1,%eax
  4d82a2:	and    $0xfffffff8,%eax
  4d82a5:	add    $0x8,%eax
  4d82a8:	mov    %eax,0x2c(%rsp)
  4d82ac:	nopl   0x0(%rax)
  4d82b0:	mov    0x40(%rsp),%rax
  4d82b5:	mov    0x50(%rsp),%rbp
  4d82ba:	mov    %rcx,0x18(%rsp)
  4d82bf:	movl   $0x0,0x28(%rsp)
  4d82c7:	add    %rcx,%rax
  4d82ca:	mov    %rax,0x20(%rsp)
  4d82cf:	mov    0x48(%rsp),%rax
  4d82d4:	mov    %rax,0x10(%rsp)
  4d82d9:	nopl   0x0(%rax)
  4d82e0:	mov    0x28(%rsp),%eax
  4d82e4:	mov    0xf0(%rsp),%ebx
  4d82eb:	push   %r12
  4d82ed:	push   %r15
  4d82ef:	mov    0x20(%rsp),%r8
  4d82f4:	sub    %eax,%ebx
  4d82f6:	mov    $0x8,%eax
  4d82fb:	mov    0x44(%rsp),%ecx
  4d82ff:	mov    0x30(%rsp),%rdx
  4d8304:	cmp    %eax,%ebx
  4d8306:	mov    0x28(%rsp),%rsi
  4d830b:	mov    0x40(%rsp),%edi
  4d830f:	cmovg  %eax,%ebx
  4d8312:	xor    %r13d,%r13d
  4d8315:	mov    %ebx,%r9d
  4d8318:	movslq %ebx,%rbx
  4d831b:	call   4d75f0 <_ZN5guild7kernels3cpu12_GLOBAL__N_19dot_2rowsEiPKhS4_iPKPKviPfS9_>
  4d8320:	shl    $0x2,%rbx
  4d8324:	pop    %rax
  4d8325:	pop    %rdx
  4d8326:	cs nopw 0x0(%rax,%rax,1)
  4d8330:	vmovss (%r15,%r13,1),%xmm1
  4d8336:	vxorps 0x1a9562(%rip),%xmm1,%xmm0        # 6818a0 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0x28a0>
  4d833e:	vmovss %xmm1,0xc(%rsp)
  4d8344:	call   13b60 <expf@plt>
  4d8349:	vmovss 0xc(%rsp),%xmm1
  4d834f:	mov    0x0(%rbp,%r13,2),%rax
  4d8354:	vaddss 0x160ca8(%rip),%xmm0,%xmm0        # 639004 <_IO_stdin_used+0x4>
  4d835c:	vdivss %xmm0,%xmm1,%xmm0
  4d8360:	vmulss (%r12,%r13,1),%xmm0,%xmm0
  4d8366:	add    $0x4,%r13
  4d836a:	vmovss %xmm0,(%rax,%r14,1)
  4d8370:	cmp    %r13,%rbx
  4d8373:	jne    4d8330 <_ZN5guild7kernels3cpu13kq256_gu_rowsEiPKhmmiPKPKviPKPfii+0x140>
  4d8375:	addl   $0x8,0x28(%rsp)
  4d837a:	mov    0x2c(%rsp),%esi
  4d837e:	add    $0x40,%rbp
  4d8382:	mov    0x28(%rsp),%eax
  4d8386:	addq   $0x40,0x10(%rsp)
  4d838c:	cmp    %esi,%eax
  4d838e:	jne    4d82e0 <_ZN5guild7kernels3cpu13kq256_gu_rowsEiPKhmmiPKPKviPKPfii+0xf0>
  4d8394:	mov    0x58(%rsp),%rax
  4d8399:	mov    0x18(%rsp),%rcx
  4d839e:	add    $0x4,%r14
  4d83a2:	add    %rax,%rcx
  4d83a5:	mov    0x38(%rsp),%rax
  4d83aa:	cmp    %rax,%r14
  4d83ad:	jne    4d82b0 <_ZN5guild7kernels3cpu13kq256_gu_rowsEiPKhmmiPKPKviPKPfii+0xc0>
  4d83b3:	mov    0xa8(%rsp),%rax
  4d83bb:	sub    %fs:0x28,%rax
  4d83c4:	jne    4d83d8 <_ZN5guild7kernels3cpu13kq256_gu_rowsEiPKhmmiPKPKviPKPfii+0x1e8>
  4d83c6:	add    $0xb8,%rsp
  4d83cd:	pop    %rbx
  4d83ce:	pop    %rbp
  4d83cf:	pop    %r12
  4d83d1:	pop    %r13
  4d83d3:	pop    %r14
  4d83d5:	pop    %r15
  4d83d7:	ret
  4d83d8:	call   13390 <__stack_chk_fail@plt>

Disassembly of section .fini:
