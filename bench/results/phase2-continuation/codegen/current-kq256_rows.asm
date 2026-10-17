
build-cuda12/ornith_prefill_bench:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

00000000004d83e0 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii>:
  4d83e0:	endbr64
  4d83e4:	push   %r15
  4d83e6:	mov    %rsi,%r10
  4d83e9:	mov    %rdx,%rsi
  4d83ec:	push   %r14
  4d83ee:	push   %r13
  4d83f0:	push   %r12
  4d83f2:	push   %rbp
  4d83f3:	push   %rbx
  4d83f4:	sub    $0xa8,%rsp
  4d83fb:	mov    %ecx,0xc(%rsp)
  4d83ff:	mov    0xe8(%rsp),%ebx
  4d8406:	mov    %edi,0x8(%rsp)
  4d840a:	mov    0xf0(%rsp),%ecx
  4d8411:	mov    %r8,0x38(%rsp)
  4d8416:	mov    0xe0(%rsp),%r13
  4d841e:	mov    %fs:0x28,%rax
  4d8427:	mov    %rax,0x98(%rsp)
  4d842f:	xor    %eax,%eax
  4d8431:	lea    0x1(%rbx),%eax
  4d8434:	cmp    %eax,%ecx
  4d8436:	jle    4d86a5 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x2c5>
  4d843c:	mov    %rdx,%rdi
  4d843f:	lea    -0x2(%rcx),%edx
  4d8442:	lea    -0x1(%r9),%r14d
  4d8446:	movslq %ebx,%rax
  4d8449:	sub    %ebx,%edx
  4d844b:	imul   %rax,%rdi
  4d844f:	lea    0x0(,%rax,4),%rbp
  4d8457:	shr    $0x3,%r14d
  4d845b:	shr    $1,%edx
  4d845d:	add    $0x1,%r14d
  4d8461:	lea    (%rdx,%rdx,1),%r8d
  4d8465:	lea    0x2(%rax,%r8,1),%rax
  4d846a:	add    %r10,%rdi
  4d846d:	shl    $0x2,%rax
  4d8471:	mov    %rax,0x30(%rsp)
  4d8476:	lea    0x0(,%r14,8),%rax
  4d847e:	mov    %rax,0x18(%rsp)
  4d8483:	lea    (%rsi,%rdi,1),%r14
  4d8487:	test   %r9d,%r9d
  4d848a:	jle    4d8705 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x325>
  4d8490:	lea    0x70(%rsp),%rax
  4d8495:	mov    %edx,0x44(%rsp)
  4d8499:	mov    %rbp,%r12
  4d849c:	mov    %r10,0x48(%rsp)
  4d84a1:	mov    %r9d,0x40(%rsp)
  4d84a6:	mov    %ebx,0xe8(%rsp)
  4d84ad:	mov    %ecx,0xf0(%rsp)
  4d84b4:	mov    %rax,(%rsp)
  4d84b8:	lea    0x50(%rsp),%rax
  4d84bd:	mov    %rax,0x20(%rsp)
  4d84c2:	mov    %rdi,%rax
  4d84c5:	nopl   (%rax)
  4d84c8:	mov    0x38(%rsp),%rbp
  4d84cd:	mov    %rsi,0x28(%rsp)
  4d84d2:	xor    %r15d,%r15d
  4d84d5:	mov    %rax,0x10(%rsp)
  4d84da:	mov    0x40(%rsp),%ebx
  4d84de:	mov    %r13,%rax
  4d84e1:	mov    %r15,%r13
  4d84e4:	mov    %rbp,%r15
  4d84e7:	mov    %rax,%rbp
  4d84ea:	nopw   0x0(%rax,%rax,1)
  4d84f0:	mov    $0x8,%r9d
  4d84f6:	push   (%rsp)
  4d84f9:	mov    %r14,%rdx
  4d84fc:	mov    %r15,%r8
  4d84ff:	push   0x28(%rsp)
  4d8503:	cmp    %r9d,%ebx
  4d8506:	mov    0x1c(%rsp),%ecx
  4d850a:	cmovle %ebx,%r9d
  4d850e:	mov    0x20(%rsp),%rsi
  4d8513:	mov    0x18(%rsp),%edi
  4d8517:	call   4d75f0 <_ZN5guild7kernels3cpu12_GLOBAL__N_19dot_2rowsEiPKhS4_iPKPKviPfS9_>
  4d851c:	vmovss 0x60(%rsp),%xmm0
  4d8522:	mov    0x0(%rbp,%r13,8),%rdx
  4d8527:	vinsertps $0x10,0x80(%rsp),%xmm0,%xmm0
  4d8532:	vmovlps %xmm0,(%rdx,%r12,1)
  4d8538:	pop    %rax
  4d8539:	pop    %rdx
  4d853a:	cmp    $0x1,%ebx
  4d853d:	jle    4d8646 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x266>
  4d8543:	lea    0x1(%r13),%ecx
  4d8547:	vmovss 0x54(%rsp),%xmm0
  4d854d:	mov    %r13d,%edx
  4d8550:	vinsertps $0x10,0x74(%rsp),%xmm0,%xmm0
  4d8558:	mov    0x0(%rbp,%rcx,8),%rcx
  4d855d:	vmovlps %xmm0,(%rcx,%r12,1)
  4d8563:	cmp    $0x2,%ebx
  4d8566:	jle    4d8646 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x266>
  4d856c:	lea    0x2(%r13),%ecx
  4d8570:	vmovss 0x58(%rsp),%xmm0
  4d8576:	vinsertps $0x10,0x78(%rsp),%xmm0,%xmm0
  4d857e:	mov    0x0(%rbp,%rcx,8),%rcx
  4d8583:	vmovlps %xmm0,(%rcx,%r12,1)
  4d8589:	cmp    $0x3,%ebx
  4d858c:	jle    4d8646 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x266>
  4d8592:	lea    0x3(%r13),%ecx
  4d8596:	vmovss 0x5c(%rsp),%xmm0
  4d859c:	vinsertps $0x10,0x7c(%rsp),%xmm0,%xmm0
  4d85a4:	mov    0x0(%rbp,%rcx,8),%rcx
  4d85a9:	vmovlps %xmm0,(%rcx,%r12,1)
  4d85af:	cmp    $0x4,%ebx
  4d85b2:	jle    4d8646 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x266>
  4d85b8:	lea    0x4(%r13),%ecx
  4d85bc:	vmovss 0x60(%rsp),%xmm0
  4d85c2:	vinsertps $0x10,0x80(%rsp),%xmm0,%xmm0
  4d85cd:	mov    0x0(%rbp,%rcx,8),%rcx
  4d85d2:	vmovlps %xmm0,(%rcx,%r12,1)
  4d85d8:	cmp    $0x5,%ebx
  4d85db:	jle    4d8646 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x266>
  4d85dd:	lea    0x5(%r13),%ecx
  4d85e1:	vmovss 0x64(%rsp),%xmm0
  4d85e7:	vinsertps $0x10,0x84(%rsp),%xmm0,%xmm0
  4d85f2:	mov    0x0(%rbp,%rcx,8),%rcx
  4d85f7:	vmovlps %xmm0,(%rcx,%r12,1)
  4d85fd:	cmp    $0x6,%ebx
  4d8600:	jle    4d8646 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x266>
  4d8602:	lea    0x6(%r13),%ecx
  4d8606:	vmovss 0x68(%rsp),%xmm0
  4d860c:	vinsertps $0x10,0x88(%rsp),%xmm0,%xmm0
  4d8617:	mov    0x0(%rbp,%rcx,8),%rcx
  4d861c:	vmovlps %xmm0,(%rcx,%r12,1)
  4d8622:	cmp    $0x7,%ebx
  4d8625:	jle    4d8646 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x266>
  4d8627:	add    $0x7,%edx
  4d862a:	vmovss 0x6c(%rsp),%xmm0
  4d8630:	vinsertps $0x10,0x8c(%rsp),%xmm0,%xmm0
  4d863b:	mov    0x0(%rbp,%rdx,8),%rdx
  4d8640:	vmovlps %xmm0,(%rdx,%r12,1)
  4d8646:	mov    0x18(%rsp),%rax
  4d864b:	add    $0x8,%r13
  4d864f:	sub    $0x8,%ebx
  4d8652:	add    $0x40,%r15
  4d8656:	cmp    %rax,%r13
  4d8659:	jne    4d84f0 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x110>
  4d865f:	mov    0x28(%rsp),%rsi
  4d8664:	mov    0x30(%rsp),%rax
  4d8669:	add    $0x8,%r12
  4d866d:	mov    %rbp,%r13
  4d8670:	lea    (%rsi,%r14,1),%rdi
  4d8674:	cmp    %rax,%r12
  4d8677:	je     4d8685 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x2a5>
  4d8679:	mov    %rdi,%rax
  4d867c:	lea    (%rsi,%rdi,1),%r14
  4d8680:	jmp    4d84c8 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0xe8>
  4d8685:	mov    0x44(%rsp),%edx
  4d8689:	mov    0x48(%rsp),%r10
  4d868e:	mov    0x40(%rsp),%r9d
  4d8693:	mov    0xe8(%rsp),%ebx
  4d869a:	mov    0xf0(%rsp),%ecx
  4d86a1:	lea    0x2(%rbx,%rdx,2),%ebx
  4d86a5:	cmp    %ebx,%ecx
  4d86a7:	jle    4d86dc <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x2fc>
  4d86a9:	movslq %ebx,%rdi
  4d86ac:	mov    %rsi,%rdx
  4d86af:	sub    %ebx,%ecx
  4d86b1:	imul   %rdi,%rdx
  4d86b5:	add    %rdi,%rcx
  4d86b8:	lea    0x0(,%rdi,4),%rax
  4d86c0:	lea    0x0(,%rcx,4),%rdi
  4d86c8:	add    %r10,%rdx
  4d86cb:	test   %r9d,%r9d
  4d86ce:	jg     4d871d <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x33d>
  4d86d0:	add    $0x4,%rax
  4d86d4:	add    %rsi,%rdx
  4d86d7:	cmp    %rdi,%rax
  4d86da:	jne    4d86cb <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x2eb>
  4d86dc:	mov    0x98(%rsp),%rax
  4d86e4:	sub    %fs:0x28,%rax
  4d86ed:	jne    4d88bd <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x4dd>
  4d86f3:	add    $0xa8,%rsp
  4d86fa:	pop    %rbx
  4d86fb:	pop    %rbp
  4d86fc:	pop    %r12
  4d86fe:	pop    %r13
  4d8700:	pop    %r14
  4d8702:	pop    %r15
  4d8704:	ret
  4d8705:	mov    0x30(%rsp),%rax
  4d870a:	add    $0x8,%rbp
  4d870e:	lea    (%rsi,%r14,1),%rdi
  4d8712:	cmp    %rax,%rbp
  4d8715:	jne    4d8483 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0xa3>
  4d871b:	jmp    4d86a1 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x2c1>
  4d871d:	lea    -0x1(%r9),%ebx
  4d8721:	lea    0x70(%rsp),%rcx
  4d8726:	mov    %rax,%r14
  4d8729:	mov    %r13,%r15
  4d872c:	shr    $0x3,%ebx
  4d872f:	mov    %rcx,(%rsp)
  4d8733:	add    $0x1,%ebx
  4d8736:	shl    $0x3,%rbx
  4d873a:	mov    %rbx,%rax
  4d873d:	mov    %rdx,%rbx
  4d8740:	mov    0x38(%rsp),%r13
  4d8745:	mov    %rdi,0x18(%rsp)
  4d874a:	xor    %r12d,%r12d
  4d874d:	mov    %r9d,%ebp
  4d8750:	mov    %rsi,0x20(%rsp)
  4d8755:	mov    %r9d,0x28(%rsp)
  4d875a:	mov    %r13,%rdx
  4d875d:	mov    %r14,%r13
  4d8760:	mov    %rbx,0x10(%rsp)
  4d8765:	mov    %rax,%rbx
  4d8768:	mov    %r15,%rax
  4d876b:	mov    %rdx,%r14
  4d876e:	mov    %r12,%r15
  4d8771:	mov    %rax,%r12
  4d8774:	nopl   0x0(%rax)
  4d8778:	mov    $0x8,%r8d
  4d877e:	mov    0xc(%rsp),%edx
  4d8782:	mov    (%rsp),%r9
  4d8786:	mov    %r14,%rcx
  4d8789:	cmp    %r8d,%ebp
  4d878c:	mov    0x10(%rsp),%rsi
  4d8791:	mov    0x8(%rsp),%edi
  4d8795:	cmovle %ebp,%r8d
  4d8799:	call   4ce1a0 <_ZN5guild7kernels3cpu12_GLOBAL__N_18dot_rowsEiPKhiPKPKviPf>
  4d879e:	mov    (%r12,%r15,8),%rdx
  4d87a2:	vmovss 0x70(%rsp),%xmm0
  4d87a8:	vmovss %xmm0,(%rdx,%r13,1)
  4d87ae:	cmp    $0x1,%ebp
  4d87b1:	jle    4d8877 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x497>
  4d87b7:	lea    0x1(%r15),%esi
  4d87bb:	vmovss 0x74(%rsp),%xmm0
  4d87c1:	mov    %r15d,%edx
  4d87c4:	mov    (%r12,%rsi,8),%rsi
  4d87c8:	vmovss %xmm0,(%rsi,%r13,1)
  4d87ce:	cmp    $0x2,%ebp
  4d87d1:	jle    4d8877 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x497>
  4d87d7:	lea    0x2(%r15),%esi
  4d87db:	vmovss 0x78(%rsp),%xmm0
  4d87e1:	mov    (%r12,%rsi,8),%rsi
  4d87e5:	vmovss %xmm0,(%rsi,%r13,1)
  4d87eb:	cmp    $0x3,%ebp
  4d87ee:	jle    4d8877 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x497>
  4d87f4:	lea    0x3(%r15),%esi
  4d87f8:	vmovss 0x7c(%rsp),%xmm0
  4d87fe:	mov    (%r12,%rsi,8),%rsi
  4d8802:	vmovss %xmm0,(%rsi,%r13,1)
  4d8808:	cmp    $0x4,%ebp
  4d880b:	jle    4d8877 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x497>
  4d880d:	vmovss 0x80(%rsp),%xmm0
  4d8816:	lea    0x4(%r15),%esi
  4d881a:	mov    (%r12,%rsi,8),%rsi
  4d881e:	vmovss %xmm0,(%rsi,%r13,1)
  4d8824:	cmp    $0x5,%ebp
  4d8827:	jle    4d8877 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x497>
  4d8829:	vmovss 0x84(%rsp),%xmm0
  4d8832:	lea    0x5(%r15),%esi
  4d8836:	mov    (%r12,%rsi,8),%rsi
  4d883a:	vmovss %xmm0,(%rsi,%r13,1)
  4d8840:	cmp    $0x6,%ebp
  4d8843:	jle    4d8877 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x497>
  4d8845:	vmovss 0x88(%rsp),%xmm0
  4d884e:	lea    0x6(%r15),%esi
  4d8852:	mov    (%r12,%rsi,8),%rsi
  4d8856:	vmovss %xmm0,(%rsi,%r13,1)
  4d885c:	cmp    $0x7,%ebp
  4d885f:	jle    4d8877 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x497>
  4d8861:	vmovss 0x8c(%rsp),%xmm0
  4d886a:	add    $0x7,%edx
  4d886d:	mov    (%r12,%rdx,8),%rdx
  4d8871:	vmovss %xmm0,(%rdx,%r13,1)
  4d8877:	add    $0x8,%r15
  4d887b:	sub    $0x8,%ebp
  4d887e:	add    $0x40,%r14
  4d8882:	cmp    %r15,%rbx
  4d8885:	jne    4d8778 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x398>
  4d888b:	mov    %r13,%r14
  4d888e:	mov    0x18(%rsp),%rdi
  4d8893:	mov    %rbx,%rax
  4d8896:	mov    0x20(%rsp),%rsi
  4d889b:	add    $0x4,%r14
  4d889f:	mov    0x10(%rsp),%rbx
  4d88a4:	mov    0x28(%rsp),%r9d
  4d88a9:	mov    %r12,%r15
  4d88ac:	cmp    %r14,%rdi
  4d88af:	je     4d86dc <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x2fc>
  4d88b5:	add    %rsi,%rbx
  4d88b8:	jmp    4d8740 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x360>
  4d88bd:	call   13390 <__stack_chk_fail@plt>

Disassembly of section .fini:
