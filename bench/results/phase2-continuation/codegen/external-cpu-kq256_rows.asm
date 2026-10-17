
/home/ubuntu/guild-kernels-cpu/build/libguild_cpu_plugin.so:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

00000000000811e0 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii>:
   811e0:	endbr64
   811e4:	push   %r15
   811e6:	mov    %rsi,%r10
   811e9:	mov    %rdx,%rsi
   811ec:	push   %r14
   811ee:	push   %r13
   811f0:	push   %r12
   811f2:	push   %rbp
   811f3:	push   %rbx
   811f4:	sub    $0xa8,%rsp
   811fb:	mov    %ecx,0xc(%rsp)
   811ff:	mov    0xe8(%rsp),%ebx
   81206:	mov    %edi,0x8(%rsp)
   8120a:	mov    0xf0(%rsp),%ecx
   81211:	mov    %r8,0x38(%rsp)
   81216:	mov    0xe0(%rsp),%r13
   8121e:	mov    %fs:0x28,%rax
   81227:	mov    %rax,0x98(%rsp)
   8122f:	xor    %eax,%eax
   81231:	lea    0x1(%rbx),%eax
   81234:	cmp    %eax,%ecx
   81236:	jle    814a5 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x2c5>
   8123c:	mov    %rdx,%rdi
   8123f:	lea    -0x2(%rcx),%edx
   81242:	lea    -0x1(%r9),%r14d
   81246:	movslq %ebx,%rax
   81249:	sub    %ebx,%edx
   8124b:	imul   %rax,%rdi
   8124f:	lea    0x0(,%rax,4),%rbp
   81257:	shr    $0x3,%r14d
   8125b:	shr    $1,%edx
   8125d:	add    $0x1,%r14d
   81261:	lea    (%rdx,%rdx,1),%r8d
   81265:	lea    0x2(%rax,%r8,1),%rax
   8126a:	add    %r10,%rdi
   8126d:	shl    $0x2,%rax
   81271:	mov    %rax,0x30(%rsp)
   81276:	lea    0x0(,%r14,8),%rax
   8127e:	mov    %rax,0x18(%rsp)
   81283:	lea    (%rsi,%rdi,1),%r14
   81287:	test   %r9d,%r9d
   8128a:	jle    81505 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x325>
   81290:	lea    0x70(%rsp),%rax
   81295:	mov    %edx,0x44(%rsp)
   81299:	mov    %rbp,%r12
   8129c:	mov    %r10,0x48(%rsp)
   812a1:	mov    %r9d,0x40(%rsp)
   812a6:	mov    %ebx,0xe8(%rsp)
   812ad:	mov    %ecx,0xf0(%rsp)
   812b4:	mov    %rax,(%rsp)
   812b8:	lea    0x50(%rsp),%rax
   812bd:	mov    %rax,0x20(%rsp)
   812c2:	mov    %rdi,%rax
   812c5:	nopl   (%rax)
   812c8:	mov    0x38(%rsp),%rbp
   812cd:	mov    %rsi,0x28(%rsp)
   812d2:	xor    %r15d,%r15d
   812d5:	mov    %rax,0x10(%rsp)
   812da:	mov    0x40(%rsp),%ebx
   812de:	mov    %r13,%rax
   812e1:	mov    %r15,%r13
   812e4:	mov    %rbp,%r15
   812e7:	mov    %rax,%rbp
   812ea:	nopw   0x0(%rax,%rax,1)
   812f0:	mov    $0x8,%r9d
   812f6:	push   (%rsp)
   812f9:	mov    %r14,%rdx
   812fc:	mov    %r15,%r8
   812ff:	push   0x28(%rsp)
   81303:	cmp    %r9d,%ebx
   81306:	mov    0x1c(%rsp),%ecx
   8130a:	cmovle %ebx,%r9d
   8130e:	mov    0x20(%rsp),%rsi
   81313:	mov    0x18(%rsp),%edi
   81317:	call   803f0 <_ZN5guild7kernels3cpu12_GLOBAL__N_19dot_2rowsEiPKhS4_iPKPKviPfS9_>
   8131c:	vmovss 0x60(%rsp),%xmm0
   81322:	mov    0x0(%rbp,%r13,8),%rdx
   81327:	vinsertps $0x10,0x80(%rsp),%xmm0,%xmm0
   81332:	vmovlps %xmm0,(%rdx,%r12,1)
   81338:	pop    %rax
   81339:	pop    %rdx
   8133a:	cmp    $0x1,%ebx
   8133d:	jle    81446 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x266>
   81343:	lea    0x1(%r13),%ecx
   81347:	vmovss 0x54(%rsp),%xmm0
   8134d:	mov    %r13d,%edx
   81350:	vinsertps $0x10,0x74(%rsp),%xmm0,%xmm0
   81358:	mov    0x0(%rbp,%rcx,8),%rcx
   8135d:	vmovlps %xmm0,(%rcx,%r12,1)
   81363:	cmp    $0x2,%ebx
   81366:	jle    81446 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x266>
   8136c:	lea    0x2(%r13),%ecx
   81370:	vmovss 0x58(%rsp),%xmm0
   81376:	vinsertps $0x10,0x78(%rsp),%xmm0,%xmm0
   8137e:	mov    0x0(%rbp,%rcx,8),%rcx
   81383:	vmovlps %xmm0,(%rcx,%r12,1)
   81389:	cmp    $0x3,%ebx
   8138c:	jle    81446 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x266>
   81392:	lea    0x3(%r13),%ecx
   81396:	vmovss 0x5c(%rsp),%xmm0
   8139c:	vinsertps $0x10,0x7c(%rsp),%xmm0,%xmm0
   813a4:	mov    0x0(%rbp,%rcx,8),%rcx
   813a9:	vmovlps %xmm0,(%rcx,%r12,1)
   813af:	cmp    $0x4,%ebx
   813b2:	jle    81446 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x266>
   813b8:	lea    0x4(%r13),%ecx
   813bc:	vmovss 0x60(%rsp),%xmm0
   813c2:	vinsertps $0x10,0x80(%rsp),%xmm0,%xmm0
   813cd:	mov    0x0(%rbp,%rcx,8),%rcx
   813d2:	vmovlps %xmm0,(%rcx,%r12,1)
   813d8:	cmp    $0x5,%ebx
   813db:	jle    81446 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x266>
   813dd:	lea    0x5(%r13),%ecx
   813e1:	vmovss 0x64(%rsp),%xmm0
   813e7:	vinsertps $0x10,0x84(%rsp),%xmm0,%xmm0
   813f2:	mov    0x0(%rbp,%rcx,8),%rcx
   813f7:	vmovlps %xmm0,(%rcx,%r12,1)
   813fd:	cmp    $0x6,%ebx
   81400:	jle    81446 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x266>
   81402:	lea    0x6(%r13),%ecx
   81406:	vmovss 0x68(%rsp),%xmm0
   8140c:	vinsertps $0x10,0x88(%rsp),%xmm0,%xmm0
   81417:	mov    0x0(%rbp,%rcx,8),%rcx
   8141c:	vmovlps %xmm0,(%rcx,%r12,1)
   81422:	cmp    $0x7,%ebx
   81425:	jle    81446 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x266>
   81427:	add    $0x7,%edx
   8142a:	vmovss 0x6c(%rsp),%xmm0
   81430:	vinsertps $0x10,0x8c(%rsp),%xmm0,%xmm0
   8143b:	mov    0x0(%rbp,%rdx,8),%rdx
   81440:	vmovlps %xmm0,(%rdx,%r12,1)
   81446:	mov    0x18(%rsp),%rax
   8144b:	add    $0x8,%r13
   8144f:	sub    $0x8,%ebx
   81452:	add    $0x40,%r15
   81456:	cmp    %rax,%r13
   81459:	jne    812f0 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x110>
   8145f:	mov    0x28(%rsp),%rsi
   81464:	mov    0x30(%rsp),%rax
   81469:	add    $0x8,%r12
   8146d:	mov    %rbp,%r13
   81470:	lea    (%rsi,%r14,1),%rdi
   81474:	cmp    %rax,%r12
   81477:	je     81485 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x2a5>
   81479:	mov    %rdi,%rax
   8147c:	lea    (%rsi,%rdi,1),%r14
   81480:	jmp    812c8 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0xe8>
   81485:	mov    0x44(%rsp),%edx
   81489:	mov    0x48(%rsp),%r10
   8148e:	mov    0x40(%rsp),%r9d
   81493:	mov    0xe8(%rsp),%ebx
   8149a:	mov    0xf0(%rsp),%ecx
   814a1:	lea    0x2(%rbx,%rdx,2),%ebx
   814a5:	cmp    %ebx,%ecx
   814a7:	jle    814dc <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x2fc>
   814a9:	movslq %ebx,%rdi
   814ac:	mov    %rsi,%rdx
   814af:	sub    %ebx,%ecx
   814b1:	imul   %rdi,%rdx
   814b5:	add    %rdi,%rcx
   814b8:	lea    0x0(,%rdi,4),%rax
   814c0:	lea    0x0(,%rcx,4),%rdi
   814c8:	add    %r10,%rdx
   814cb:	test   %r9d,%r9d
   814ce:	jg     8151d <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x33d>
   814d0:	add    $0x4,%rax
   814d4:	add    %rsi,%rdx
   814d7:	cmp    %rdi,%rax
   814da:	jne    814cb <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x2eb>
   814dc:	mov    0x98(%rsp),%rax
   814e4:	sub    %fs:0x28,%rax
   814ed:	jne    816bd <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x4dd>
   814f3:	add    $0xa8,%rsp
   814fa:	pop    %rbx
   814fb:	pop    %rbp
   814fc:	pop    %r12
   814fe:	pop    %r13
   81500:	pop    %r14
   81502:	pop    %r15
   81504:	ret
   81505:	mov    0x30(%rsp),%rax
   8150a:	add    $0x8,%rbp
   8150e:	lea    (%rsi,%r14,1),%rdi
   81512:	cmp    %rax,%rbp
   81515:	jne    81283 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0xa3>
   8151b:	jmp    814a1 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x2c1>
   8151d:	lea    -0x1(%r9),%ebx
   81521:	lea    0x70(%rsp),%rcx
   81526:	mov    %rax,%r14
   81529:	mov    %r13,%r15
   8152c:	shr    $0x3,%ebx
   8152f:	mov    %rcx,(%rsp)
   81533:	add    $0x1,%ebx
   81536:	shl    $0x3,%rbx
   8153a:	mov    %rbx,%rax
   8153d:	mov    %rdx,%rbx
   81540:	mov    0x38(%rsp),%r13
   81545:	mov    %rdi,0x18(%rsp)
   8154a:	xor    %r12d,%r12d
   8154d:	mov    %r9d,%ebp
   81550:	mov    %rsi,0x20(%rsp)
   81555:	mov    %r9d,0x28(%rsp)
   8155a:	mov    %r13,%rdx
   8155d:	mov    %r14,%r13
   81560:	mov    %rbx,0x10(%rsp)
   81565:	mov    %rax,%rbx
   81568:	mov    %r15,%rax
   8156b:	mov    %rdx,%r14
   8156e:	mov    %r12,%r15
   81571:	mov    %rax,%r12
   81574:	nopl   0x0(%rax)
   81578:	mov    $0x8,%r8d
   8157e:	mov    0xc(%rsp),%edx
   81582:	mov    (%rsp),%r9
   81586:	mov    %r14,%rcx
   81589:	cmp    %r8d,%ebp
   8158c:	mov    0x10(%rsp),%rsi
   81591:	mov    0x8(%rsp),%edi
   81595:	cmovle %ebp,%r8d
   81599:	call   76fa0 <_ZN5guild7kernels3cpu12_GLOBAL__N_18dot_rowsEiPKhiPKPKviPf>
   8159e:	mov    (%r12,%r15,8),%rdx
   815a2:	vmovss 0x70(%rsp),%xmm0
   815a8:	vmovss %xmm0,(%rdx,%r13,1)
   815ae:	cmp    $0x1,%ebp
   815b1:	jle    81677 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x497>
   815b7:	lea    0x1(%r15),%esi
   815bb:	vmovss 0x74(%rsp),%xmm0
   815c1:	mov    %r15d,%edx
   815c4:	mov    (%r12,%rsi,8),%rsi
   815c8:	vmovss %xmm0,(%rsi,%r13,1)
   815ce:	cmp    $0x2,%ebp
   815d1:	jle    81677 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x497>
   815d7:	lea    0x2(%r15),%esi
   815db:	vmovss 0x78(%rsp),%xmm0
   815e1:	mov    (%r12,%rsi,8),%rsi
   815e5:	vmovss %xmm0,(%rsi,%r13,1)
   815eb:	cmp    $0x3,%ebp
   815ee:	jle    81677 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x497>
   815f4:	lea    0x3(%r15),%esi
   815f8:	vmovss 0x7c(%rsp),%xmm0
   815fe:	mov    (%r12,%rsi,8),%rsi
   81602:	vmovss %xmm0,(%rsi,%r13,1)
   81608:	cmp    $0x4,%ebp
   8160b:	jle    81677 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x497>
   8160d:	vmovss 0x80(%rsp),%xmm0
   81616:	lea    0x4(%r15),%esi
   8161a:	mov    (%r12,%rsi,8),%rsi
   8161e:	vmovss %xmm0,(%rsi,%r13,1)
   81624:	cmp    $0x5,%ebp
   81627:	jle    81677 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x497>
   81629:	vmovss 0x84(%rsp),%xmm0
   81632:	lea    0x5(%r15),%esi
   81636:	mov    (%r12,%rsi,8),%rsi
   8163a:	vmovss %xmm0,(%rsi,%r13,1)
   81640:	cmp    $0x6,%ebp
   81643:	jle    81677 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x497>
   81645:	vmovss 0x88(%rsp),%xmm0
   8164e:	lea    0x6(%r15),%esi
   81652:	mov    (%r12,%rsi,8),%rsi
   81656:	vmovss %xmm0,(%rsi,%r13,1)
   8165c:	cmp    $0x7,%ebp
   8165f:	jle    81677 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x497>
   81661:	vmovss 0x8c(%rsp),%xmm0
   8166a:	add    $0x7,%edx
   8166d:	mov    (%r12,%rdx,8),%rdx
   81671:	vmovss %xmm0,(%rdx,%r13,1)
   81677:	add    $0x8,%r15
   8167b:	sub    $0x8,%ebp
   8167e:	add    $0x40,%r14
   81682:	cmp    %r15,%rbx
   81685:	jne    81578 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x398>
   8168b:	mov    %r13,%r14
   8168e:	mov    0x18(%rsp),%rdi
   81693:	mov    %rbx,%rax
   81696:	mov    0x20(%rsp),%rsi
   8169b:	add    $0x4,%r14
   8169f:	mov    0x10(%rsp),%rbx
   816a4:	mov    0x28(%rsp),%r9d
   816a9:	mov    %r12,%r15
   816ac:	cmp    %r14,%rdi
   816af:	je     814dc <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x2fc>
   816b5:	add    %rsi,%rbx
   816b8:	jmp    81540 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x360>
   816bd:	call   1cae0 <__stack_chk_fail@plt>

Disassembly of section .fini:
