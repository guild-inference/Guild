
/home/ubuntu/Guild/bench/results/external-kernels/phase1/ornith_prefill_bench:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

0000000000474380 <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii>:
  474380:	endbr64
  474384:	push   %r15
  474386:	push   %r14
  474388:	push   %r13
  47438a:	push   %r12
  47438c:	mov    %r9d,%r12d
  47438f:	push   %rbp
  474390:	movslq %ecx,%rbp
  474393:	push   %rbx
  474394:	sub    $0x78,%rsp
  474398:	mov    %rdi,0x38(%rsp)
  47439d:	mov    0xb0(%rsp),%r13d
  4743a5:	mov    %rsi,0x48(%rsp)
  4743aa:	mov    %rdx,0x40(%rsp)
  4743af:	mov    %r8,0x50(%rsp)
  4743b4:	mov    %fs:0x28,%rax
  4743bd:	mov    %rax,0x68(%rsp)
  4743c2:	xor    %eax,%eax
  4743c4:	movzbl 0x1df3ee5(%rip),%eax        # 22682b0 <_ZGVZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfiiE6avx512>
  4743cb:	test   %al,%al
  4743cd:	jne    4743e6 <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x66>
  4743cf:	lea    0x1df3eda(%rip),%rbx        # 22682b0 <_ZGVZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfiiE6avx512>
  4743d6:	mov    %rbx,%rdi
  4743d9:	call   13d80 <__cxa_guard_acquire@plt>
  4743de:	test   %eax,%eax
  4743e0:	jne    47472a <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x3aa>
  4743e6:	movzbl 0x1df3eb3(%rip),%eax        # 22682a0 <_ZGVZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfiiE4avx2>
  4743ed:	test   %al,%al
  4743ef:	jne    474408 <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x88>
  4743f1:	lea    0x1df3ea8(%rip),%rbx        # 22682a0 <_ZGVZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfiiE4avx2>
  4743f8:	mov    %rbx,%rdi
  4743fb:	call   13d80 <__cxa_guard_acquire@plt>
  474400:	test   %eax,%eax
  474402:	jne    47470a <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x38a>
  474408:	movzbl 0x1df3e81(%rip),%eax        # 2268290 <_ZGVZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfiiE6mt_min>
  47440f:	test   %al,%al
  474411:	je     47468c <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x30c>
  474417:	movzbl 0x1df3e62(%rip),%eax        # 2268280 <_ZGVZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfiiE2kq>
  47441e:	test   %al,%al
  474420:	je     474631 <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x2b1>
  474426:	cmpb   $0x0,0x1df3e5b(%rip)        # 2268288 <_ZZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfiiE2kq>
  47442d:	je     474449 <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0xc9>
  47442f:	mov    0x38(%rsp),%rax
  474434:	mov    (%rax),%edi
  474436:	mov    %edi,%eax
  474438:	and    $0xfffffffd,%eax
  47443b:	cmp    $0xc,%eax
  47443e:	jne    474449 <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0xc9>
  474440:	cmp    $0x1,%ebp
  474443:	jg     47475b <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x3db>
  474449:	movzbl 0x1df3e20(%rip),%eax        # 2268270 <_ZGVZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfiiE6cpu512>
  474450:	test   %al,%al
  474452:	je     4746db <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x35b>
  474458:	mov    0x38(%rsp),%rax
  47445d:	cmp    %ebp,0x1df3e35(%rip)        # 2268298 <_ZZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfiiE6mt_min>
  474463:	mov    (%rax),%edi
  474465:	jle    4745d1 <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x251>
  47446b:	call   4d8970 <ggml_get_type_traits_cpu>
  474470:	mov    0x38(%rsp),%rdi
  474475:	mov    0x10(%rdi),%r14d
  474479:	cmp    %r13d,%r12d
  47447c:	jge    4745ae <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x22e>
  474482:	test   %ebp,%ebp
  474484:	jle    4745ae <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x22e>
  47448a:	movslq %r12d,%rsi
  47448d:	sub    %r12d,%r13d
  474490:	mov    0x8(%rax),%rbx
  474494:	lea    0x0(%r13,%rsi,1),%rax
  474499:	mov    %rax,0x58(%rsp)
  47449e:	mov    0x40(%rsp),%rax
  4744a3:	lea    (%rax,%rbp,8),%rax
  4744a7:	mov    %rax,0x18(%rsp)
  4744ac:	lea    0x60(%rsp),%rax
  4744b1:	mov    %rax,0x20(%rsp)
  4744b6:	lea    0x64(%rsp),%rax
  4744bb:	mov    %rax,0x28(%rsp)
  4744c0:	mov    0x38(%rsp),%rax
  4744c5:	mov    0x48(%rsp),%rcx
  4744ca:	mov    %rsi,0x30(%rsp)
  4744cf:	lea    0x0(,%rsi,4),%r12
  4744d7:	mov    0x40(%rsp),%r15
  4744dc:	mov    0x20(%rax),%rbp
  4744e0:	imul   %rsi,%rbp
  4744e4:	lea    (%rcx,%rbp,1),%r13
  4744e8:	add    0x30(%rax),%rbp
  4744ec:	lea    (%rcx,%rbp,1),%rax
  4744f0:	mov    0x50(%rsp),%rbp
  4744f5:	mov    %rax,0x10(%rsp)
  4744fa:	nopw   0x0(%rax,%rax,1)
  474500:	movl   $0x0,0x60(%rsp)
  474508:	xor    %r8d,%r8d
  47450b:	mov    %r13,%rcx
  47450e:	xor    %edx,%edx
  474510:	movl   $0x0,0x64(%rsp)
  474518:	mov    %r14d,%edi
  47451b:	add    $0x8,%r15
  47451f:	add    $0x8,%rbp
  474523:	push   $0x1
  474525:	push   $0x0
  474527:	mov    0x30(%rsp),%rsi
  47452c:	mov    -0x8(%r15),%r9
  474530:	call   *%rbx
  474532:	pop    %rax
  474533:	xor    %r8d,%r8d
  474536:	pop    %rdx
  474537:	mov    %r14d,%edi
  47453a:	push   $0x1
  47453c:	xor    %edx,%edx
  47453e:	push   $0x0
  474540:	mov    0x20(%rsp),%rcx
  474545:	mov    0x38(%rsp),%rsi
  47454a:	mov    -0x8(%r15),%r9
  47454e:	call   *%rbx
  474550:	movss  0x70(%rsp),%xmm1
  474556:	movaps %xmm1,%xmm0
  474559:	xorps  0x20c210(%rip),%xmm0        # 680770 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0x2950>
  474560:	movss  %xmm1,0x1c(%rsp)
  474566:	call   13b60 <expf@plt>
  47456b:	movss  0x1c(%rsp),%xmm1
  474571:	mov    -0x8(%rbp),%rdx
  474575:	addss  0x1c3a87(%rip),%xmm0        # 638004 <_IO_stdin_used+0x4>
  47457d:	divss  %xmm0,%xmm1
  474581:	mulss  0x74(%rsp),%xmm1
  474587:	movss  %xmm1,(%rdx,%r12,1)
  47458d:	pop    %rcx
  47458e:	pop    %rsi
  47458f:	cmp    %r15,0x18(%rsp)
  474594:	jne    474500 <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x180>
  47459a:	mov    0x30(%rsp),%rsi
  47459f:	add    $0x1,%rsi
  4745a3:	cmp    %rsi,0x58(%rsp)
  4745a8:	jne    4744c0 <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x140>
  4745ae:	mov    0x68(%rsp),%rax
  4745b3:	sub    %fs:0x28,%rax
  4745bc:	jne    47489e <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x51e>
  4745c2:	add    $0x78,%rsp
  4745c6:	pop    %rbx
  4745c7:	pop    %rbp
  4745c8:	pop    %r12
  4745ca:	pop    %r13
  4745cc:	pop    %r14
  4745ce:	pop    %r15
  4745d0:	ret
  4745d1:	call   490f00 <_ZN5guild7kernels3cpu15iq512_supportedEi>
  4745d6:	test   %al,%al
  4745d8:	jne    4745f7 <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x277>
  4745da:	mov    0x38(%rsp),%rax
  4745df:	cmpb   $0x0,0x1df3c92(%rip)        # 2268278 <_ZZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfiiE6cpu512>
  4745e6:	mov    (%rax),%edi
  4745e8:	jne    47446b <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0xeb>
  4745ee:	call   4bb0f0 <_ZN5guild7kernels3cpu15iq256_supportedEi>
  4745f3:	test   %al,%al
  4745f5:	je     474625 <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x2a5>
  4745f7:	mov    0x38(%rsp),%rax
  4745fc:	cmpb   $0x0,0x1df3cb5(%rip)        # 22682b8 <_ZZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfiiE6avx512>
  474603:	mov    (%rax),%edi
  474605:	jne    474746 <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x3c6>
  47460b:	cmpb   $0x0,0x1df3c96(%rip)        # 22682a8 <_ZZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfiiE4avx2>
  474612:	je     47446b <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0xeb>
  474618:	call   4bb0f0 <_ZN5guild7kernels3cpu15iq256_supportedEi>
  47461d:	test   %al,%al
  47461f:	jne    47482b <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x4ab>
  474625:	mov    0x38(%rsp),%rax
  47462a:	mov    (%rax),%edi
  47462c:	jmp    47446b <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0xeb>
  474631:	lea    0x1df3c48(%rip),%rbx        # 2268280 <_ZGVZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfiiE2kq>
  474638:	mov    %rbx,%rdi
  47463b:	call   13d80 <__cxa_guard_acquire@plt>
  474640:	test   %eax,%eax
  474642:	je     474426 <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0xa6>
  474648:	lea    0x1c92ab(%rip),%rdi        # 63d8fa <_IO_stdin_used+0x58fa>
  47464f:	call   134d0 <getenv@plt>
  474654:	mov    %rax,%r14
  474657:	call   46de70 <_ZN5guild7kernels3cpu11cpu_avx2_okEv>
  47465c:	test   %al,%al
  47465e:	je     474679 <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x2f9>
  474660:	test   %r14,%r14
  474663:	je     474679 <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x2f9>
  474665:	xor    %esi,%esi
  474667:	mov    $0xa,%edx
  47466c:	mov    %r14,%rdi
  47466f:	call   13f80 <__isoc23_strtol@plt>
  474674:	test   %eax,%eax
  474676:	setne  %al
  474679:	mov    %rbx,%rdi
  47467c:	mov    %al,0x1df3c06(%rip)        # 2268288 <_ZZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfiiE2kq>
  474682:	call   12e70 <__cxa_guard_release@plt>
  474687:	jmp    474426 <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0xa6>
  47468c:	lea    0x1df3bfd(%rip),%rbx        # 2268290 <_ZGVZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfiiE6mt_min>
  474693:	mov    %rbx,%rdi
  474696:	call   13d80 <__cxa_guard_acquire@plt>
  47469b:	test   %eax,%eax
  47469d:	je     474417 <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x97>
  4746a3:	lea    0x1c9240(%rip),%rdi        # 63d8ea <_IO_stdin_used+0x58ea>
  4746aa:	call   134d0 <getenv@plt>
  4746af:	mov    %rax,%rdi
  4746b2:	mov    $0x2,%eax
  4746b7:	test   %rdi,%rdi
  4746ba:	je     4746c8 <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x348>
  4746bc:	mov    $0xa,%edx
  4746c1:	xor    %esi,%esi
  4746c3:	call   13f80 <__isoc23_strtol@plt>
  4746c8:	mov    %rbx,%rdi
  4746cb:	mov    %eax,0x1df3bc7(%rip)        # 2268298 <_ZZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfiiE6mt_min>
  4746d1:	call   12e70 <__cxa_guard_release@plt>
  4746d6:	jmp    474417 <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x97>
  4746db:	lea    0x1df3b8e(%rip),%rbx        # 2268270 <_ZGVZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfiiE6cpu512>
  4746e2:	mov    %rbx,%rdi
  4746e5:	call   13d80 <__cxa_guard_acquire@plt>
  4746ea:	test   %eax,%eax
  4746ec:	je     474458 <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0xd8>
  4746f2:	call   46df40 <_ZN5guild7kernels3cpu13cpu_avx512_okEv>
  4746f7:	mov    %rbx,%rdi
  4746fa:	mov    %al,0x1df3b78(%rip)        # 2268278 <_ZZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfiiE6cpu512>
  474700:	call   12e70 <__cxa_guard_release@plt>
  474705:	jmp    474458 <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0xd8>
  47470a:	call   46de70 <_ZN5guild7kernels3cpu11cpu_avx2_okEv>
  47470f:	test   %al,%al
  474711:	jne    4747a1 <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x421>
  474717:	mov    %rbx,%rdi
  47471a:	mov    %al,0x1df3b88(%rip)        # 22682a8 <_ZZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfiiE4avx2>
  474720:	call   12e70 <__cxa_guard_release@plt>
  474725:	jmp    474408 <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x88>
  47472a:	call   46df40 <_ZN5guild7kernels3cpu13cpu_avx512_okEv>
  47472f:	test   %al,%al
  474731:	jne    47478d <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x40d>
  474733:	mov    %rbx,%rdi
  474736:	mov    %al,0x1df3b7c(%rip)        # 22682b8 <_ZZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfiiE6avx512>
  47473c:	call   12e70 <__cxa_guard_release@plt>
  474741:	jmp    4743e6 <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x66>
  474746:	call   490f00 <_ZN5guild7kernels3cpu15iq512_supportedEi>
  47474b:	test   %al,%al
  47474d:	jne    4747b8 <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x438>
  47474f:	mov    0x38(%rsp),%rax
  474754:	mov    (%rax),%edi
  474756:	jmp    47460b <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x28b>
  47475b:	mov    0x38(%rsp),%rax
  474760:	mov    0x30(%rax),%rcx
  474764:	mov    0x20(%rax),%rdx
  474768:	push   %r13
  47476a:	push   %r12
  47476c:	push   0x60(%rsp)
  474770:	push   %rbp
  474771:	mov    0x60(%rsp),%r9
  474776:	mov    0x68(%rsp),%rsi
  47477b:	mov    0x10(%rax),%r8d
  47477f:	call   4d6c90 <_ZN5guild7kernels3cpu13kq256_gu_rowsEiPKhmmiPKPKviPKPfii>
  474784:	add    $0x20,%rsp
  474788:	jmp    4745ae <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x22e>
  47478d:	lea    0x1c9138(%rip),%rdi        # 63d8cc <_IO_stdin_used+0x58cc>
  474794:	call   134d0 <getenv@plt>
  474799:	test   %rax,%rax
  47479c:	sete   %al
  47479f:	jmp    474733 <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x3b3>
  4747a1:	lea    0x1c9133(%rip),%rdi        # 63d8db <_IO_stdin_used+0x58db>
  4747a8:	call   134d0 <getenv@plt>
  4747ad:	test   %rax,%rax
  4747b0:	sete   %al
  4747b3:	jmp    474717 <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x397>
  4747b8:	test   %ebp,%ebp
  4747ba:	jle    4745ae <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x22e>
  4747c0:	lea    -0x1(%rbp),%eax
  4747c3:	mov    0x40(%rsp),%r15
  4747c8:	mov    0x50(%rsp),%r14
  4747cd:	lea    -0x8(%rbp),%ebx
  4747d0:	and    $0xfffffff8,%eax
  4747d3:	sub    %eax,%ebx
  4747d5:	mov    %ebx,0xc(%rsp)
  4747d9:	mov    %r15,%rbx
  4747dc:	mov    %r14,%r15
  4747df:	mov    0x38(%rsp),%r14
  4747e4:	mov    $0x8,%eax
  4747e9:	mov    0x30(%r14),%rcx
  4747ed:	mov    0x20(%r14),%rdx
  4747f1:	mov    %rbx,%r9
  4747f4:	cmp    %eax,%ebp
  4747f6:	mov    (%r14),%edi
  4747f9:	push   %r13
  4747fb:	cmovle %ebp,%eax
  4747fe:	push   %r12
  474800:	sub    $0x8,%ebp
  474803:	add    $0x40,%rbx
  474807:	push   %r15
  474809:	add    $0x40,%r15
  47480d:	push   %rax
  47480e:	mov    0x68(%rsp),%rsi
  474813:	mov    0x10(%r14),%r8d
  474817:	call   490f20 <_ZN5guild7kernels3cpu13iq512_gu_rowsEiPKhmmiPKPKviPKPfii>
  47481c:	add    $0x20,%rsp
  474820:	cmp    %ebp,0xc(%rsp)
  474824:	jne    4747e4 <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x464>
  474826:	jmp    4745ae <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x22e>
  47482b:	test   %ebp,%ebp
  47482d:	jle    4745ae <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x22e>
  474833:	lea    -0x1(%rbp),%eax
  474836:	mov    0x40(%rsp),%r15
  47483b:	mov    0x50(%rsp),%r14
  474840:	lea    -0x8(%rbp),%ebx
  474843:	and    $0xfffffff8,%eax
  474846:	sub    %eax,%ebx
  474848:	mov    %ebx,0xc(%rsp)
  47484c:	mov    %r15,%rbx
  47484f:	mov    %r14,%r15
  474852:	mov    0x38(%rsp),%r14
  474857:	mov    $0x8,%eax
  47485c:	mov    0x30(%r14),%rcx
  474860:	mov    0x20(%r14),%rdx
  474864:	mov    %rbx,%r9
  474867:	cmp    %eax,%ebp
  474869:	mov    (%r14),%edi
  47486c:	push   %r13
  47486e:	cmovle %ebp,%eax
  474871:	push   %r12
  474873:	sub    $0x8,%ebp
  474876:	add    $0x40,%rbx
  47487a:	push   %r15
  47487c:	add    $0x40,%r15
  474880:	push   %rax
  474881:	mov    0x68(%rsp),%rsi
  474886:	mov    0x10(%r14),%r8d
  47488a:	call   4bb110 <_ZN5guild7kernels3cpu13iq256_gu_rowsEiPKhmmiPKPKviPKPfii>
  47488f:	add    $0x20,%rsp
  474893:	cmp    %ebp,0xc(%rsp)
  474897:	jne    474857 <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x4d7>
  474899:	jmp    4745ae <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x22e>
  47489e:	call   13390 <__stack_chk_fail@plt>
  4748a3:	endbr64
  4748a7:	mov    %rax,%rbp
  4748aa:	jmp    2e879 <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii.cold>
  4748af:	endbr64
  4748b3:	mov    %rax,%rbp
  4748b6:	jmp    2e899 <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii.cold+0x20>
  4748bb:	endbr64
  4748bf:	mov    %rax,%rbp
  4748c2:	jmp    2e8be <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii.cold+0x45>
  4748c7:	endbr64
  4748cb:	mov    %rax,%rbp
  4748ce:	jmp    2e8de <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii.cold+0x65>

Disassembly of section .fini:
