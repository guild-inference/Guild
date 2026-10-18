
/home/ubuntu/Guild/build-cuda12/ornith_prefill_bench:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

000000000048e3f0 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii>:
  48e3f0:	endbr64
  48e3f4:	push   %r15
  48e3f6:	push   %r14
  48e3f8:	push   %r13
  48e3fa:	push   %r12
  48e3fc:	push   %rbp
  48e3fd:	mov    %r9d,%ebp
  48e400:	push   %rbx
  48e401:	movslq %ecx,%rbx
  48e404:	sub    $0x58,%rsp
  48e408:	mov    %rdi,0x18(%rsp)
  48e40d:	mov    0x90(%rsp),%r12d
  48e415:	mov    %rsi,0x28(%rsp)
  48e41a:	mov    %rdx,0x20(%rsp)
  48e41f:	mov    %r8,0x30(%rsp)
  48e424:	mov    %fs:0x28,%rax
  48e42d:	mov    %rax,0x48(%rsp)
  48e432:	xor    %eax,%eax
  48e434:	movzbl 0x1e221fd(%rip),%eax        # 22b0638 <_ZGVZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE8iq4nl_mt>
  48e43b:	test   %al,%al
  48e43d:	je     48e724 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x334>
  48e443:	movzbl 0x1e221de(%rip),%eax        # 22b0628 <_ZGVZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6mt_min>
  48e44a:	test   %al,%al
  48e44c:	je     48e6d5 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x2e5>
  48e452:	movzbl 0x1e221bf(%rip),%eax        # 22b0618 <_ZGVZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE2kq>
  48e459:	test   %al,%al
  48e45b:	je     48e67a <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x28a>
  48e461:	cmpb   $0x0,0x1e221b8(%rip)        # 22b0620 <_ZZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE2kq>
  48e468:	je     48e473 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x83>
  48e46a:	cmp    $0x1,%ebx
  48e46d:	jg     48e62c <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x23c>
  48e473:	call   4715c0 <_ZN5guild7kernels3cpu11cpu_avx2_okEv>
  48e478:	test   %al,%al
  48e47a:	mov    0x18(%rsp),%rax
  48e47f:	mov    0x4(%rax),%edi
  48e482:	je     48e490 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0xa0>
  48e484:	cmp    %ebx,0x1e221a6(%rip)        # 22b0630 <_ZZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6mt_min>
  48e48a:	jle    48e59a <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x1aa>
  48e490:	call   4f2cb0 <ggml_get_type_traits_cpu>
  48e495:	mov    0x18(%rsp),%rcx
  48e49a:	mov    0x18(%rcx),%r15d
  48e49e:	cmp    %r12d,%ebp
  48e4a1:	jge    48e577 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x187>
  48e4a7:	test   %ebx,%ebx
  48e4a9:	jle    48e577 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x187>
  48e4af:	movslq %ebp,%rsi
  48e4b2:	sub    %ebp,%r12d
  48e4b5:	mov    0x8(%rax),%r14
  48e4b9:	lea    (%r12,%rsi,1),%rax
  48e4bd:	mov    %rax,0x38(%rsp)
  48e4c2:	mov    0x20(%rsp),%rax
  48e4c7:	lea    (%rax,%rbx,8),%rax
  48e4cb:	mov    %rax,(%rsp)
  48e4cf:	lea    0x44(%rsp),%rax
  48e4d4:	mov    %rax,0x8(%rsp)
  48e4d9:	nopl   0x0(%rax)
  48e4e0:	mov    0x18(%rsp),%rax
  48e4e5:	mov    0x20(%rsp),%r12
  48e4ea:	mov    %rsi,0x10(%rsp)
  48e4ef:	lea    0x0(,%rsi,4),%rbp
  48e4f7:	mov    0x30(%rsp),%r13
  48e4fc:	mov    0x28(%rax),%rbx
  48e500:	imul   %rsi,%rbx
  48e504:	add    0x38(%rax),%rbx
  48e508:	mov    0x28(%rsp),%rax
  48e50d:	add    %rax,%rbx
  48e510:	mov    %rbp,%rax
  48e513:	mov    %rbx,%rbp
  48e516:	mov    %r12,%rbx
  48e519:	mov    %rax,%r12
  48e51c:	nopl   0x0(%rax)
  48e520:	movl   $0x0,0x44(%rsp)
  48e528:	xor    %edx,%edx
  48e52a:	xor    %r8d,%r8d
  48e52d:	mov    %rbp,%rcx
  48e530:	push   $0x1
  48e532:	mov    %r15d,%edi
  48e535:	add    $0x8,%rbx
  48e539:	add    $0x8,%r13
  48e53d:	push   $0x0
  48e53f:	mov    0x18(%rsp),%rsi
  48e544:	mov    -0x8(%rbx),%r9
  48e548:	call   *%r14
  48e54b:	movss  0x54(%rsp),%xmm0
  48e551:	mov    -0x8(%r13),%rdx
  48e555:	movss  %xmm0,(%rdx,%r12,1)
  48e55b:	pop    %rax
  48e55c:	pop    %rdx
  48e55d:	cmp    %rbx,(%rsp)
  48e561:	jne    48e520 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x130>
  48e563:	mov    0x10(%rsp),%rsi
  48e568:	add    $0x1,%rsi
  48e56c:	cmp    %rsi,0x38(%rsp)
  48e571:	jne    48e4e0 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0xf0>
  48e577:	mov    0x48(%rsp),%rax
  48e57c:	sub    %fs:0x28,%rax
  48e585:	jne    48e75e <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x36e>
  48e58b:	add    $0x58,%rsp
  48e58f:	pop    %rbx
  48e590:	pop    %rbp
  48e591:	pop    %r12
  48e593:	pop    %r13
  48e595:	pop    %r14
  48e597:	pop    %r15
  48e599:	ret
  48e59a:	cmp    $0x14,%edi
  48e59d:	jne    48e490 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0xa0>
  48e5a3:	cmpb   $0x0,0x1e22096(%rip)        # 22b0640 <_ZZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE8iq4nl_mt>
  48e5aa:	je     48e490 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0xa0>
  48e5b0:	test   %ebx,%ebx
  48e5b2:	jle    48e577 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x187>
  48e5b4:	mov    0x20(%rsp),%r15
  48e5b9:	lea    -0x1(%rbx),%edx
  48e5bc:	lea    -0x8(%rbx),%r13d
  48e5c0:	mov    %r12d,0x90(%rsp)
  48e5c8:	and    $0xfffffff8,%edx
  48e5cb:	mov    %ebp,%r12d
  48e5ce:	mov    0x30(%rsp),%r14
  48e5d3:	sub    %edx,%r13d
  48e5d6:	mov    %r15,%rbp
  48e5d9:	mov    0x90(%rsp),%r15d
  48e5e1:	mov    %r13d,(%rsp)
  48e5e5:	mov    %rax,%r13
  48e5e8:	mov    $0x8,%r8d
  48e5ee:	mov    0x28(%r13),%rsi
  48e5f2:	mov    %rbp,%rcx
  48e5f5:	mov    %r14,%r9
  48e5f8:	mov    0x28(%rsp),%rdi
  48e5fd:	add    0x38(%r13),%rdi
  48e601:	push   %r15
  48e603:	cmp    %r8d,%ebx
  48e606:	push   %r12
  48e608:	cmovle %ebx,%r8d
  48e60c:	mov    0x18(%r13),%edx
  48e610:	sub    $0x8,%ebx
  48e613:	add    $0x40,%rbp
  48e617:	add    $0x40,%r14
  48e61b:	call   4d6490 <_ZN5guild7kernels3cpu18iq4nl256_down_rowsEPKhmiPKPKviPKPfii>
  48e620:	pop    %rcx
  48e621:	pop    %rsi
  48e622:	cmp    %ebx,(%rsp)
  48e625:	jne    48e5e8 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x1f8>
  48e627:	jmp    48e577 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x187>
  48e62c:	mov    0x18(%rsp),%rax
  48e631:	mov    0x4(%rax),%edi
  48e634:	call   4eff90 <_ZN5guild7kernels3cpu15kq256_supportedEi>
  48e639:	test   %al,%al
  48e63b:	je     48e473 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x83>
  48e641:	mov    0x18(%rsp),%rax
  48e646:	mov    0x28(%rsp),%rsi
  48e64b:	sub    $0x8,%rsp
  48e64f:	mov    %ebx,%r9d
  48e652:	mov    0x28(%rax),%rdx
  48e656:	mov    0x4(%rax),%edi
  48e659:	add    0x38(%rax),%rsi
  48e65d:	push   %r12
  48e65f:	push   %rbp
  48e660:	push   0x48(%rsp)
  48e664:	mov    0x40(%rsp),%r8
  48e669:	mov    0x18(%rax),%ecx
  48e66c:	call   4f0990 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii>
  48e671:	add    $0x20,%rsp
  48e675:	jmp    48e577 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x187>
  48e67a:	lea    0x1e21f97(%rip),%r13        # 22b0618 <_ZGVZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE2kq>
  48e681:	mov    %r13,%rdi
  48e684:	call   14e00 <__cxa_guard_acquire@plt>
  48e689:	test   %eax,%eax
  48e68b:	je     48e461 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x71>
  48e691:	lea    0x1c95c3(%rip),%rdi        # 657c5b <_IO_stdin_used+0x5c5b>
  48e698:	call   14540 <getenv@plt>
  48e69d:	mov    %rax,%r14
  48e6a0:	call   4715c0 <_ZN5guild7kernels3cpu11cpu_avx2_okEv>
  48e6a5:	test   %al,%al
  48e6a7:	je     48e6c2 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x2d2>
  48e6a9:	test   %r14,%r14
  48e6ac:	je     48e6c2 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x2d2>
  48e6ae:	xor    %esi,%esi
  48e6b0:	mov    $0xa,%edx
  48e6b5:	mov    %r14,%rdi
  48e6b8:	call   15000 <__isoc23_strtol@plt>
  48e6bd:	test   %eax,%eax
  48e6bf:	setne  %al
  48e6c2:	mov    %r13,%rdi
  48e6c5:	mov    %al,0x1e21f55(%rip)        # 22b0620 <_ZZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE2kq>
  48e6cb:	call   13eb0 <__cxa_guard_release@plt>
  48e6d0:	jmp    48e461 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x71>
  48e6d5:	lea    0x1e21f4c(%rip),%r13        # 22b0628 <_ZGVZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6mt_min>
  48e6dc:	mov    %r13,%rdi
  48e6df:	call   14e00 <__cxa_guard_acquire@plt>
  48e6e4:	test   %eax,%eax
  48e6e6:	je     48e452 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x62>
  48e6ec:	lea    0x1c9558(%rip),%rdi        # 657c4b <_IO_stdin_used+0x5c4b>
  48e6f3:	call   14540 <getenv@plt>
  48e6f8:	mov    %rax,%rdi
  48e6fb:	mov    $0x2,%eax
  48e700:	test   %rdi,%rdi
  48e703:	je     48e711 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x321>
  48e705:	mov    $0xa,%edx
  48e70a:	xor    %esi,%esi
  48e70c:	call   15000 <__isoc23_strtol@plt>
  48e711:	mov    %r13,%rdi
  48e714:	mov    %eax,0x1e21f16(%rip)        # 22b0630 <_ZZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6mt_min>
  48e71a:	call   13eb0 <__cxa_guard_release@plt>
  48e71f:	jmp    48e452 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x62>
  48e724:	lea    0x1e21f0d(%rip),%r13        # 22b0638 <_ZGVZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE8iq4nl_mt>
  48e72b:	mov    %r13,%rdi
  48e72e:	call   14e00 <__cxa_guard_acquire@plt>
  48e733:	test   %eax,%eax
  48e735:	je     48e443 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x53>
  48e73b:	lea    0x1c9525(%rip),%rdi        # 657c67 <_IO_stdin_used+0x5c67>
  48e742:	call   14540 <getenv@plt>
  48e747:	mov    %r13,%rdi
  48e74a:	test   %rax,%rax
  48e74d:	sete   0x1e21eec(%rip)        # 22b0640 <_ZZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE8iq4nl_mt>
  48e754:	call   13eb0 <__cxa_guard_release@plt>
  48e759:	jmp    48e443 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x53>
  48e75e:	call   143e0 <__stack_chk_fail@plt>
  48e763:	endbr64
  48e767:	mov    %rax,%rbx
  48e76a:	jmp    39770 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii.cold>

Disassembly of section .fini:
