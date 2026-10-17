
/home/ubuntu/guild-kernels-cpu/build/libguild_cpu_plugin.so:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

000000000001e6e0 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii>:
   1e6e0:	endbr64
   1e6e4:	push   %r15
   1e6e6:	push   %r14
   1e6e8:	push   %r13
   1e6ea:	push   %r12
   1e6ec:	mov    %r9d,%r12d
   1e6ef:	push   %rbp
   1e6f0:	movslq %ecx,%rbp
   1e6f3:	push   %rbx
   1e6f4:	sub    $0x78,%rsp
   1e6f8:	mov    %rdi,0x38(%rsp)
   1e6fd:	mov    0xb0(%rsp),%r13d
   1e705:	mov    %rsi,0x48(%rsp)
   1e70a:	mov    %rdx,0x40(%rsp)
   1e70f:	mov    %r8,0x50(%rsp)
   1e714:	mov    %fs:0x28,%rax
   1e71d:	mov    %rax,0x68(%rsp)
   1e722:	xor    %eax,%eax
   1e724:	movzbl 0x21cfb5(%rip),%eax        # 23b6e0 <_ZGVZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6avx512>
   1e72b:	test   %al,%al
   1e72d:	jne    1e746 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x66>
   1e72f:	lea    0x21cfaa(%rip),%rbx        # 23b6e0 <_ZGVZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6avx512>
   1e736:	mov    %rbx,%rdi
   1e739:	call   1ce40 <__cxa_guard_acquire@plt>
   1e73e:	test   %eax,%eax
   1e740:	jne    1ea8a <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x3aa>
   1e746:	movzbl 0x21cf83(%rip),%eax        # 23b6d0 <_ZGVZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE4avx2>
   1e74d:	test   %al,%al
   1e74f:	jne    1e768 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x88>
   1e751:	lea    0x21cf78(%rip),%rbx        # 23b6d0 <_ZGVZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE4avx2>
   1e758:	mov    %rbx,%rdi
   1e75b:	call   1ce40 <__cxa_guard_acquire@plt>
   1e760:	test   %eax,%eax
   1e762:	jne    1ea6a <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x38a>
   1e768:	movzbl 0x21cf51(%rip),%eax        # 23b6c0 <_ZGVZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6mt_min>
   1e76f:	test   %al,%al
   1e771:	je     1e9ec <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x30c>
   1e777:	movzbl 0x21cf32(%rip),%eax        # 23b6b0 <_ZGVZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE2kq>
   1e77e:	test   %al,%al
   1e780:	je     1e991 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x2b1>
   1e786:	cmpb   $0x0,0x21cf2b(%rip)        # 23b6b8 <_ZZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE2kq>
   1e78d:	je     1e7a9 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0xc9>
   1e78f:	mov    0x38(%rsp),%rax
   1e794:	mov    (%rax),%edi
   1e796:	mov    %edi,%eax
   1e798:	and    $0xfffffffd,%eax
   1e79b:	cmp    $0xc,%eax
   1e79e:	jne    1e7a9 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0xc9>
   1e7a0:	cmp    $0x1,%ebp
   1e7a3:	jg     1eabb <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x3db>
   1e7a9:	movzbl 0x21cef0(%rip),%eax        # 23b6a0 <_ZGVZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6cpu512>
   1e7b0:	test   %al,%al
   1e7b2:	je     1ea3b <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x35b>
   1e7b8:	mov    0x38(%rsp),%rax
   1e7bd:	cmp    %ebp,0x21cf05(%rip)        # 23b6c8 <_ZZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6mt_min>
   1e7c3:	mov    (%rax),%edi
   1e7c5:	jle    1e931 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x251>
   1e7cb:	call   9e6d0 <ggml_get_type_traits_cpu>
   1e7d0:	mov    0x38(%rsp),%rdi
   1e7d5:	mov    0x10(%rdi),%r14d
   1e7d9:	cmp    %r13d,%r12d
   1e7dc:	jge    1e90e <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x22e>
   1e7e2:	test   %ebp,%ebp
   1e7e4:	jle    1e90e <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x22e>
   1e7ea:	movslq %r12d,%rsi
   1e7ed:	sub    %r12d,%r13d
   1e7f0:	mov    0x8(%rax),%rbx
   1e7f4:	lea    0x0(%r13,%rsi,1),%rax
   1e7f9:	mov    %rax,0x58(%rsp)
   1e7fe:	mov    0x40(%rsp),%rax
   1e803:	lea    (%rax,%rbp,8),%rax
   1e807:	mov    %rax,0x18(%rsp)
   1e80c:	lea    0x60(%rsp),%rax
   1e811:	mov    %rax,0x20(%rsp)
   1e816:	lea    0x64(%rsp),%rax
   1e81b:	mov    %rax,0x28(%rsp)
   1e820:	mov    0x38(%rsp),%rax
   1e825:	mov    0x48(%rsp),%rcx
   1e82a:	mov    %rsi,0x30(%rsp)
   1e82f:	lea    0x0(,%rsi,4),%r12
   1e837:	mov    0x40(%rsp),%r15
   1e83c:	mov    0x20(%rax),%rbp
   1e840:	imul   %rsi,%rbp
   1e844:	lea    (%rcx,%rbp,1),%r13
   1e848:	add    0x30(%rax),%rbp
   1e84c:	lea    (%rcx,%rbp,1),%rax
   1e850:	mov    0x50(%rsp),%rbp
   1e855:	mov    %rax,0x10(%rsp)
   1e85a:	nopw   0x0(%rax,%rax,1)
   1e860:	movl   $0x0,0x60(%rsp)
   1e868:	xor    %r8d,%r8d
   1e86b:	mov    %r13,%rcx
   1e86e:	xor    %edx,%edx
   1e870:	movl   $0x0,0x64(%rsp)
   1e878:	mov    %r14d,%edi
   1e87b:	add    $0x8,%r15
   1e87f:	add    $0x8,%rbp
   1e883:	push   $0x1
   1e885:	push   $0x0
   1e887:	mov    0x30(%rsp),%rsi
   1e88c:	mov    -0x8(%r15),%r9
   1e890:	call   *%rbx
   1e892:	pop    %rax
   1e893:	xor    %r8d,%r8d
   1e896:	pop    %rdx
   1e897:	mov    %r14d,%edi
   1e89a:	push   $0x1
   1e89c:	xor    %edx,%edx
   1e89e:	push   $0x0
   1e8a0:	mov    0x20(%rsp),%rcx
   1e8a5:	mov    0x38(%rsp),%rsi
   1e8aa:	mov    -0x8(%r15),%r9
   1e8ae:	call   *%rbx
   1e8b0:	movss  0x70(%rsp),%xmm1
   1e8b6:	movaps %xmm1,%xmm0
   1e8b9:	xorps  0x1d3880(%rip),%xmm0        # 1f2140 <_fini+0x9904>
   1e8c0:	movss  %xmm1,0x1c(%rsp)
   1e8c6:	call   1cda0 <expf@plt>
   1e8cb:	movss  0x1c(%rsp),%xmm1
   1e8d1:	mov    -0x8(%rbp),%rdx
   1e8d5:	addss  0x1d3f13(%rip),%xmm0        # 1f27f0 <_fini+0x9fb4>
   1e8dd:	divss  %xmm0,%xmm1
   1e8e1:	mulss  0x74(%rsp),%xmm1
   1e8e7:	movss  %xmm1,(%rdx,%r12,1)
   1e8ed:	pop    %rcx
   1e8ee:	pop    %rsi
   1e8ef:	cmp    %r15,0x18(%rsp)
   1e8f4:	jne    1e860 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x180>
   1e8fa:	mov    0x30(%rsp),%rsi
   1e8ff:	add    $0x1,%rsi
   1e903:	cmp    %rsi,0x58(%rsp)
   1e908:	jne    1e820 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x140>
   1e90e:	mov    0x68(%rsp),%rax
   1e913:	sub    %fs:0x28,%rax
   1e91c:	jne    1ebfe <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x51e>
   1e922:	add    $0x78,%rsp
   1e926:	pop    %rbx
   1e927:	pop    %rbp
   1e928:	pop    %r12
   1e92a:	pop    %r13
   1e92c:	pop    %r14
   1e92e:	pop    %r15
   1e930:	ret
   1e931:	call   3b260 <_ZN5guild7kernels3cpu15iq512_supportedEi>
   1e936:	test   %al,%al
   1e938:	jne    1e957 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x277>
   1e93a:	mov    0x38(%rsp),%rax
   1e93f:	cmpb   $0x0,0x21cd62(%rip)        # 23b6a8 <_ZZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6cpu512>
   1e946:	mov    (%rax),%edi
   1e948:	jne    1e7cb <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0xeb>
   1e94e:	call   65450 <_ZN5guild7kernels3cpu15iq256_supportedEi>
   1e953:	test   %al,%al
   1e955:	je     1e985 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x2a5>
   1e957:	mov    0x38(%rsp),%rax
   1e95c:	cmpb   $0x0,0x21cd85(%rip)        # 23b6e8 <_ZZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6avx512>
   1e963:	mov    (%rax),%edi
   1e965:	jne    1eaa6 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x3c6>
   1e96b:	cmpb   $0x0,0x21cd66(%rip)        # 23b6d8 <_ZZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE4avx2>
   1e972:	je     1e7cb <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0xeb>
   1e978:	call   65450 <_ZN5guild7kernels3cpu15iq256_supportedEi>
   1e97d:	test   %al,%al
   1e97f:	jne    1eb8b <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x4ab>
   1e985:	mov    0x38(%rsp),%rax
   1e98a:	mov    (%rax),%edi
   1e98c:	jmp    1e7cb <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0xeb>
   1e991:	lea    0x21cd18(%rip),%rbx        # 23b6b0 <_ZGVZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE2kq>
   1e998:	mov    %rbx,%rdi
   1e99b:	call   1ce40 <__cxa_guard_acquire@plt>
   1e9a0:	test   %eax,%eax
   1e9a2:	je     1e786 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0xa6>
   1e9a8:	lea    0x1ca6e1(%rip),%rdi        # 1e9090 <_fini+0x854>
   1e9af:	call   1cb90 <getenv@plt>
   1e9b4:	mov    %rax,%r14
   1e9b7:	call   81b10 <_ZN5guild7kernels3cpu11cpu_avx2_okEv>
   1e9bc:	test   %al,%al
   1e9be:	je     1e9d9 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x2f9>
   1e9c0:	test   %r14,%r14
   1e9c3:	je     1e9d9 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x2f9>
   1e9c5:	xor    %esi,%esi
   1e9c7:	mov    $0xa,%edx
   1e9cc:	mov    %r14,%rdi
   1e9cf:	call   1ced0 <__isoc23_strtol@plt>
   1e9d4:	test   %eax,%eax
   1e9d6:	setne  %al
   1e9d9:	mov    %rbx,%rdi
   1e9dc:	mov    %al,0x21ccd6(%rip)        # 23b6b8 <_ZZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE2kq>
   1e9e2:	call   1c900 <__cxa_guard_release@plt>
   1e9e7:	jmp    1e786 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0xa6>
   1e9ec:	lea    0x21cccd(%rip),%rbx        # 23b6c0 <_ZGVZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6mt_min>
   1e9f3:	mov    %rbx,%rdi
   1e9f6:	call   1ce40 <__cxa_guard_acquire@plt>
   1e9fb:	test   %eax,%eax
   1e9fd:	je     1e777 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x97>
   1ea03:	lea    0x1ca676(%rip),%rdi        # 1e9080 <_fini+0x844>
   1ea0a:	call   1cb90 <getenv@plt>
   1ea0f:	mov    %rax,%rdi
   1ea12:	mov    $0x2,%eax
   1ea17:	test   %rdi,%rdi
   1ea1a:	je     1ea28 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x348>
   1ea1c:	mov    $0xa,%edx
   1ea21:	xor    %esi,%esi
   1ea23:	call   1ced0 <__isoc23_strtol@plt>
   1ea28:	mov    %rbx,%rdi
   1ea2b:	mov    %eax,0x21cc97(%rip)        # 23b6c8 <_ZZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6mt_min>
   1ea31:	call   1c900 <__cxa_guard_release@plt>
   1ea36:	jmp    1e777 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x97>
   1ea3b:	lea    0x21cc5e(%rip),%rbx        # 23b6a0 <_ZGVZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6cpu512>
   1ea42:	mov    %rbx,%rdi
   1ea45:	call   1ce40 <__cxa_guard_acquire@plt>
   1ea4a:	test   %eax,%eax
   1ea4c:	je     1e7b8 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0xd8>
   1ea52:	call   81950 <_ZN5guild7kernels3cpu13cpu_avx512_okEv>
   1ea57:	mov    %rbx,%rdi
   1ea5a:	mov    %al,0x21cc48(%rip)        # 23b6a8 <_ZZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6cpu512>
   1ea60:	call   1c900 <__cxa_guard_release@plt>
   1ea65:	jmp    1e7b8 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0xd8>
   1ea6a:	call   81b10 <_ZN5guild7kernels3cpu11cpu_avx2_okEv>
   1ea6f:	test   %al,%al
   1ea71:	jne    1eb01 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x421>
   1ea77:	mov    %rbx,%rdi
   1ea7a:	mov    %al,0x21cc58(%rip)        # 23b6d8 <_ZZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE4avx2>
   1ea80:	call   1c900 <__cxa_guard_release@plt>
   1ea85:	jmp    1e768 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x88>
   1ea8a:	call   81950 <_ZN5guild7kernels3cpu13cpu_avx512_okEv>
   1ea8f:	test   %al,%al
   1ea91:	jne    1eaed <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x40d>
   1ea93:	mov    %rbx,%rdi
   1ea96:	mov    %al,0x21cc4c(%rip)        # 23b6e8 <_ZZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6avx512>
   1ea9c:	call   1c900 <__cxa_guard_release@plt>
   1eaa1:	jmp    1e746 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x66>
   1eaa6:	call   3b260 <_ZN5guild7kernels3cpu15iq512_supportedEi>
   1eaab:	test   %al,%al
   1eaad:	jne    1eb18 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x438>
   1eaaf:	mov    0x38(%rsp),%rax
   1eab4:	mov    (%rax),%edi
   1eab6:	jmp    1e96b <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x28b>
   1eabb:	mov    0x38(%rsp),%rax
   1eac0:	mov    0x30(%rax),%rcx
   1eac4:	mov    0x20(%rax),%rdx
   1eac8:	push   %r13
   1eaca:	push   %r12
   1eacc:	push   0x60(%rsp)
   1ead0:	push   %rbp
   1ead1:	mov    0x60(%rsp),%r9
   1ead6:	mov    0x68(%rsp),%rsi
   1eadb:	mov    0x10(%rax),%r8d
   1eadf:	call   80ff0 <_ZN5guild7kernels3cpu13kq256_gu_rowsEiPKhmmiPKPKviPKPfii>
   1eae4:	add    $0x20,%rsp
   1eae8:	jmp    1e90e <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x22e>
   1eaed:	lea    0x1ca56e(%rip),%rdi        # 1e9062 <_fini+0x826>
   1eaf4:	call   1cb90 <getenv@plt>
   1eaf9:	test   %rax,%rax
   1eafc:	sete   %al
   1eaff:	jmp    1ea93 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x3b3>
   1eb01:	lea    0x1ca569(%rip),%rdi        # 1e9071 <_fini+0x835>
   1eb08:	call   1cb90 <getenv@plt>
   1eb0d:	test   %rax,%rax
   1eb10:	sete   %al
   1eb13:	jmp    1ea77 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x397>
   1eb18:	test   %ebp,%ebp
   1eb1a:	jle    1e90e <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x22e>
   1eb20:	lea    -0x1(%rbp),%eax
   1eb23:	mov    0x40(%rsp),%r15
   1eb28:	mov    0x50(%rsp),%r14
   1eb2d:	lea    -0x8(%rbp),%ebx
   1eb30:	and    $0xfffffff8,%eax
   1eb33:	sub    %eax,%ebx
   1eb35:	mov    %ebx,0xc(%rsp)
   1eb39:	mov    %r15,%rbx
   1eb3c:	mov    %r14,%r15
   1eb3f:	mov    0x38(%rsp),%r14
   1eb44:	mov    $0x8,%eax
   1eb49:	mov    0x30(%r14),%rcx
   1eb4d:	mov    0x20(%r14),%rdx
   1eb51:	mov    %rbx,%r9
   1eb54:	cmp    %eax,%ebp
   1eb56:	mov    (%r14),%edi
   1eb59:	push   %r13
   1eb5b:	cmovle %ebp,%eax
   1eb5e:	push   %r12
   1eb60:	sub    $0x8,%ebp
   1eb63:	add    $0x40,%rbx
   1eb67:	push   %r15
   1eb69:	add    $0x40,%r15
   1eb6d:	push   %rax
   1eb6e:	mov    0x68(%rsp),%rsi
   1eb73:	mov    0x10(%r14),%r8d
   1eb77:	call   3b280 <_ZN5guild7kernels3cpu13iq512_gu_rowsEiPKhmmiPKPKviPKPfii>
   1eb7c:	add    $0x20,%rsp
   1eb80:	cmp    %ebp,0xc(%rsp)
   1eb84:	jne    1eb44 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x464>
   1eb86:	jmp    1e90e <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x22e>
   1eb8b:	test   %ebp,%ebp
   1eb8d:	jle    1e90e <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x22e>
   1eb93:	lea    -0x1(%rbp),%eax
   1eb96:	mov    0x40(%rsp),%r15
   1eb9b:	mov    0x50(%rsp),%r14
   1eba0:	lea    -0x8(%rbp),%ebx
   1eba3:	and    $0xfffffff8,%eax
   1eba6:	sub    %eax,%ebx
   1eba8:	mov    %ebx,0xc(%rsp)
   1ebac:	mov    %r15,%rbx
   1ebaf:	mov    %r14,%r15
   1ebb2:	mov    0x38(%rsp),%r14
   1ebb7:	mov    $0x8,%eax
   1ebbc:	mov    0x30(%r14),%rcx
   1ebc0:	mov    0x20(%r14),%rdx
   1ebc4:	mov    %rbx,%r9
   1ebc7:	cmp    %eax,%ebp
   1ebc9:	mov    (%r14),%edi
   1ebcc:	push   %r13
   1ebce:	cmovle %ebp,%eax
   1ebd1:	push   %r12
   1ebd3:	sub    $0x8,%ebp
   1ebd6:	add    $0x40,%rbx
   1ebda:	push   %r15
   1ebdc:	add    $0x40,%r15
   1ebe0:	push   %rax
   1ebe1:	mov    0x68(%rsp),%rsi
   1ebe6:	mov    0x10(%r14),%r8d
   1ebea:	call   65470 <_ZN5guild7kernels3cpu13iq256_gu_rowsEiPKhmmiPKPKviPKPfii>
   1ebef:	add    $0x20,%rsp
   1ebf3:	cmp    %ebp,0xc(%rsp)
   1ebf7:	jne    1ebb7 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x4d7>
   1ebf9:	jmp    1e90e <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x22e>
   1ebfe:	call   1cae0 <__stack_chk_fail@plt>
   1ec03:	endbr64
   1ec07:	mov    %rax,%rbp
   1ec0a:	jmp    1cf45 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii.cold>
   1ec0f:	endbr64
   1ec13:	mov    %rax,%rbp
   1ec16:	jmp    1cf65 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii.cold+0x20>
   1ec1b:	endbr64
   1ec1f:	mov    %rax,%rbp
   1ec22:	jmp    1cf8a <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii.cold+0x45>
   1ec27:	endbr64
   1ec2b:	mov    %rax,%rbp
   1ec2e:	jmp    1cfaa <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii.cold+0x65>

Disassembly of section .fini:
