
bench/results/external-kernels/phase1/ornith_prefill_bench:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

00000000004748e0 <_ZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii>:
  4748e0:	endbr64
  4748e4:	push   %r15
  4748e6:	push   %r14
  4748e8:	push   %r13
  4748ea:	push   %r12
  4748ec:	push   %rbp
  4748ed:	mov    %r9d,%ebp
  4748f0:	push   %rbx
  4748f1:	movslq %ecx,%rbx
  4748f4:	sub    $0x58,%rsp
  4748f8:	mov    %rdi,0x18(%rsp)
  4748fd:	mov    0x90(%rsp),%r12d
  474905:	mov    %rsi,0x28(%rsp)
  47490a:	mov    %rdx,0x20(%rsp)
  47490f:	mov    %r8,0x30(%rsp)
  474914:	mov    %fs:0x28,%rax
  47491d:	mov    %rax,0x48(%rsp)
  474922:	xor    %eax,%eax
  474924:	movzbl 0x1df3935(%rip),%eax        # 2268260 <_ZGVZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfiiE8iq4nl_mt>
  47492b:	test   %al,%al
  47492d:	je     474c14 <_ZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x334>
  474933:	movzbl 0x1df3916(%rip),%eax        # 2268250 <_ZGVZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfiiE6mt_min>
  47493a:	test   %al,%al
  47493c:	je     474bc5 <_ZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x2e5>
  474942:	movzbl 0x1df38f7(%rip),%eax        # 2268240 <_ZGVZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfiiE2kq>
  474949:	test   %al,%al
  47494b:	je     474b6a <_ZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x28a>
  474951:	cmpb   $0x0,0x1df38f0(%rip)        # 2268248 <_ZZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfiiE2kq>
  474958:	je     474963 <_ZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x83>
  47495a:	cmp    $0x1,%ebx
  47495d:	jg     474b1c <_ZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x23c>
  474963:	call   46de70 <_ZN5guild7kernels3cpu11cpu_avx2_okEv>
  474968:	test   %al,%al
  47496a:	mov    0x18(%rsp),%rax
  47496f:	mov    0x4(%rax),%edi
  474972:	je     474980 <_ZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0xa0>
  474974:	cmp    %ebx,0x1df38de(%rip)        # 2268258 <_ZZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfiiE6mt_min>
  47497a:	jle    474a8a <_ZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x1aa>
  474980:	call   4d8970 <ggml_get_type_traits_cpu>
  474985:	mov    0x18(%rsp),%rcx
  47498a:	mov    0x18(%rcx),%r15d
  47498e:	cmp    %r12d,%ebp
  474991:	jge    474a67 <_ZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x187>
  474997:	test   %ebx,%ebx
  474999:	jle    474a67 <_ZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x187>
  47499f:	movslq %ebp,%rsi
  4749a2:	sub    %ebp,%r12d
  4749a5:	mov    0x8(%rax),%r14
  4749a9:	lea    (%r12,%rsi,1),%rax
  4749ad:	mov    %rax,0x38(%rsp)
  4749b2:	mov    0x20(%rsp),%rax
  4749b7:	lea    (%rax,%rbx,8),%rax
  4749bb:	mov    %rax,(%rsp)
  4749bf:	lea    0x44(%rsp),%rax
  4749c4:	mov    %rax,0x8(%rsp)
  4749c9:	nopl   0x0(%rax)
  4749d0:	mov    0x18(%rsp),%rax
  4749d5:	mov    0x20(%rsp),%r12
  4749da:	mov    %rsi,0x10(%rsp)
  4749df:	lea    0x0(,%rsi,4),%rbp
  4749e7:	mov    0x30(%rsp),%r13
  4749ec:	mov    0x28(%rax),%rbx
  4749f0:	imul   %rsi,%rbx
  4749f4:	add    0x38(%rax),%rbx
  4749f8:	mov    0x28(%rsp),%rax
  4749fd:	add    %rax,%rbx
  474a00:	mov    %rbp,%rax
  474a03:	mov    %rbx,%rbp
  474a06:	mov    %r12,%rbx
  474a09:	mov    %rax,%r12
  474a0c:	nopl   0x0(%rax)
  474a10:	movl   $0x0,0x44(%rsp)
  474a18:	xor    %edx,%edx
  474a1a:	xor    %r8d,%r8d
  474a1d:	mov    %rbp,%rcx
  474a20:	push   $0x1
  474a22:	mov    %r15d,%edi
  474a25:	add    $0x8,%rbx
  474a29:	add    $0x8,%r13
  474a2d:	push   $0x0
  474a2f:	mov    0x18(%rsp),%rsi
  474a34:	mov    -0x8(%rbx),%r9
  474a38:	call   *%r14
  474a3b:	movss  0x54(%rsp),%xmm0
  474a41:	mov    -0x8(%r13),%rdx
  474a45:	movss  %xmm0,(%rdx,%r12,1)
  474a4b:	pop    %rax
  474a4c:	pop    %rdx
  474a4d:	cmp    %rbx,(%rsp)
  474a51:	jne    474a10 <_ZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x130>
  474a53:	mov    0x10(%rsp),%rsi
  474a58:	add    $0x1,%rsi
  474a5c:	cmp    %rsi,0x38(%rsp)
  474a61:	jne    4749d0 <_ZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0xf0>
  474a67:	mov    0x48(%rsp),%rax
  474a6c:	sub    %fs:0x28,%rax
  474a75:	jne    474c4e <_ZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x36e>
  474a7b:	add    $0x58,%rsp
  474a7f:	pop    %rbx
  474a80:	pop    %rbp
  474a81:	pop    %r12
  474a83:	pop    %r13
  474a85:	pop    %r14
  474a87:	pop    %r15
  474a89:	ret
  474a8a:	cmp    $0x14,%edi
  474a8d:	jne    474980 <_ZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0xa0>
  474a93:	cmpb   $0x0,0x1df37ce(%rip)        # 2268268 <_ZZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfiiE8iq4nl_mt>
  474a9a:	je     474980 <_ZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0xa0>
  474aa0:	test   %ebx,%ebx
  474aa2:	jle    474a67 <_ZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x187>
  474aa4:	mov    0x20(%rsp),%r15
  474aa9:	lea    -0x1(%rbx),%edx
  474aac:	lea    -0x8(%rbx),%r13d
  474ab0:	mov    %r12d,0x90(%rsp)
  474ab8:	and    $0xfffffff8,%edx
  474abb:	mov    %ebp,%r12d
  474abe:	mov    0x30(%rsp),%r14
  474ac3:	sub    %edx,%r13d
  474ac6:	mov    %r15,%rbp
  474ac9:	mov    0x90(%rsp),%r15d
  474ad1:	mov    %r13d,(%rsp)
  474ad5:	mov    %rax,%r13
  474ad8:	mov    $0x8,%r8d
  474ade:	mov    0x28(%r13),%rsi
  474ae2:	mov    %rbp,%rcx
  474ae5:	mov    %r14,%r9
  474ae8:	mov    0x28(%rsp),%rdi
  474aed:	add    0x38(%r13),%rdi
  474af1:	push   %r15
  474af3:	cmp    %r8d,%ebx
  474af6:	push   %r12
  474af8:	cmovle %ebx,%r8d
  474afc:	mov    0x18(%r13),%edx
  474b00:	sub    $0x8,%ebx
  474b03:	add    $0x40,%rbp
  474b07:	add    $0x40,%r14
  474b0b:	call   4bc980 <_ZN5guild7kernels3cpu18iq4nl256_down_rowsEPKhmiPKPKviPKPfii>
  474b10:	pop    %rcx
  474b11:	pop    %rsi
  474b12:	cmp    %ebx,(%rsp)
  474b15:	jne    474ad8 <_ZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x1f8>
  474b17:	jmp    474a67 <_ZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x187>
  474b1c:	mov    0x18(%rsp),%rax
  474b21:	mov    0x4(%rax),%edi
  474b24:	call   4d6480 <_ZN5guild7kernels3cpu15kq256_supportedEi>
  474b29:	test   %al,%al
  474b2b:	je     474963 <_ZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x83>
  474b31:	mov    0x18(%rsp),%rax
  474b36:	mov    0x28(%rsp),%rsi
  474b3b:	sub    $0x8,%rsp
  474b3f:	mov    %ebx,%r9d
  474b42:	mov    0x28(%rax),%rdx
  474b46:	mov    0x4(%rax),%edi
  474b49:	add    0x38(%rax),%rsi
  474b4d:	push   %r12
  474b4f:	push   %rbp
  474b50:	push   0x48(%rsp)
  474b54:	mov    0x40(%rsp),%r8
  474b59:	mov    0x18(%rax),%ecx
  474b5c:	call   4d6e80 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii>
  474b61:	add    $0x20,%rsp
  474b65:	jmp    474a67 <_ZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x187>
  474b6a:	lea    0x1df36cf(%rip),%r13        # 2268240 <_ZGVZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfiiE2kq>
  474b71:	mov    %r13,%rdi
  474b74:	call   13d80 <__cxa_guard_acquire@plt>
  474b79:	test   %eax,%eax
  474b7b:	je     474951 <_ZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x71>
  474b81:	lea    0x1c8d72(%rip),%rdi        # 63d8fa <_IO_stdin_used+0x58fa>
  474b88:	call   134d0 <getenv@plt>
  474b8d:	mov    %rax,%r14
  474b90:	call   46de70 <_ZN5guild7kernels3cpu11cpu_avx2_okEv>
  474b95:	test   %al,%al
  474b97:	je     474bb2 <_ZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x2d2>
  474b99:	test   %r14,%r14
  474b9c:	je     474bb2 <_ZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x2d2>
  474b9e:	xor    %esi,%esi
  474ba0:	mov    $0xa,%edx
  474ba5:	mov    %r14,%rdi
  474ba8:	call   13f80 <__isoc23_strtol@plt>
  474bad:	test   %eax,%eax
  474baf:	setne  %al
  474bb2:	mov    %r13,%rdi
  474bb5:	mov    %al,0x1df368d(%rip)        # 2268248 <_ZZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfiiE2kq>
  474bbb:	call   12e70 <__cxa_guard_release@plt>
  474bc0:	jmp    474951 <_ZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x71>
  474bc5:	lea    0x1df3684(%rip),%r13        # 2268250 <_ZGVZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfiiE6mt_min>
  474bcc:	mov    %r13,%rdi
  474bcf:	call   13d80 <__cxa_guard_acquire@plt>
  474bd4:	test   %eax,%eax
  474bd6:	je     474942 <_ZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x62>
  474bdc:	lea    0x1c8d07(%rip),%rdi        # 63d8ea <_IO_stdin_used+0x58ea>
  474be3:	call   134d0 <getenv@plt>
  474be8:	mov    %rax,%rdi
  474beb:	mov    $0x2,%eax
  474bf0:	test   %rdi,%rdi
  474bf3:	je     474c01 <_ZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x321>
  474bf5:	mov    $0xa,%edx
  474bfa:	xor    %esi,%esi
  474bfc:	call   13f80 <__isoc23_strtol@plt>
  474c01:	mov    %r13,%rdi
  474c04:	mov    %eax,0x1df364e(%rip)        # 2268258 <_ZZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfiiE6mt_min>
  474c0a:	call   12e70 <__cxa_guard_release@plt>
  474c0f:	jmp    474942 <_ZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x62>
  474c14:	lea    0x1df3645(%rip),%r13        # 2268260 <_ZGVZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfiiE8iq4nl_mt>
  474c1b:	mov    %r13,%rdi
  474c1e:	call   13d80 <__cxa_guard_acquire@plt>
  474c23:	test   %eax,%eax
  474c25:	je     474933 <_ZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x53>
  474c2b:	lea    0x1c8cd4(%rip),%rdi        # 63d906 <_IO_stdin_used+0x5906>
  474c32:	call   134d0 <getenv@plt>
  474c37:	mov    %r13,%rdi
  474c3a:	test   %rax,%rax
  474c3d:	sete   0x1df3624(%rip)        # 2268268 <_ZZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfiiE8iq4nl_mt>
  474c44:	call   12e70 <__cxa_guard_release@plt>
  474c49:	jmp    474933 <_ZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii+0x53>
  474c4e:	call   13390 <__stack_chk_fail@plt>
  474c53:	endbr64
  474c57:	mov    %rax,%rbx
  474c5a:	jmp    2e90d <_ZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii.cold>

Disassembly of section .fini:
