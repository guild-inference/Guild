
build-cuda12/ornith_prefill_bench:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

00000000004758e0 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii>:
  4758e0:	endbr64
  4758e4:	push   %r15
  4758e6:	push   %r14
  4758e8:	push   %r13
  4758ea:	push   %r12
  4758ec:	mov    %r9d,%r12d
  4758ef:	push   %rbp
  4758f0:	movslq %ecx,%rbp
  4758f3:	push   %rbx
  4758f4:	sub    $0x78,%rsp
  4758f8:	mov    %rdi,0x38(%rsp)
  4758fd:	mov    0xb0(%rsp),%r13d
  475905:	mov    %rsi,0x48(%rsp)
  47590a:	mov    %rdx,0x40(%rsp)
  47590f:	mov    %r8,0x50(%rsp)
  475914:	mov    %fs:0x28,%rax
  47591d:	mov    %rax,0x68(%rsp)
  475922:	xor    %eax,%eax
  475924:	movzbl 0x1df39c5(%rip),%eax        # 22692f0 <_ZGVZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6avx512>
  47592b:	test   %al,%al
  47592d:	jne    475946 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x66>
  47592f:	lea    0x1df39ba(%rip),%rbx        # 22692f0 <_ZGVZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6avx512>
  475936:	mov    %rbx,%rdi
  475939:	call   13d80 <__cxa_guard_acquire@plt>
  47593e:	test   %eax,%eax
  475940:	jne    475c8a <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x3aa>
  475946:	movzbl 0x1df3993(%rip),%eax        # 22692e0 <_ZGVZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE4avx2>
  47594d:	test   %al,%al
  47594f:	jne    475968 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x88>
  475951:	lea    0x1df3988(%rip),%rbx        # 22692e0 <_ZGVZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE4avx2>
  475958:	mov    %rbx,%rdi
  47595b:	call   13d80 <__cxa_guard_acquire@plt>
  475960:	test   %eax,%eax
  475962:	jne    475c6a <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x38a>
  475968:	movzbl 0x1df3961(%rip),%eax        # 22692d0 <_ZGVZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6mt_min>
  47596f:	test   %al,%al
  475971:	je     475bec <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x30c>
  475977:	movzbl 0x1df3942(%rip),%eax        # 22692c0 <_ZGVZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE2kq>
  47597e:	test   %al,%al
  475980:	je     475b91 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x2b1>
  475986:	cmpb   $0x0,0x1df393b(%rip)        # 22692c8 <_ZZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE2kq>
  47598d:	je     4759a9 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0xc9>
  47598f:	mov    0x38(%rsp),%rax
  475994:	mov    (%rax),%edi
  475996:	mov    %edi,%eax
  475998:	and    $0xfffffffd,%eax
  47599b:	cmp    $0xc,%eax
  47599e:	jne    4759a9 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0xc9>
  4759a0:	cmp    $0x1,%ebp
  4759a3:	jg     475cbb <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x3db>
  4759a9:	movzbl 0x1df3900(%rip),%eax        # 22692b0 <_ZGVZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6cpu512>
  4759b0:	test   %al,%al
  4759b2:	je     475c3b <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x35b>
  4759b8:	mov    0x38(%rsp),%rax
  4759bd:	cmp    %ebp,0x1df3915(%rip)        # 22692d8 <_ZZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6mt_min>
  4759c3:	mov    (%rax),%edi
  4759c5:	jle    475b31 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x251>
  4759cb:	call   4d9ed0 <ggml_get_type_traits_cpu>
  4759d0:	mov    0x38(%rsp),%rdi
  4759d5:	mov    0x10(%rdi),%r14d
  4759d9:	cmp    %r13d,%r12d
  4759dc:	jge    475b0e <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x22e>
  4759e2:	test   %ebp,%ebp
  4759e4:	jle    475b0e <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x22e>
  4759ea:	movslq %r12d,%rsi
  4759ed:	sub    %r12d,%r13d
  4759f0:	mov    0x8(%rax),%rbx
  4759f4:	lea    0x0(%r13,%rsi,1),%rax
  4759f9:	mov    %rax,0x58(%rsp)
  4759fe:	mov    0x40(%rsp),%rax
  475a03:	lea    (%rax,%rbp,8),%rax
  475a07:	mov    %rax,0x18(%rsp)
  475a0c:	lea    0x60(%rsp),%rax
  475a11:	mov    %rax,0x20(%rsp)
  475a16:	lea    0x64(%rsp),%rax
  475a1b:	mov    %rax,0x28(%rsp)
  475a20:	mov    0x38(%rsp),%rax
  475a25:	mov    0x48(%rsp),%rcx
  475a2a:	mov    %rsi,0x30(%rsp)
  475a2f:	lea    0x0(,%rsi,4),%r12
  475a37:	mov    0x40(%rsp),%r15
  475a3c:	mov    0x20(%rax),%rbp
  475a40:	imul   %rsi,%rbp
  475a44:	lea    (%rcx,%rbp,1),%r13
  475a48:	add    0x30(%rax),%rbp
  475a4c:	lea    (%rcx,%rbp,1),%rax
  475a50:	mov    0x50(%rsp),%rbp
  475a55:	mov    %rax,0x10(%rsp)
  475a5a:	nopw   0x0(%rax,%rax,1)
  475a60:	movl   $0x0,0x60(%rsp)
  475a68:	xor    %r8d,%r8d
  475a6b:	mov    %r13,%rcx
  475a6e:	xor    %edx,%edx
  475a70:	movl   $0x0,0x64(%rsp)
  475a78:	mov    %r14d,%edi
  475a7b:	add    $0x8,%r15
  475a7f:	add    $0x8,%rbp
  475a83:	push   $0x1
  475a85:	push   $0x0
  475a87:	mov    0x30(%rsp),%rsi
  475a8c:	mov    -0x8(%r15),%r9
  475a90:	call   *%rbx
  475a92:	pop    %rax
  475a93:	xor    %r8d,%r8d
  475a96:	pop    %rdx
  475a97:	mov    %r14d,%edi
  475a9a:	push   $0x1
  475a9c:	xor    %edx,%edx
  475a9e:	push   $0x0
  475aa0:	mov    0x20(%rsp),%rcx
  475aa5:	mov    0x38(%rsp),%rsi
  475aaa:	mov    -0x8(%r15),%r9
  475aae:	call   *%rbx
  475ab0:	movss  0x70(%rsp),%xmm1
  475ab6:	movaps %xmm1,%xmm0
  475ab9:	xorps  0x20bde0(%rip),%xmm0        # 6818a0 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0x28a0>
  475ac0:	movss  %xmm1,0x1c(%rsp)
  475ac6:	call   13b60 <expf@plt>
  475acb:	movss  0x1c(%rsp),%xmm1
  475ad1:	mov    -0x8(%rbp),%rdx
  475ad5:	addss  0x1c3527(%rip),%xmm0        # 639004 <_IO_stdin_used+0x4>
  475add:	divss  %xmm0,%xmm1
  475ae1:	mulss  0x74(%rsp),%xmm1
  475ae7:	movss  %xmm1,(%rdx,%r12,1)
  475aed:	pop    %rcx
  475aee:	pop    %rsi
  475aef:	cmp    %r15,0x18(%rsp)
  475af4:	jne    475a60 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x180>
  475afa:	mov    0x30(%rsp),%rsi
  475aff:	add    $0x1,%rsi
  475b03:	cmp    %rsi,0x58(%rsp)
  475b08:	jne    475a20 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x140>
  475b0e:	mov    0x68(%rsp),%rax
  475b13:	sub    %fs:0x28,%rax
  475b1c:	jne    475dfe <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x51e>
  475b22:	add    $0x78,%rsp
  475b26:	pop    %rbx
  475b27:	pop    %rbp
  475b28:	pop    %r12
  475b2a:	pop    %r13
  475b2c:	pop    %r14
  475b2e:	pop    %r15
  475b30:	ret
  475b31:	call   492460 <_ZN5guild7kernels3cpu15iq512_supportedEi>
  475b36:	test   %al,%al
  475b38:	jne    475b57 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x277>
  475b3a:	mov    0x38(%rsp),%rax
  475b3f:	cmpb   $0x0,0x1df3772(%rip)        # 22692b8 <_ZZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6cpu512>
  475b46:	mov    (%rax),%edi
  475b48:	jne    4759cb <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0xeb>
  475b4e:	call   4bc650 <_ZN5guild7kernels3cpu15iq256_supportedEi>
  475b53:	test   %al,%al
  475b55:	je     475b85 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x2a5>
  475b57:	mov    0x38(%rsp),%rax
  475b5c:	cmpb   $0x0,0x1df3795(%rip)        # 22692f8 <_ZZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6avx512>
  475b63:	mov    (%rax),%edi
  475b65:	jne    475ca6 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x3c6>
  475b6b:	cmpb   $0x0,0x1df3776(%rip)        # 22692e8 <_ZZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE4avx2>
  475b72:	je     4759cb <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0xeb>
  475b78:	call   4bc650 <_ZN5guild7kernels3cpu15iq256_supportedEi>
  475b7d:	test   %al,%al
  475b7f:	jne    475d8b <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x4ab>
  475b85:	mov    0x38(%rsp),%rax
  475b8a:	mov    (%rax),%edi
  475b8c:	jmp    4759cb <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0xeb>
  475b91:	lea    0x1df3728(%rip),%rbx        # 22692c0 <_ZGVZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE2kq>
  475b98:	mov    %rbx,%rdi
  475b9b:	call   13d80 <__cxa_guard_acquire@plt>
  475ba0:	test   %eax,%eax
  475ba2:	je     475986 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0xa6>
  475ba8:	lea    0x1c8d6c(%rip),%rdi        # 63e91b <_IO_stdin_used+0x591b>
  475baf:	call   134d0 <getenv@plt>
  475bb4:	mov    %rax,%r14
  475bb7:	call   46f3d0 <_ZN5guild7kernels3cpu11cpu_avx2_okEv>
  475bbc:	test   %al,%al
  475bbe:	je     475bd9 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x2f9>
  475bc0:	test   %r14,%r14
  475bc3:	je     475bd9 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x2f9>
  475bc5:	xor    %esi,%esi
  475bc7:	mov    $0xa,%edx
  475bcc:	mov    %r14,%rdi
  475bcf:	call   13f80 <__isoc23_strtol@plt>
  475bd4:	test   %eax,%eax
  475bd6:	setne  %al
  475bd9:	mov    %rbx,%rdi
  475bdc:	mov    %al,0x1df36e6(%rip)        # 22692c8 <_ZZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE2kq>
  475be2:	call   12e70 <__cxa_guard_release@plt>
  475be7:	jmp    475986 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0xa6>
  475bec:	lea    0x1df36dd(%rip),%rbx        # 22692d0 <_ZGVZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6mt_min>
  475bf3:	mov    %rbx,%rdi
  475bf6:	call   13d80 <__cxa_guard_acquire@plt>
  475bfb:	test   %eax,%eax
  475bfd:	je     475977 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x97>
  475c03:	lea    0x1c8d01(%rip),%rdi        # 63e90b <_IO_stdin_used+0x590b>
  475c0a:	call   134d0 <getenv@plt>
  475c0f:	mov    %rax,%rdi
  475c12:	mov    $0x2,%eax
  475c17:	test   %rdi,%rdi
  475c1a:	je     475c28 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x348>
  475c1c:	mov    $0xa,%edx
  475c21:	xor    %esi,%esi
  475c23:	call   13f80 <__isoc23_strtol@plt>
  475c28:	mov    %rbx,%rdi
  475c2b:	mov    %eax,0x1df36a7(%rip)        # 22692d8 <_ZZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6mt_min>
  475c31:	call   12e70 <__cxa_guard_release@plt>
  475c36:	jmp    475977 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x97>
  475c3b:	lea    0x1df366e(%rip),%rbx        # 22692b0 <_ZGVZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6cpu512>
  475c42:	mov    %rbx,%rdi
  475c45:	call   13d80 <__cxa_guard_acquire@plt>
  475c4a:	test   %eax,%eax
  475c4c:	je     4759b8 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0xd8>
  475c52:	call   46f4a0 <_ZN5guild7kernels3cpu13cpu_avx512_okEv>
  475c57:	mov    %rbx,%rdi
  475c5a:	mov    %al,0x1df3658(%rip)        # 22692b8 <_ZZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6cpu512>
  475c60:	call   12e70 <__cxa_guard_release@plt>
  475c65:	jmp    4759b8 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0xd8>
  475c6a:	call   46f3d0 <_ZN5guild7kernels3cpu11cpu_avx2_okEv>
  475c6f:	test   %al,%al
  475c71:	jne    475d01 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x421>
  475c77:	mov    %rbx,%rdi
  475c7a:	mov    %al,0x1df3668(%rip)        # 22692e8 <_ZZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE4avx2>
  475c80:	call   12e70 <__cxa_guard_release@plt>
  475c85:	jmp    475968 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x88>
  475c8a:	call   46f4a0 <_ZN5guild7kernels3cpu13cpu_avx512_okEv>
  475c8f:	test   %al,%al
  475c91:	jne    475ced <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x40d>
  475c93:	mov    %rbx,%rdi
  475c96:	mov    %al,0x1df365c(%rip)        # 22692f8 <_ZZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6avx512>
  475c9c:	call   12e70 <__cxa_guard_release@plt>
  475ca1:	jmp    475946 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x66>
  475ca6:	call   492460 <_ZN5guild7kernels3cpu15iq512_supportedEi>
  475cab:	test   %al,%al
  475cad:	jne    475d18 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x438>
  475caf:	mov    0x38(%rsp),%rax
  475cb4:	mov    (%rax),%edi
  475cb6:	jmp    475b6b <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x28b>
  475cbb:	mov    0x38(%rsp),%rax
  475cc0:	mov    0x30(%rax),%rcx
  475cc4:	mov    0x20(%rax),%rdx
  475cc8:	push   %r13
  475cca:	push   %r12
  475ccc:	push   0x60(%rsp)
  475cd0:	push   %rbp
  475cd1:	mov    0x60(%rsp),%r9
  475cd6:	mov    0x68(%rsp),%rsi
  475cdb:	mov    0x10(%rax),%r8d
  475cdf:	call   4d81f0 <_ZN5guild7kernels3cpu13kq256_gu_rowsEiPKhmmiPKPKviPKPfii>
  475ce4:	add    $0x20,%rsp
  475ce8:	jmp    475b0e <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x22e>
  475ced:	lea    0x1c8bf9(%rip),%rdi        # 63e8ed <_IO_stdin_used+0x58ed>
  475cf4:	call   134d0 <getenv@plt>
  475cf9:	test   %rax,%rax
  475cfc:	sete   %al
  475cff:	jmp    475c93 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x3b3>
  475d01:	lea    0x1c8bf4(%rip),%rdi        # 63e8fc <_IO_stdin_used+0x58fc>
  475d08:	call   134d0 <getenv@plt>
  475d0d:	test   %rax,%rax
  475d10:	sete   %al
  475d13:	jmp    475c77 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x397>
  475d18:	test   %ebp,%ebp
  475d1a:	jle    475b0e <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x22e>
  475d20:	lea    -0x1(%rbp),%eax
  475d23:	mov    0x40(%rsp),%r15
  475d28:	mov    0x50(%rsp),%r14
  475d2d:	lea    -0x8(%rbp),%ebx
  475d30:	and    $0xfffffff8,%eax
  475d33:	sub    %eax,%ebx
  475d35:	mov    %ebx,0xc(%rsp)
  475d39:	mov    %r15,%rbx
  475d3c:	mov    %r14,%r15
  475d3f:	mov    0x38(%rsp),%r14
  475d44:	mov    $0x8,%eax
  475d49:	mov    0x30(%r14),%rcx
  475d4d:	mov    0x20(%r14),%rdx
  475d51:	mov    %rbx,%r9
  475d54:	cmp    %eax,%ebp
  475d56:	mov    (%r14),%edi
  475d59:	push   %r13
  475d5b:	cmovle %ebp,%eax
  475d5e:	push   %r12
  475d60:	sub    $0x8,%ebp
  475d63:	add    $0x40,%rbx
  475d67:	push   %r15
  475d69:	add    $0x40,%r15
  475d6d:	push   %rax
  475d6e:	mov    0x68(%rsp),%rsi
  475d73:	mov    0x10(%r14),%r8d
  475d77:	call   492480 <_ZN5guild7kernels3cpu13iq512_gu_rowsEiPKhmmiPKPKviPKPfii>
  475d7c:	add    $0x20,%rsp
  475d80:	cmp    %ebp,0xc(%rsp)
  475d84:	jne    475d44 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x464>
  475d86:	jmp    475b0e <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x22e>
  475d8b:	test   %ebp,%ebp
  475d8d:	jle    475b0e <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x22e>
  475d93:	lea    -0x1(%rbp),%eax
  475d96:	mov    0x40(%rsp),%r15
  475d9b:	mov    0x50(%rsp),%r14
  475da0:	lea    -0x8(%rbp),%ebx
  475da3:	and    $0xfffffff8,%eax
  475da6:	sub    %eax,%ebx
  475da8:	mov    %ebx,0xc(%rsp)
  475dac:	mov    %r15,%rbx
  475daf:	mov    %r14,%r15
  475db2:	mov    0x38(%rsp),%r14
  475db7:	mov    $0x8,%eax
  475dbc:	mov    0x30(%r14),%rcx
  475dc0:	mov    0x20(%r14),%rdx
  475dc4:	mov    %rbx,%r9
  475dc7:	cmp    %eax,%ebp
  475dc9:	mov    (%r14),%edi
  475dcc:	push   %r13
  475dce:	cmovle %ebp,%eax
  475dd1:	push   %r12
  475dd3:	sub    $0x8,%ebp
  475dd6:	add    $0x40,%rbx
  475dda:	push   %r15
  475ddc:	add    $0x40,%r15
  475de0:	push   %rax
  475de1:	mov    0x68(%rsp),%rsi
  475de6:	mov    0x10(%r14),%r8d
  475dea:	call   4bc670 <_ZN5guild7kernels3cpu13iq256_gu_rowsEiPKhmmiPKPKviPKPfii>
  475def:	add    $0x20,%rsp
  475df3:	cmp    %ebp,0xc(%rsp)
  475df7:	jne    475db7 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x4d7>
  475df9:	jmp    475b0e <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x22e>
  475dfe:	call   13390 <__stack_chk_fail@plt>
  475e03:	endbr64
  475e07:	mov    %rax,%rbp
  475e0a:	jmp    2e76d <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii.cold>
  475e0f:	endbr64
  475e13:	mov    %rax,%rbp
  475e16:	jmp    2e78d <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii.cold+0x20>
  475e1b:	endbr64
  475e1f:	mov    %rax,%rbp
  475e22:	jmp    2e7b2 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii.cold+0x45>
  475e27:	endbr64
  475e2b:	mov    %rax,%rbp
  475e2e:	jmp    2e7d2 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii.cold+0x65>

Disassembly of section .fini:
