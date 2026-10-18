
/home/ubuntu/Guild/build-cuda12/ornith_prefill_bench:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

00000000004f0990 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii>:
  4f0990:	endbr64
  4f0994:	push   %r15
  4f0996:	mov    %rsi,%r10
  4f0999:	mov    %rdx,%rsi
  4f099c:	push   %r14
  4f099e:	push   %r13
  4f09a0:	push   %r12
  4f09a2:	push   %rbp
  4f09a3:	push   %rbx
  4f09a4:	sub    $0xa8,%rsp
  4f09ab:	mov    %ecx,0xc(%rsp)
  4f09af:	mov    0xe8(%rsp),%ebx
  4f09b6:	mov    %edi,0x8(%rsp)
  4f09ba:	mov    0xf0(%rsp),%ecx
  4f09c1:	mov    %r8,0x38(%rsp)
  4f09c6:	mov    0xe0(%rsp),%r13
  4f09ce:	mov    %fs:0x28,%rax
  4f09d7:	mov    %rax,0x98(%rsp)
  4f09df:	xor    %eax,%eax
  4f09e1:	lea    0x1(%rbx),%eax
  4f09e4:	cmp    %eax,%ecx
  4f09e6:	jle    4f0c55 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x2c5>
  4f09ec:	mov    %rdx,%rdi
  4f09ef:	lea    -0x2(%rcx),%edx
  4f09f2:	lea    -0x1(%r9),%r14d
  4f09f6:	movslq %ebx,%rax
  4f09f9:	sub    %ebx,%edx
  4f09fb:	imul   %rax,%rdi
  4f09ff:	lea    0x0(,%rax,4),%rbp
  4f0a07:	shr    $0x3,%r14d
  4f0a0b:	shr    $1,%edx
  4f0a0d:	add    $0x1,%r14d
  4f0a11:	lea    (%rdx,%rdx,1),%r8d
  4f0a15:	lea    0x2(%rax,%r8,1),%rax
  4f0a1a:	add    %r10,%rdi
  4f0a1d:	shl    $0x2,%rax
  4f0a21:	mov    %rax,0x30(%rsp)
  4f0a26:	lea    0x0(,%r14,8),%rax
  4f0a2e:	mov    %rax,0x18(%rsp)
  4f0a33:	lea    (%rsi,%rdi,1),%r14
  4f0a37:	test   %r9d,%r9d
  4f0a3a:	jle    4f0cb5 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x325>
  4f0a40:	lea    0x70(%rsp),%rax
  4f0a45:	mov    %edx,0x44(%rsp)
  4f0a49:	mov    %rbp,%r12
  4f0a4c:	mov    %r10,0x48(%rsp)
  4f0a51:	mov    %r9d,0x40(%rsp)
  4f0a56:	mov    %ebx,0xe8(%rsp)
  4f0a5d:	mov    %ecx,0xf0(%rsp)
  4f0a64:	mov    %rax,(%rsp)
  4f0a68:	lea    0x50(%rsp),%rax
  4f0a6d:	mov    %rax,0x20(%rsp)
  4f0a72:	mov    %rdi,%rax
  4f0a75:	nopl   (%rax)
  4f0a78:	mov    0x38(%rsp),%rbp
  4f0a7d:	mov    %rsi,0x28(%rsp)
  4f0a82:	xor    %r15d,%r15d
  4f0a85:	mov    %rax,0x10(%rsp)
  4f0a8a:	mov    0x40(%rsp),%ebx
  4f0a8e:	mov    %r13,%rax
  4f0a91:	mov    %r15,%r13
  4f0a94:	mov    %rbp,%r15
  4f0a97:	mov    %rax,%rbp
  4f0a9a:	nopw   0x0(%rax,%rax,1)
  4f0aa0:	mov    $0x8,%r9d
  4f0aa6:	push   (%rsp)
  4f0aa9:	mov    %r14,%rdx
  4f0aac:	mov    %r15,%r8
  4f0aaf:	push   0x28(%rsp)
  4f0ab3:	cmp    %r9d,%ebx
  4f0ab6:	mov    0x1c(%rsp),%ecx
  4f0aba:	cmovle %ebx,%r9d
  4f0abe:	mov    0x20(%rsp),%rsi
  4f0ac3:	mov    0x18(%rsp),%edi
  4f0ac7:	call   4efba0 <_ZN5guild7kernels3cpu12_GLOBAL__N_19dot_2rowsEiPKhS4_iPKPKviPfS9_>
  4f0acc:	vmovss 0x60(%rsp),%xmm0
  4f0ad2:	mov    0x0(%rbp,%r13,8),%rdx
  4f0ad7:	vinsertps $0x10,0x80(%rsp),%xmm0,%xmm0
  4f0ae2:	vmovlps %xmm0,(%rdx,%r12,1)
  4f0ae8:	pop    %rax
  4f0ae9:	pop    %rdx
  4f0aea:	cmp    $0x1,%ebx
  4f0aed:	jle    4f0bf6 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x266>
  4f0af3:	lea    0x1(%r13),%ecx
  4f0af7:	vmovss 0x54(%rsp),%xmm0
  4f0afd:	mov    %r13d,%edx
  4f0b00:	vinsertps $0x10,0x74(%rsp),%xmm0,%xmm0
  4f0b08:	mov    0x0(%rbp,%rcx,8),%rcx
  4f0b0d:	vmovlps %xmm0,(%rcx,%r12,1)
  4f0b13:	cmp    $0x2,%ebx
  4f0b16:	jle    4f0bf6 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x266>
  4f0b1c:	lea    0x2(%r13),%ecx
  4f0b20:	vmovss 0x58(%rsp),%xmm0
  4f0b26:	vinsertps $0x10,0x78(%rsp),%xmm0,%xmm0
  4f0b2e:	mov    0x0(%rbp,%rcx,8),%rcx
  4f0b33:	vmovlps %xmm0,(%rcx,%r12,1)
  4f0b39:	cmp    $0x3,%ebx
  4f0b3c:	jle    4f0bf6 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x266>
  4f0b42:	lea    0x3(%r13),%ecx
  4f0b46:	vmovss 0x5c(%rsp),%xmm0
  4f0b4c:	vinsertps $0x10,0x7c(%rsp),%xmm0,%xmm0
  4f0b54:	mov    0x0(%rbp,%rcx,8),%rcx
  4f0b59:	vmovlps %xmm0,(%rcx,%r12,1)
  4f0b5f:	cmp    $0x4,%ebx
  4f0b62:	jle    4f0bf6 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x266>
  4f0b68:	lea    0x4(%r13),%ecx
  4f0b6c:	vmovss 0x60(%rsp),%xmm0
  4f0b72:	vinsertps $0x10,0x80(%rsp),%xmm0,%xmm0
  4f0b7d:	mov    0x0(%rbp,%rcx,8),%rcx
  4f0b82:	vmovlps %xmm0,(%rcx,%r12,1)
  4f0b88:	cmp    $0x5,%ebx
  4f0b8b:	jle    4f0bf6 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x266>
  4f0b8d:	lea    0x5(%r13),%ecx
  4f0b91:	vmovss 0x64(%rsp),%xmm0
  4f0b97:	vinsertps $0x10,0x84(%rsp),%xmm0,%xmm0
  4f0ba2:	mov    0x0(%rbp,%rcx,8),%rcx
  4f0ba7:	vmovlps %xmm0,(%rcx,%r12,1)
  4f0bad:	cmp    $0x6,%ebx
  4f0bb0:	jle    4f0bf6 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x266>
  4f0bb2:	lea    0x6(%r13),%ecx
  4f0bb6:	vmovss 0x68(%rsp),%xmm0
  4f0bbc:	vinsertps $0x10,0x88(%rsp),%xmm0,%xmm0
  4f0bc7:	mov    0x0(%rbp,%rcx,8),%rcx
  4f0bcc:	vmovlps %xmm0,(%rcx,%r12,1)
  4f0bd2:	cmp    $0x7,%ebx
  4f0bd5:	jle    4f0bf6 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x266>
  4f0bd7:	add    $0x7,%edx
  4f0bda:	vmovss 0x6c(%rsp),%xmm0
  4f0be0:	vinsertps $0x10,0x8c(%rsp),%xmm0,%xmm0
  4f0beb:	mov    0x0(%rbp,%rdx,8),%rdx
  4f0bf0:	vmovlps %xmm0,(%rdx,%r12,1)
  4f0bf6:	mov    0x18(%rsp),%rax
  4f0bfb:	add    $0x8,%r13
  4f0bff:	sub    $0x8,%ebx
  4f0c02:	add    $0x40,%r15
  4f0c06:	cmp    %rax,%r13
  4f0c09:	jne    4f0aa0 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x110>
  4f0c0f:	mov    0x28(%rsp),%rsi
  4f0c14:	mov    0x30(%rsp),%rax
  4f0c19:	add    $0x8,%r12
  4f0c1d:	mov    %rbp,%r13
  4f0c20:	lea    (%rsi,%r14,1),%rdi
  4f0c24:	cmp    %rax,%r12
  4f0c27:	je     4f0c35 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x2a5>
  4f0c29:	mov    %rdi,%rax
  4f0c2c:	lea    (%rsi,%rdi,1),%r14
  4f0c30:	jmp    4f0a78 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0xe8>
  4f0c35:	mov    0x44(%rsp),%edx
  4f0c39:	mov    0x48(%rsp),%r10
  4f0c3e:	mov    0x40(%rsp),%r9d
  4f0c43:	mov    0xe8(%rsp),%ebx
  4f0c4a:	mov    0xf0(%rsp),%ecx
  4f0c51:	lea    0x2(%rbx,%rdx,2),%ebx
  4f0c55:	cmp    %ebx,%ecx
  4f0c57:	jle    4f0c8c <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x2fc>
  4f0c59:	movslq %ebx,%rdi
  4f0c5c:	mov    %rsi,%rdx
  4f0c5f:	sub    %ebx,%ecx
  4f0c61:	imul   %rdi,%rdx
  4f0c65:	add    %rdi,%rcx
  4f0c68:	lea    0x0(,%rdi,4),%rax
  4f0c70:	lea    0x0(,%rcx,4),%rdi
  4f0c78:	add    %r10,%rdx
  4f0c7b:	test   %r9d,%r9d
  4f0c7e:	jg     4f0ccd <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x33d>
  4f0c80:	add    $0x4,%rax
  4f0c84:	add    %rsi,%rdx
  4f0c87:	cmp    %rdi,%rax
  4f0c8a:	jne    4f0c7b <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x2eb>
  4f0c8c:	mov    0x98(%rsp),%rax
  4f0c94:	sub    %fs:0x28,%rax
  4f0c9d:	jne    4f0e6d <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x4dd>
  4f0ca3:	add    $0xa8,%rsp
  4f0caa:	pop    %rbx
  4f0cab:	pop    %rbp
  4f0cac:	pop    %r12
  4f0cae:	pop    %r13
  4f0cb0:	pop    %r14
  4f0cb2:	pop    %r15
  4f0cb4:	ret
  4f0cb5:	mov    0x30(%rsp),%rax
  4f0cba:	add    $0x8,%rbp
  4f0cbe:	lea    (%rsi,%r14,1),%rdi
  4f0cc2:	cmp    %rax,%rbp
  4f0cc5:	jne    4f0a33 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0xa3>
  4f0ccb:	jmp    4f0c51 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x2c1>
  4f0ccd:	lea    -0x1(%r9),%ebx
  4f0cd1:	lea    0x70(%rsp),%rcx
  4f0cd6:	mov    %rax,%r14
  4f0cd9:	mov    %r13,%r15
  4f0cdc:	shr    $0x3,%ebx
  4f0cdf:	mov    %rcx,(%rsp)
  4f0ce3:	add    $0x1,%ebx
  4f0ce6:	shl    $0x3,%rbx
  4f0cea:	mov    %rbx,%rax
  4f0ced:	mov    %rdx,%rbx
  4f0cf0:	mov    0x38(%rsp),%r13
  4f0cf5:	mov    %rdi,0x18(%rsp)
  4f0cfa:	xor    %r12d,%r12d
  4f0cfd:	mov    %r9d,%ebp
  4f0d00:	mov    %rsi,0x20(%rsp)
  4f0d05:	mov    %r9d,0x28(%rsp)
  4f0d0a:	mov    %r13,%rdx
  4f0d0d:	mov    %r14,%r13
  4f0d10:	mov    %rbx,0x10(%rsp)
  4f0d15:	mov    %rax,%rbx
  4f0d18:	mov    %r15,%rax
  4f0d1b:	mov    %rdx,%r14
  4f0d1e:	mov    %r12,%r15
  4f0d21:	mov    %rax,%r12
  4f0d24:	nopl   0x0(%rax)
  4f0d28:	mov    $0x8,%r8d
  4f0d2e:	mov    0xc(%rsp),%edx
  4f0d32:	mov    (%rsp),%r9
  4f0d36:	mov    %r14,%rcx
  4f0d39:	cmp    %r8d,%ebp
  4f0d3c:	mov    0x10(%rsp),%rsi
  4f0d41:	mov    0x8(%rsp),%edi
  4f0d45:	cmovle %ebp,%r8d
  4f0d49:	call   4e6750 <_ZN5guild7kernels3cpu12_GLOBAL__N_18dot_rowsEiPKhiPKPKviPf>
  4f0d4e:	mov    (%r12,%r15,8),%rdx
  4f0d52:	vmovss 0x70(%rsp),%xmm0
  4f0d58:	vmovss %xmm0,(%rdx,%r13,1)
  4f0d5e:	cmp    $0x1,%ebp
  4f0d61:	jle    4f0e27 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x497>
  4f0d67:	lea    0x1(%r15),%esi
  4f0d6b:	vmovss 0x74(%rsp),%xmm0
  4f0d71:	mov    %r15d,%edx
  4f0d74:	mov    (%r12,%rsi,8),%rsi
  4f0d78:	vmovss %xmm0,(%rsi,%r13,1)
  4f0d7e:	cmp    $0x2,%ebp
  4f0d81:	jle    4f0e27 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x497>
  4f0d87:	lea    0x2(%r15),%esi
  4f0d8b:	vmovss 0x78(%rsp),%xmm0
  4f0d91:	mov    (%r12,%rsi,8),%rsi
  4f0d95:	vmovss %xmm0,(%rsi,%r13,1)
  4f0d9b:	cmp    $0x3,%ebp
  4f0d9e:	jle    4f0e27 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x497>
  4f0da4:	lea    0x3(%r15),%esi
  4f0da8:	vmovss 0x7c(%rsp),%xmm0
  4f0dae:	mov    (%r12,%rsi,8),%rsi
  4f0db2:	vmovss %xmm0,(%rsi,%r13,1)
  4f0db8:	cmp    $0x4,%ebp
  4f0dbb:	jle    4f0e27 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x497>
  4f0dbd:	vmovss 0x80(%rsp),%xmm0
  4f0dc6:	lea    0x4(%r15),%esi
  4f0dca:	mov    (%r12,%rsi,8),%rsi
  4f0dce:	vmovss %xmm0,(%rsi,%r13,1)
  4f0dd4:	cmp    $0x5,%ebp
  4f0dd7:	jle    4f0e27 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x497>
  4f0dd9:	vmovss 0x84(%rsp),%xmm0
  4f0de2:	lea    0x5(%r15),%esi
  4f0de6:	mov    (%r12,%rsi,8),%rsi
  4f0dea:	vmovss %xmm0,(%rsi,%r13,1)
  4f0df0:	cmp    $0x6,%ebp
  4f0df3:	jle    4f0e27 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x497>
  4f0df5:	vmovss 0x88(%rsp),%xmm0
  4f0dfe:	lea    0x6(%r15),%esi
  4f0e02:	mov    (%r12,%rsi,8),%rsi
  4f0e06:	vmovss %xmm0,(%rsi,%r13,1)
  4f0e0c:	cmp    $0x7,%ebp
  4f0e0f:	jle    4f0e27 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x497>
  4f0e11:	vmovss 0x8c(%rsp),%xmm0
  4f0e1a:	add    $0x7,%edx
  4f0e1d:	mov    (%r12,%rdx,8),%rdx
  4f0e21:	vmovss %xmm0,(%rdx,%r13,1)
  4f0e27:	add    $0x8,%r15
  4f0e2b:	sub    $0x8,%ebp
  4f0e2e:	add    $0x40,%r14
  4f0e32:	cmp    %r15,%rbx
  4f0e35:	jne    4f0d28 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x398>
  4f0e3b:	mov    %r13,%r14
  4f0e3e:	mov    0x18(%rsp),%rdi
  4f0e43:	mov    %rbx,%rax
  4f0e46:	mov    0x20(%rsp),%rsi
  4f0e4b:	add    $0x4,%r14
  4f0e4f:	mov    0x10(%rsp),%rbx
  4f0e54:	mov    0x28(%rsp),%r9d
  4f0e59:	mov    %r12,%r15
  4f0e5c:	cmp    %r14,%rdi
  4f0e5f:	je     4f0c8c <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x2fc>
  4f0e65:	add    %rsi,%rbx
  4f0e68:	jmp    4f0cf0 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x360>
  4f0e6d:	call   143e0 <__stack_chk_fail@plt>

Disassembly of section .fini:
