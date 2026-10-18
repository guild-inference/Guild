
/home/ubuntu/Guild/bench/results/external-kernels/phase1/ornith_prefill_bench:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

00000000004d6c90 <_ZN5guild7kernels3cpu13kq256_gu_rowsEiPKhmmiPKPKviPKPfii>:
  4d6c90:	endbr64
  4d6c94:	push   %r15
  4d6c96:	push   %r14
  4d6c98:	push   %r13
  4d6c9a:	push   %r12
  4d6c9c:	push   %rbp
  4d6c9d:	push   %rbx
  4d6c9e:	mov    %rdx,%rbx
  4d6ca1:	sub    $0xb8,%rsp
  4d6ca8:	mov    0xf8(%rsp),%rax
  4d6cb0:	mov    %edi,0x30(%rsp)
  4d6cb4:	mov    %rsi,%rdi
  4d6cb7:	mov    %rcx,0x40(%rsp)
  4d6cbc:	mov    0x100(%rsp),%esi
  4d6cc3:	mov    %rax,0x50(%rsp)
  4d6cc8:	mov    0x108(%rsp),%eax
  4d6ccf:	mov    %r8d,0x34(%rsp)
  4d6cd4:	mov    %r9,0x48(%rsp)
  4d6cd9:	mov    %fs:0x28,%rdx
  4d6ce2:	mov    %rdx,0xa8(%rsp)
  4d6cea:	xor    %edx,%edx
  4d6cec:	cmp    %eax,%esi
  4d6cee:	jge    4d6e53 <_ZN5guild7kernels3cpu13kq256_gu_rowsEiPKhmmiPKPKviPKPfii+0x1c3>
  4d6cf4:	mov    0xf0(%rsp),%ecx
  4d6cfb:	test   %ecx,%ecx
  4d6cfd:	jle    4d6e53 <_ZN5guild7kernels3cpu13kq256_gu_rowsEiPKhmmiPKPKviPKPfii+0x1c3>
  4d6d03:	movslq %esi,%rdx
  4d6d06:	sub    %esi,%eax
  4d6d08:	mov    %rbx,%rcx
  4d6d0b:	mov    %rbx,0x58(%rsp)
  4d6d10:	add    %rdx,%rax
  4d6d13:	imul   %rdx,%rcx
  4d6d17:	lea    0x0(,%rdx,4),%r14
  4d6d1f:	shl    $0x2,%rax
  4d6d23:	lea    0x80(%rsp),%r12
  4d6d2b:	lea    0x60(%rsp),%r15
  4d6d30:	mov    %rax,0x38(%rsp)
  4d6d35:	mov    0xf0(%rsp),%eax
  4d6d3c:	add    %rdi,%rcx
  4d6d3f:	sub    $0x1,%eax
  4d6d42:	and    $0xfffffff8,%eax
  4d6d45:	add    $0x8,%eax
  4d6d48:	mov    %eax,0x2c(%rsp)
  4d6d4c:	nopl   0x0(%rax)
  4d6d50:	mov    0x40(%rsp),%rax
  4d6d55:	mov    0x50(%rsp),%rbp
  4d6d5a:	mov    %rcx,0x18(%rsp)
  4d6d5f:	movl   $0x0,0x28(%rsp)
  4d6d67:	add    %rcx,%rax
  4d6d6a:	mov    %rax,0x20(%rsp)
  4d6d6f:	mov    0x48(%rsp),%rax
  4d6d74:	mov    %rax,0x10(%rsp)
  4d6d79:	nopl   0x0(%rax)
  4d6d80:	mov    0x28(%rsp),%eax
  4d6d84:	mov    0xf0(%rsp),%ebx
  4d6d8b:	push   %r12
  4d6d8d:	push   %r15
  4d6d8f:	mov    0x20(%rsp),%r8
  4d6d94:	sub    %eax,%ebx
  4d6d96:	mov    $0x8,%eax
  4d6d9b:	mov    0x44(%rsp),%ecx
  4d6d9f:	mov    0x30(%rsp),%rdx
  4d6da4:	cmp    %eax,%ebx
  4d6da6:	mov    0x28(%rsp),%rsi
  4d6dab:	mov    0x40(%rsp),%edi
  4d6daf:	cmovg  %eax,%ebx
  4d6db2:	xor    %r13d,%r13d
  4d6db5:	mov    %ebx,%r9d
  4d6db8:	movslq %ebx,%rbx
  4d6dbb:	call   4d6090 <_ZN5guild7kernels3cpu12_GLOBAL__N_19dot_2rowsEiPKhS4_iPKPKviPfS9_>
  4d6dc0:	shl    $0x2,%rbx
  4d6dc4:	pop    %rax
  4d6dc5:	pop    %rdx
  4d6dc6:	cs nopw 0x0(%rax,%rax,1)
  4d6dd0:	vmovss (%r15,%r13,1),%xmm1
  4d6dd6:	vxorps 0x1a9992(%rip),%xmm1,%xmm0        # 680770 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0x2950>
  4d6dde:	vmovss %xmm1,0xc(%rsp)
  4d6de4:	call   13b60 <expf@plt>
  4d6de9:	vmovss 0xc(%rsp),%xmm1
  4d6def:	mov    0x0(%rbp,%r13,2),%rax
  4d6df4:	vaddss 0x161208(%rip),%xmm0,%xmm0        # 638004 <_IO_stdin_used+0x4>
  4d6dfc:	vdivss %xmm0,%xmm1,%xmm0
  4d6e00:	vmulss (%r12,%r13,1),%xmm0,%xmm0
  4d6e06:	add    $0x4,%r13
  4d6e0a:	vmovss %xmm0,(%rax,%r14,1)
  4d6e10:	cmp    %r13,%rbx
  4d6e13:	jne    4d6dd0 <_ZN5guild7kernels3cpu13kq256_gu_rowsEiPKhmmiPKPKviPKPfii+0x140>
  4d6e15:	addl   $0x8,0x28(%rsp)
  4d6e1a:	mov    0x2c(%rsp),%esi
  4d6e1e:	add    $0x40,%rbp
  4d6e22:	mov    0x28(%rsp),%eax
  4d6e26:	addq   $0x40,0x10(%rsp)
  4d6e2c:	cmp    %esi,%eax
  4d6e2e:	jne    4d6d80 <_ZN5guild7kernels3cpu13kq256_gu_rowsEiPKhmmiPKPKviPKPfii+0xf0>
  4d6e34:	mov    0x58(%rsp),%rax
  4d6e39:	mov    0x18(%rsp),%rcx
  4d6e3e:	add    $0x4,%r14
  4d6e42:	add    %rax,%rcx
  4d6e45:	mov    0x38(%rsp),%rax
  4d6e4a:	cmp    %rax,%r14
  4d6e4d:	jne    4d6d50 <_ZN5guild7kernels3cpu13kq256_gu_rowsEiPKhmmiPKPKviPKPfii+0xc0>
  4d6e53:	mov    0xa8(%rsp),%rax
  4d6e5b:	sub    %fs:0x28,%rax
  4d6e64:	jne    4d6e78 <_ZN5guild7kernels3cpu13kq256_gu_rowsEiPKhmmiPKPKviPKPfii+0x1e8>
  4d6e66:	add    $0xb8,%rsp
  4d6e6d:	pop    %rbx
  4d6e6e:	pop    %rbp
  4d6e6f:	pop    %r12
  4d6e71:	pop    %r13
  4d6e73:	pop    %r14
  4d6e75:	pop    %r15
  4d6e77:	ret
  4d6e78:	call   13390 <__stack_chk_fail@plt>

Disassembly of section .fini:
