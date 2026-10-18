
/home/ubuntu/Guild/bench/results/external-kernels/phase1/ornith_prefill_bench:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

00000000004667a0 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj>:
  4667a0:	endbr64
  4667a4:	push   %r15
  4667a6:	push   %r14
  4667a8:	lea    0x1(%rsi),%r14d
  4667ac:	push   %r13
  4667ae:	push   %r12
  4667b0:	push   %rbp
  4667b1:	push   %rbx
  4667b2:	mov    %rdi,%rbx
  4667b5:	sub    $0x168,%rsp
  4667bc:	mov    %rdx,0x50(%rsp)
  4667c1:	mov    %ecx,0x70(%rsp)
  4667c5:	mov    %fs:0x28,%rax
  4667ce:	mov    %rax,0x158(%rsp)
  4667d6:	xor    %eax,%eax
  4667d8:	movslq %esi,%rax
  4667db:	mov    %esi,0x1c(%rsp)
  4667df:	shl    $0x2,%rax
  4667e3:	mov    %r14d,0x74(%rsp)
  4667e8:	mov    %rax,0x40(%rsp)
  4667ed:	movslq %r14d,%rax
  4667f0:	shl    $0x6,%rax
  4667f4:	mov    %rax,0x48(%rsp)
  4667f9:	lea    0x26c0(%rdi),%rax
  466800:	mov    %rax,0x8(%rsp)
  466805:	nopl   (%rax)
  466808:	mov    0x8(%rsp),%rax
  46680d:	mov    (%rax),%rax
  466810:	mov    %rax,%rdx
  466813:	shr    $0x20,%rdx
  466817:	cmp    %edx,0x70(%rsp)
  46681b:	jne    4672ea <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xb4a>
  466821:	mov    %eax,%edx
  466823:	movzwl %ax,%ebp
  466826:	shr    $0x10,%edx
  466829:	cmp    %edx,%ebp
  46682b:	jae    4672ea <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xb4a>
  466831:	mov    0x8(%rsp),%rsi
  466836:	lea    0x1(%rax),%rdx
  46683a:	lock cmpxchg %rdx,(%rsi)
  46683f:	jne    466810 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x70>
  466841:	mov    0x1c(%rsp),%r8d
  466846:	test   %r8d,%r8d
  466849:	js     4672b8 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xb18>
  46684f:	mov    0x2698(%rbx),%rax
  466856:	mov    0x40(%rsp),%rdi
  46685b:	mov    %ebp,(%rax,%rdi,1)
  46685e:	movq   $0x0,0x10(%rsp)
  466867:	cmpq   $0x0,0x50(%rbx)
  46686c:	je     4668c2 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x122>
  46686e:	pxor   %xmm0,%xmm0
  466872:	lea    0xc0(%rsp),%rsi
  46687a:	mov    $0x3,%edi
  46687f:	movaps %xmm0,0xc0(%rsp)
  466887:	call   131a0 <clock_gettime@plt>
  46688c:	pxor   %xmm1,%xmm1
  466890:	pxor   %xmm0,%xmm0
  466894:	cvtsi2sdq 0xc0(%rsp),%xmm1
  46689e:	mulsd  0x21761a(%rip),%xmm1        # 67dec0 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0xa0>
  4668a6:	cvtsi2sdq 0xc8(%rsp),%xmm0
  4668b0:	divsd  0x21a668(%rip),%xmm0        # 680f20 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0x3100>
  4668b8:	addsd  %xmm0,%xmm1
  4668bc:	movsd  %xmm1,0x10(%rsp)
  4668c2:	mov    0x2898(%rbx),%r13d
  4668c9:	cmp    $0xa,%r13d
  4668cd:	ja     4668e8 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x148>
  4668cf:	lea    0x27ff4a(%rip),%rdx        # 6e6820 <_ZN5guild4spec12_GLOBAL__N_1L6kShapeE+0xca0>
  4668d6:	mov    %r13d,%eax
  4668d9:	movslq (%rdx,%rax,4),%rax
  4668dd:	add    %rdx,%rax
  4668e0:	notrack jmp *%rax
  4668e3:	nopl   0x0(%rax,%rax,1)
  4668e8:	cmp    $0x4,%r13d
  4668ec:	jle    467359 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xbb9>
  4668f2:	mov    0x28d8(%rbx),%rdi
  4668f9:	cmp    $0x5,%r13d
  4668fd:	je     467f92 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x17f2>
  466903:	mov    0x10(%rdi),%ecx
  466906:	mov    0x28c8(%rbx),%r8
  46690d:	mov    %ebp,%eax
  46690f:	movslq 0x28d0(%rbx),%rsi
  466916:	imul   %r8,%rax
  46691a:	cqto
  46691c:	idiv   %rsi
  46691f:	mov    %rax,%r10
  466922:	lea    0x1(%rbp),%eax
  466925:	imul   %r8,%rax
  466929:	cqto
  46692b:	idiv   %rsi
  46692e:	mov    %rax,%r9
  466931:	cmp    %rax,%r10
  466934:	jge    466cb8 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x518>
  46693a:	lea    0x110(%rsp),%rax
  466942:	mov    %r9,0x28(%rsp)
  466947:	movslq %ecx,%r11
  46694a:	mov    %r10,%r14
  46694d:	mov    %rax,0x20(%rsp)
  466952:	jmp    466a73 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x2d3>
  466957:	nopw   0x0(%rax,%rax,1)
  466960:	cmpl   $0x2a,0x4(%rdi)
  466964:	je     4675bd <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xe1d>
  46696a:	add    0x28c0(%rbx),%r15
  466971:	mov    0x8(%r15),%ecx
  466975:	test   %ecx,%ecx
  466977:	jle    466a23 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x283>
  46697d:	lea    0xc000(%r12),%rax
  466985:	mov    %rax,0x110(%rsp)
  46698d:	cmp    $0x1,%ecx
  466990:	je     466a23 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x283>
  466996:	lea    0xc400(%r12),%rax
  46699e:	mov    %rax,0x118(%rsp)
  4669a6:	cmp    $0x2,%ecx
  4669a9:	je     466a23 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x283>
  4669ab:	lea    0xc800(%r12),%rax
  4669b3:	mov    %rax,0x120(%rsp)
  4669bb:	cmp    $0x3,%ecx
  4669be:	je     466a23 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x283>
  4669c0:	lea    0xcc00(%r12),%rax
  4669c8:	mov    %rax,0x128(%rsp)
  4669d0:	cmp    $0x4,%ecx
  4669d3:	je     466a23 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x283>
  4669d5:	lea    0xd000(%r12),%rax
  4669dd:	mov    %rax,0x130(%rsp)
  4669e5:	cmp    $0x5,%ecx
  4669e8:	je     466a23 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x283>
  4669ea:	lea    0xd400(%r12),%rax
  4669f2:	mov    %rax,0x138(%rsp)
  4669fa:	cmp    $0x6,%ecx
  4669fd:	je     466a23 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x283>
  4669ff:	lea    0xd800(%r12),%rax
  466a07:	mov    %rax,0x140(%rsp)
  466a0f:	cmp    $0x7,%ecx
  466a12:	je     466a23 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x283>
  466a14:	add    $0xdc00,%r12
  466a1b:	mov    %r12,0x148(%rsp)
  466a23:	mov    %r11,0x30(%rsp)
  466a28:	sub    $0x8,%rsp
  466a2c:	mov    (%r15),%rsi
  466a2f:	mov    %r10d,%r9d
  466a32:	push   %rbp
  466a33:	mov    0x30(%rsp),%rdx
  466a38:	lea    0x50(%r15),%r8
  466a3c:	mov    %r10,0x10(%rsp)
  466a41:	call   4748e0 <_ZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii>
  466a46:	pop    %rax
  466a47:	pop    %rdx
  466a48:	mov    (%rsp),%r10
  466a4c:	mov    0x30(%rsp),%r11
  466a51:	sub    %r10d,%ebp
  466a54:	movslq %ebp,%rbp
  466a57:	add    %rbp,%r14
  466a5a:	cmp    %r14,0x28(%rsp)
  466a5f:	jle    466d90 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x5f0>
  466a65:	mov    0x2898(%rbx),%r13d
  466a6c:	mov    0x28d8(%rbx),%rdi
  466a73:	mov    %r14,%rax
  466a76:	mov    0x28(%rsp),%rbp
  466a7b:	cqto
  466a7d:	idiv   %r11
  466a80:	sub    %r14,%rbp
  466a83:	add    %rdx,%rbp
  466a86:	cltq
  466a88:	mov    %rdx,%r10
  466a8b:	cmp    %r11,%rbp
  466a8e:	lea    0x0(,%rax,8),%r15
  466a96:	cmovg  %r11,%rbp
  466a9a:	sub    %rax,%r15
  466a9d:	mov    %r15,%r12
  466aa0:	shl    $0x5,%r15
  466aa4:	shl    $0xd,%r12
  466aa8:	add    0x2918(%rbx),%r12
  466aaf:	cmp    $0x5,%r13d
  466ab3:	jne    466960 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1c0>
  466ab9:	cmpl   $0x2a,(%rdi)
  466abc:	je     467701 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xf61>
  466ac2:	add    0x28c0(%rbx),%r15
  466ac9:	mov    0x8(%r15),%ecx
  466acd:	test   %ecx,%ecx
  466acf:	jle    466b73 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x3d3>
  466ad5:	mov    %r12,0x110(%rsp)
  466add:	cmp    $0x1,%ecx
  466ae0:	je     466b73 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x3d3>
  466ae6:	lea    0xa00(%r12),%rax
  466aee:	mov    %rax,0x118(%rsp)
  466af6:	cmp    $0x2,%ecx
  466af9:	je     466b73 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x3d3>
  466afb:	lea    0x1400(%r12),%rax
  466b03:	mov    %rax,0x120(%rsp)
  466b0b:	cmp    $0x3,%ecx
  466b0e:	je     466b73 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x3d3>
  466b10:	lea    0x1e00(%r12),%rax
  466b18:	mov    %rax,0x128(%rsp)
  466b20:	cmp    $0x4,%ecx
  466b23:	je     466b73 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x3d3>
  466b25:	lea    0x2800(%r12),%rax
  466b2d:	mov    %rax,0x130(%rsp)
  466b35:	cmp    $0x5,%ecx
  466b38:	je     466b73 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x3d3>
  466b3a:	lea    0x3200(%r12),%rax
  466b42:	mov    %rax,0x138(%rsp)
  466b4a:	cmp    $0x6,%ecx
  466b4d:	je     466b73 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x3d3>
  466b4f:	lea    0x3c00(%r12),%rax
  466b57:	mov    %rax,0x140(%rsp)
  466b5f:	cmp    $0x7,%ecx
  466b62:	je     466b73 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x3d3>
  466b64:	add    $0x4600,%r12
  466b6b:	mov    %r12,0x148(%rsp)
  466b73:	mov    %r11,0x30(%rsp)
  466b78:	sub    $0x8,%rsp
  466b7c:	mov    (%r15),%rsi
  466b7f:	mov    %r10d,%r9d
  466b82:	push   %rbp
  466b83:	mov    0x30(%rsp),%r8
  466b88:	lea    0x90(%r15),%rdx
  466b8f:	mov    %r10,0x10(%rsp)
  466b94:	call   474380 <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii>
  466b99:	pop    %rdi
  466b9a:	pop    %r8
  466b9c:	mov    (%rsp),%r10
  466ba0:	mov    0x30(%rsp),%r11
  466ba5:	jmp    466a51 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x2b1>
  466baa:	movslq 0x2908(%rbx),%r8
  466bb1:	mov    %ebp,%edi
  466bb3:	movslq 0x290c(%rbx),%rax
  466bba:	mov    %rdi,%r9
  466bbd:	mov    0x28f0(%rbx),%rsi
  466bc4:	mov    0x28f8(%rbx),%r10
  466bcb:	imul   %r8,%r9
  466bcf:	mov    0x2900(%rbx),%rdx
  466bd6:	mov    %r8,%rcx
  466bd9:	lea    0x0(,%rax,4),%r13
  466be1:	mov    %r9,%r12
  466be4:	imul   %rax,%r12
  466be8:	test   %eax,%eax
  466bea:	jle    466cb8 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x518>
  466bf0:	add    %r9,%r8
  466bf3:	imul   %r13,%rdi
  466bf7:	add    %r12,%rax
  466bfa:	mov    %rsi,0x38(%rsp)
  466bff:	lea    (%r10,%r8,4),%rbp
  466c03:	mov    %r12,%r8
  466c06:	mov    %rax,0x20(%rsp)
  466c0b:	lea    (%r10,%r9,4),%rax
  466c0f:	neg    %r8
  466c12:	mov    %rax,0x30(%rsp)
  466c17:	xor    %r14d,%r14d
  466c1a:	mov    %r12,%r15
  466c1d:	lea    (%rdi,%r8,4),%rdi
  466c21:	mov    %ecx,0x28(%rsp)
  466c25:	lea    (%rdx,%rdi,1),%rax
  466c29:	mov    %rbx,0x58(%rsp)
  466c2e:	mov    %rax,(%rsp)
  466c32:	nopw   0x0(%rax,%rax,1)
  466c38:	mov    0x28(%rsp),%edi
  466c3c:	pxor   %xmm2,%xmm2
  466c40:	test   %edi,%edi
  466c42:	jle    466c92 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x4f2>
  466c44:	mov    0x38(%rsp),%rax
  466c49:	mov    0x30(%rsp),%r12
  466c4e:	pxor   %xmm2,%xmm2
  466c52:	lea    (%rax,%r15,4),%rbx
  466c56:	cs nopw 0x0(%rax,%rax,1)
  466c60:	movss  (%rbx),%xmm0
  466c64:	movss  (%r12),%xmm1
  466c6a:	movd   %xmm0,%eax
  466c6e:	not    %eax
  466c70:	test   $0x7f800000,%eax
  466c75:	sete   %al
  466c78:	add    $0x4,%r12
  466c7c:	add    %r13,%rbx
  466c7f:	movzbl %al,%eax
  466c82:	or     %eax,%r14d
  466c85:	call   13480 <fmaf@plt>
  466c8a:	movaps %xmm0,%xmm2
  466c8d:	cmp    %rbp,%r12
  466c90:	jne    466c60 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x4c0>
  466c92:	mov    (%rsp),%rax
  466c96:	movss  %xmm2,(%rax,%r15,4)
  466c9c:	mov    0x20(%rsp),%rax
  466ca1:	add    $0x1,%r15
  466ca5:	cmp    %rax,%r15
  466ca8:	jne    466c38 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x498>
  466caa:	mov    0x58(%rsp),%rbx
  466caf:	test   %r14d,%r14d
  466cb2:	jne    4675e4 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xe44>
  466cb8:	cmpq   $0x0,0x50(%rbx)
  466cbd:	je     466d50 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x5b0>
  466cc3:	pxor   %xmm0,%xmm0
  466cc7:	mov    $0x3,%edi
  466ccc:	mov    0x48(%rsp),%rbp
  466cd1:	lea    0xc0(%rsp),%rsi
  466cd9:	movaps %xmm0,0xc0(%rsp)
  466ce1:	add    0x38(%rbx),%rbp
  466ce5:	call   131a0 <clock_gettime@plt>
  466cea:	pxor   %xmm0,%xmm0
  466cee:	mov    0x2898(%rbx),%eax
  466cf4:	pxor   %xmm1,%xmm1
  466cf8:	cvtsi2sdq 0xc0(%rsp),%xmm0
  466d02:	mulsd  0x2171b6(%rip),%xmm0        # 67dec0 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0xa0>
  466d0a:	cvtsi2sdq 0xc8(%rsp),%xmm1
  466d14:	divsd  0x21a204(%rip),%xmm1        # 680f20 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0x3100>
  466d1c:	addsd  %xmm1,%xmm0
  466d20:	subsd  0x10(%rsp),%xmm0
  466d26:	cmp    $0x5,%eax
  466d29:	je     467318 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xb78>
  466d2f:	cmp    $0x6,%eax
  466d32:	je     46734a <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xbaa>
  466d38:	sub    $0x7,%eax
  466d3b:	and    $0xfffffffd,%eax
  466d3e:	jne    466d50 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x5b0>
  466d40:	addsd  0x8(%rbp),%xmm0
  466d45:	movsd  %xmm0,0x8(%rbp)
  466d4a:	nopw   0x0(%rax,%rax,1)
  466d50:	lock addl $0x1,0x2700(%rbx)
  466d58:	jmp    466808 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x68>
  466d5d:	mov    0x28d8(%rbx),%rdi
  466d64:	mov    %rbp,%rdx
  466d67:	mov    0x28e0(%rbx),%rax
  466d6e:	shl    $0xc,%rdx
  466d72:	add    0x28e8(%rbx),%rdx
  466d79:	imul   0x10(%rdi),%rbp
  466d7e:	lea    (%rax,%rbp,4),%rsi
  466d82:	call   474320 <_ZN5guild7kernels3cpu16native_quant_actERKNS1_9NativeFmtEPKfPv>
  466d87:	nopw   0x0(%rax,%rax,1)
  466d90:	cmpq   $0x0,0x50(%rbx)
  466d95:	je     466d50 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x5b0>
  466d97:	cmpl   $0x8,0x2898(%rbx)
  466d9e:	je     466d50 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x5b0>
  466da0:	jmp    466cc3 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x523>
  466da5:	lea    0x0(,%rbp,8),%rax
  466dad:	movq   $0x0,0x78(%rsp)
  466db6:	sub    %rbp,%rax
  466db9:	shl    $0x5,%rax
  466dbd:	add    0x28c0(%rbx),%rax
  466dc4:	mov    0xd0(%rax),%r12
  466dcb:	mov    %rax,%rcx
  466dce:	mov    0x8(%rax),%ebp
  466dd1:	mov    %rax,0x28(%rsp)
  466dd6:	mov    0xd8(%rcx),%rsi
  466ddd:	lea    0x90(%rax),%rax
  466de4:	test   %r12,%r12
  466de7:	cmove  %rax,%r12
  466deb:	mov    %rcx,%rax
  466dee:	add    $0x50,%rax
  466df2:	test   %rsi,%rsi
  466df5:	cmovne %rsi,%rax
  466df9:	mov    0x1c(%rsp),%esi
  466dfd:	mov    %rax,0x38(%rsp)
  466e02:	test   %esi,%esi
  466e04:	js     467a20 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1280>
  466e0a:	mov    0x28d8(%rbx),%rdi
  466e11:	cmpl   $0x2a,(%rdi)
  466e14:	je     467a3a <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x129a>
  466e1a:	test   %ebp,%ebp
  466e1c:	jle    466eca <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x72a>
  466e22:	mov    %fs:0x0,%r11
  466e2b:	add    $0xffffffffffffb000,%r11
  466e32:	mov    %r11,0x110(%rsp)
  466e3a:	cmp    $0x1,%ebp
  466e3d:	je     466eca <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x72a>
  466e43:	lea    0xa00(%r11),%rax
  466e4a:	mov    %rax,0x118(%rsp)
  466e52:	cmp    $0x2,%ebp
  466e55:	je     466eca <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x72a>
  466e57:	lea    0x1400(%r11),%rax
  466e5e:	mov    %rax,0x120(%rsp)
  466e66:	cmp    $0x3,%ebp
  466e69:	je     466eca <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x72a>
  466e6b:	lea    0x1e00(%r11),%rax
  466e72:	mov    %rax,0x128(%rsp)
  466e7a:	cmp    $0x4,%ebp
  466e7d:	je     466eca <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x72a>
  466e7f:	lea    0x2800(%r11),%rax
  466e86:	mov    %rax,0x130(%rsp)
  466e8e:	cmp    $0x5,%ebp
  466e91:	je     466eca <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x72a>
  466e93:	lea    0x3200(%r11),%rax
  466e9a:	mov    %rax,0x138(%rsp)
  466ea2:	cmp    $0x6,%ebp
  466ea5:	je     466eca <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x72a>
  466ea7:	lea    0x3c00(%r11),%rax
  466eae:	mov    %rax,0x140(%rsp)
  466eb6:	cmp    $0x7,%ebp
  466eb9:	je     466eca <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x72a>
  466ebb:	add    $0x4600,%r11
  466ec2:	mov    %r11,0x148(%rsp)
  466eca:	mov    0x28(%rsp),%rax
  466ecf:	sub    $0x8,%rsp
  466ed3:	xor    %r9d,%r9d
  466ed6:	mov    %ebp,%ecx
  466ed8:	mov    %r12,%rdx
  466edb:	mov    (%rax),%rsi
  466ede:	mov    0x18(%rdi),%rax
  466ee2:	push   %rax
  466ee3:	lea    0x120(%rsp),%r8
  466eeb:	call   474380 <_ZN5guild7kernels3cpu14native_gu_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii>
  466ef0:	pop    %r10
  466ef2:	pop    %r11
  466ef4:	movq   $0x0,0x58(%rsp)
  466efd:	mov    0x1c(%rsp),%r9d
  466f02:	test   %r9d,%r9d
  466f05:	js     467e30 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1690>
  466f0b:	movq   $0x0,0x30(%rsp)
  466f14:	cmpq   $0x0,0x50(%rbx)
  466f19:	je     466f6f <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x7cf>
  466f1b:	pxor   %xmm0,%xmm0
  466f1f:	lea    0xc0(%rsp),%rsi
  466f27:	mov    $0x3,%edi
  466f2c:	movaps %xmm0,0xc0(%rsp)
  466f34:	call   131a0 <clock_gettime@plt>
  466f39:	pxor   %xmm1,%xmm1
  466f3d:	pxor   %xmm0,%xmm0
  466f41:	cvtsi2sdq 0xc0(%rsp),%xmm1
  466f4b:	mulsd  0x216f6d(%rip),%xmm1        # 67dec0 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0xa0>
  466f53:	cvtsi2sdq 0xc8(%rsp),%xmm0
  466f5d:	divsd  0x219fbb(%rip),%xmm0        # 680f20 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0x3100>
  466f65:	addsd  %xmm0,%xmm1
  466f69:	movsd  %xmm1,0x30(%rsp)
  466f6f:	mov    0x28d8(%rbx),%rax
  466f76:	cmpl   $0x2a,0x4(%rax)
  466f7a:	je     467e3f <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x169f>
  466f80:	test   %ebp,%ebp
  466f82:	jle    467fd3 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1833>
  466f88:	lea    0x110(%rsp),%rax
  466f90:	movslq %ebp,%rdx
  466f93:	mov    %fs:0x0,%r11
  466f9c:	mov    %rax,0x20(%rsp)
  466fa1:	mov    %rax,%r12
  466fa4:	lea    (%rax,%rdx,8),%r15
  466fa8:	lea    -0x5000(%r11),%r14
  466faf:	lea    -0xe000(%r11),%r13
  466fb6:	cs nopw 0x0(%rax,%rax,1)
  466fc0:	mov    0x28d8(%rbx),%rdi
  466fc7:	mov    %r13,%rdx
  466fca:	mov    %r14,%rsi
  466fcd:	add    $0x8,%r12
  466fd1:	add    $0xa00,%r14
  466fd8:	call   474350 <_ZN5guild7kernels3cpu14native_quant_hERKNS1_9NativeFmtEPKfPv>
  466fdd:	mov    %r13,-0x8(%r12)
  466fe2:	add    $0x400,%r13
  466fe9:	cmp    %r12,%r15
  466fec:	jne    466fc0 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x820>
  466fee:	mov    0x1c(%rsp),%ecx
  466ff2:	xor    %r12d,%r12d
  466ff5:	test   %ecx,%ecx
  466ff7:	js     467f9a <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x17fa>
  466ffd:	cmpq   $0x0,0x50(%rbx)
  467002:	movq   $0x0,(%rsp)
  46700a:	je     46705f <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x8bf>
  46700c:	pxor   %xmm0,%xmm0
  467010:	lea    0xc0(%rsp),%rsi
  467018:	mov    $0x3,%edi
  46701d:	movaps %xmm0,0xc0(%rsp)
  467025:	call   131a0 <clock_gettime@plt>
  46702a:	pxor   %xmm1,%xmm1
  46702e:	pxor   %xmm0,%xmm0
  467032:	cvtsi2sdq 0xc0(%rsp),%xmm1
  46703c:	mulsd  0x216e7c(%rip),%xmm1        # 67dec0 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0xa0>
  467044:	cvtsi2sdq 0xc8(%rsp),%xmm0
  46704e:	divsd  0x219eca(%rip),%xmm0        # 680f20 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0x3100>
  467056:	addsd  %xmm0,%xmm1
  46705a:	movsd  %xmm1,(%rsp)
  46705f:	mov    0x28(%rsp),%rax
  467064:	sub    $0x8,%rsp
  467068:	xor    %r9d,%r9d
  46706b:	mov    %ebp,%ecx
  46706d:	mov    0x28d8(%rbx),%rdi
  467074:	mov    (%rax),%rsi
  467077:	mov    0x10(%rdi),%rax
  46707b:	push   %rax
  46707c:	mov    0x48(%rsp),%r8
  467081:	mov    0x30(%rsp),%rdx
  467086:	call   4748e0 <_ZN5guild7kernels3cpu16native_down_rowsERKNS1_9NativeFmtEPKhPKPKviPKPfii>
  46708b:	pop    %rax
  46708c:	pop    %rdx
  46708d:	mov    0x1c(%rsp),%r15d
  467092:	test   %r15d,%r15d
  467095:	js     467dbf <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x161f>
  46709b:	cmpq   $0x0,0x50(%rbx)
  4670a0:	je     466d50 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x5b0>
  4670a6:	movsd  0x30(%rsp),%xmm4
  4670ac:	movsd  (%rsp),%xmm1
  4670b1:	lea    0xc0(%rsp),%rsi
  4670b9:	mov    $0x3,%edi
  4670be:	movslq 0x74(%rsp),%rbp
  4670c3:	subsd  %xmm4,%xmm1
  4670c7:	movapd %xmm4,%xmm0
  4670cb:	subsd  0x10(%rsp),%xmm0
  4670d1:	shl    $0x6,%rbp
  4670d5:	add    0x38(%rbx),%rbp
  4670d9:	unpcklpd %xmm1,%xmm0
  4670dd:	addpd  0x0(%rbp),%xmm0
  4670e2:	movaps %xmm0,0x0(%rbp)
  4670e6:	pxor   %xmm0,%xmm0
  4670ea:	movaps %xmm0,0xc0(%rsp)
  4670f2:	call   131a0 <clock_gettime@plt>
  4670f7:	pxor   %xmm0,%xmm0
  4670fb:	pxor   %xmm1,%xmm1
  4670ff:	cvtsi2sdq 0xc0(%rsp),%xmm0
  467109:	mulsd  0x216daf(%rip),%xmm0        # 67dec0 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0xa0>
  467111:	cvtsi2sdq 0xc8(%rsp),%xmm1
  46711b:	divsd  0x219dfd(%rip),%xmm1        # 680f20 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0x3100>
  467123:	addsd  %xmm1,%xmm0
  467127:	subsd  (%rsp),%xmm0
  46712c:	addsd  0x10(%rbp),%xmm0
  467131:	movsd  %xmm0,0x10(%rbp)
  467136:	jmp    466d90 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x5f0>
  46713b:	lea    0x0(,%rbp,8),%rdx
  467143:	mov    0x28c0(%rbx),%rax
  46714a:	sub    %rbp,%rdx
  46714d:	mov    %rdx,%r12
  467150:	shl    $0x5,%rdx
  467154:	mov    0x8(%rax,%rdx,1),%r14d
  467159:	shl    $0xd,%r12
  46715d:	add    0x2918(%rbx),%r12
  467164:	test   %r14d,%r14d
  467167:	jle    466cb8 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x518>
  46716d:	lea    0x5000(%r12),%rbp
  467175:	lea    0xc000(%r12),%r14
  46717d:	xor    %r13d,%r13d
  467180:	mov    %rdx,%r15
  467183:	jmp    4671be <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xa1e>
  467185:	nopl   (%rax)
  467188:	mov    %r14,%rdx
  46718b:	mov    %r12,%rsi
  46718e:	call   474350 <_ZN5guild7kernels3cpu14native_quant_hERKNS1_9NativeFmtEPKfPv>
  467193:	mov    0x28c0(%rbx),%rax
  46719a:	add    $0x1,%r13d
  46719e:	add    $0xa00,%r12
  4671a5:	add    $0xe00,%rbp
  4671ac:	add    $0x400,%r14
  4671b3:	cmp    %r13d,0x8(%rax,%r15,1)
  4671b8:	jle    466d90 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x5f0>
  4671be:	mov    0x28d8(%rbx),%rdi
  4671c5:	cmpl   $0x2a,0x4(%rdi)
  4671c9:	jne    467188 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x9e8>
  4671cb:	call   46de70 <_ZN5guild7kernels3cpu11cpu_avx2_okEv>
  4671d0:	test   %al,%al
  4671d2:	jne    467330 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xb90>
  4671d8:	mov    0x28d8(%rbx),%rdi
  4671df:	jmp    467188 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x9e8>
  4671e1:	mov    0x289c(%rbx),%esi
  4671e7:	mov    %ebp,%eax
  4671e9:	cltd
  4671ea:	idiv   %esi
  4671ec:	lea    (%rdx,%rdx,4),%ecx
  4671ef:	movslq %eax,%r9
  4671f2:	shl    $0x7,%ecx
  4671f5:	mov    %r9,%rdi
  4671f8:	lea    0x280(%rcx),%eax
  4671fe:	shl    $0x5,%rdi
  467202:	add    0x60(%rbx),%rdi
  467206:	cltd
  467207:	idiv   %esi
  467209:	mov    %eax,%r8d
  46720c:	mov    %ecx,%eax
  46720e:	cltd
  46720f:	idiv   %esi
  467211:	mov    0x8(%rdi),%rsi
  467215:	mov    (%rdi),%rdi
  467218:	lea    (%r9,%r9,2),%rdx
  46721c:	shl    $0xb,%rdx
  467220:	add    0x28a8(%rbx),%rdx
  467227:	mov    %eax,%ecx
  467229:	call   459890 <_ZN5guild7kernels3cpu17s2_expert_gu_rowsEPKhRKNS1_4ActQEPfii>
  46722e:	jmp    466d90 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x5f0>
  467233:	mov    0x28a0(%rbx),%r9d
  46723a:	mov    %ebp,%eax
  46723c:	cltd
  46723d:	idiv   %r9d
  467240:	lea    (%rdx,%rdx,4),%ecx
  467243:	movslq %eax,%rsi
  467246:	shl    $0x9,%ecx
  467249:	mov    %rsi,%rdi
  46724c:	lea    (%rsi,%rsi,2),%rsi
  467250:	lea    0xa00(%rcx),%eax
  467256:	shl    $0x5,%rdi
  46725a:	add    0x60(%rbx),%rdi
  46725e:	shl    $0xb,%rsi
  467262:	cltd
  467263:	add    0x28a8(%rbx),%rsi
  46726a:	idiv   %r9d
  46726d:	add    $0xa00,%rsi
  467274:	mov    %eax,%r8d
  467277:	mov    %ecx,%eax
  467279:	cltd
  46727a:	idiv   %r9d
  46727d:	mov    0x10(%rdi),%rdx
  467281:	mov    (%rdi),%rdi
  467284:	mov    %eax,%ecx
  467286:	call   45a2d0 <_ZN5guild7kernels3cpu19s2_expert_down_rowsEPKhRKNS1_4ActQEPfii>
  46728b:	jmp    466d90 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x5f0>
  467290:	shl    $0x5,%rbp
  467294:	add    0x60(%rbx),%rbp
  467298:	mov    0x50(%rsp),%rcx
  46729d:	mov    0x10(%rbp),%rdx
  4672a1:	mov    0x8(%rbp),%rsi
  4672a5:	mov    0x0(%rbp),%rdi
  4672a9:	call   458130 <_ZN5guild7kernels3cpu16s2_expert_vnni_qEPKhRKNS1_4ActQEPfRNS1_13ExpertScratchE>
  4672ae:	jmp    466d90 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x5f0>
  4672b3:	nopl   0x0(%rax,%rax,1)
  4672b8:	mov    %ebp,0x26a0(%rbx)
  4672be:	call   13650 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  4672c3:	mov    %rax,%rcx
  4672c6:	movabs $0x431bde82d7b634db,%rax
  4672d0:	imul   %rcx
  4672d3:	sar    $0x3f,%rcx
  4672d7:	sar    $0x12,%rdx
  4672db:	sub    %rcx,%rdx
  4672de:	mov    %rdx,0x26a8(%rbx)
  4672e5:	jmp    46685e <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xbe>
  4672ea:	mov    0x158(%rsp),%rax
  4672f2:	sub    %fs:0x28,%rax
  4672fb:	jne    467fe5 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1845>
  467301:	add    $0x168,%rsp
  467308:	pop    %rbx
  467309:	pop    %rbp
  46730a:	pop    %r12
  46730c:	pop    %r13
  46730e:	pop    %r14
  467310:	pop    %r15
  467312:	ret
  467313:	nopl   0x0(%rax,%rax,1)
  467318:	addsd  0x0(%rbp),%xmm0
  46731d:	movsd  %xmm0,0x0(%rbp)
  467322:	jmp    466d50 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x5b0>
  467327:	nopw   0x0(%rax,%rax,1)
  467330:	mov    0x28d8(%rbx),%rax
  467337:	mov    %rbp,%rdx
  46733a:	mov    %r12,%rdi
  46733d:	mov    0x18(%rax),%esi
  467340:	call   46e030 <_ZN5guild7kernels3cpu13act_quant_anyEPKfiRNS1_4ActQE>
  467345:	jmp    467193 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x9f3>
  46734a:	addsd  0x10(%rbp),%xmm0
  46734f:	movsd  %xmm0,0x10(%rbp)
  467354:	jmp    466d50 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x5b0>
  467359:	cmp    $0x3,%r13d
  46735d:	mov    0x28c8(%rbx),%rsi
  467364:	mov    $0x280,%eax
  467369:	mov    $0xa00,%r14d
  46736f:	cmove  %rax,%r14
  467373:	mov    %ebp,%eax
  467375:	movslq 0x28d0(%rbx),%rcx
  46737c:	imul   %rsi,%rax
  467380:	cqto
  467382:	idiv   %rcx
  467385:	mov    %rax,%r12
  467388:	lea    0x1(%rbp),%eax
  46738b:	imul   %rsi,%rax
  46738f:	cqto
  467391:	idiv   %rcx
  467394:	mov    %rax,%rbp
  467397:	cmp    %rax,%r12
  46739a:	jge    466cb8 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x518>
  4673a0:	lea    0x110(%rsp),%rax
  4673a8:	mov    %r13d,%ecx
  4673ab:	mov    %rbx,%r13
  4673ae:	mov    %rax,0x20(%rsp)
  4673b3:	jmp    467499 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xcf9>
  4673b8:	nopl   0x0(%rax,%rax,1)
  4673c0:	test   %esi,%esi
  4673c2:	jle    467467 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xcc7>
  4673c8:	lea    0x5000(%rdx),%rcx
  4673cf:	mov    %rcx,0x110(%rsp)
  4673d7:	cmp    $0x1,%esi
  4673da:	je     467467 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xcc7>
  4673e0:	lea    0x5e00(%rdx),%rcx
  4673e7:	mov    %rcx,0x118(%rsp)
  4673ef:	cmp    $0x2,%esi
  4673f2:	je     467467 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xcc7>
  4673f4:	lea    0x6c00(%rdx),%rcx
  4673fb:	mov    %rcx,0x120(%rsp)
  467403:	cmp    $0x3,%esi
  467406:	je     467467 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xcc7>
  467408:	lea    0x7a00(%rdx),%rcx
  46740f:	mov    %rcx,0x128(%rsp)
  467417:	cmp    $0x4,%esi
  46741a:	je     467467 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xcc7>
  46741c:	lea    0x8800(%rdx),%rcx
  467423:	mov    %rcx,0x130(%rsp)
  46742b:	cmp    $0x5,%esi
  46742e:	je     467467 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xcc7>
  467430:	lea    0x9600(%rdx),%rcx
  467437:	mov    %rcx,0x138(%rsp)
  46743f:	cmp    $0x6,%esi
  467442:	je     467467 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xcc7>
  467444:	lea    0xa400(%rdx),%rcx
  46744b:	mov    %rcx,0x140(%rsp)
  467453:	cmp    $0x7,%esi
  467456:	je     467467 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xcc7>
  467458:	add    $0xb200,%rdx
  46745f:	mov    %rdx,0x148(%rsp)
  467467:	mov    %r15d,%r9d
  46746a:	sub    %ebx,%r15d
  46746d:	mov    %esi,%edx
  46746f:	mov    (%rax),%rdi
  467472:	mov    0x20(%rsp),%rsi
  467477:	movslq %r15d,%r15
  46747a:	lea    0x50(%rax),%rcx
  46747e:	mov    %ebx,%r8d
  467481:	add    %r15,%r12
  467484:	call   45bd20 <_ZN5guild7kernels3cpu25s2_expert_down_rows_multiEPKhPKPKNS1_4ActQEiPKPfii>
  467489:	cmp    %r12,%rbp
  46748c:	jle    4675b5 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xe15>
  467492:	mov    0x2898(%r13),%ecx
  467499:	mov    %r12,%rax
  46749c:	mov    %rbp,%r15
  46749f:	cqto
  4674a1:	sub    %r12,%r15
  4674a4:	idiv   %r14
  4674a7:	add    %rdx,%r15
  4674aa:	mov    %rdx,%rbx
  4674ad:	movslq %eax,%rdx
  4674b0:	cmp    %r14,%r15
  4674b3:	lea    0x0(,%rdx,8),%rax
  4674bb:	cmovg  %r14,%r15
  4674bf:	sub    %rdx,%rax
  4674c2:	mov    %rax,%rdx
  4674c5:	shl    $0x5,%rax
  4674c9:	add    0x28c0(%r13),%rax
  4674d0:	shl    $0xd,%rdx
  4674d4:	mov    0x8(%rax),%esi
  4674d7:	add    0x2918(%r13),%rdx
  4674de:	cmp    $0x3,%ecx
  4674e1:	jne    4673c0 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xc20>
  4674e7:	test   %esi,%esi
  4674e9:	jle    467587 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xde7>
  4674ef:	mov    %rdx,0x110(%rsp)
  4674f7:	cmp    $0x1,%esi
  4674fa:	je     467587 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xde7>
  467500:	lea    0xa00(%rdx),%rcx
  467507:	mov    %rcx,0x118(%rsp)
  46750f:	cmp    $0x2,%esi
  467512:	je     467587 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xde7>
  467514:	lea    0x1400(%rdx),%rcx
  46751b:	mov    %rcx,0x120(%rsp)
  467523:	cmp    $0x3,%esi
  467526:	je     467587 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xde7>
  467528:	lea    0x1e00(%rdx),%rcx
  46752f:	mov    %rcx,0x128(%rsp)
  467537:	cmp    $0x4,%esi
  46753a:	je     467587 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xde7>
  46753c:	lea    0x2800(%rdx),%rcx
  467543:	mov    %rcx,0x130(%rsp)
  46754b:	cmp    $0x5,%esi
  46754e:	je     467587 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xde7>
  467550:	lea    0x3200(%rdx),%rcx
  467557:	mov    %rcx,0x138(%rsp)
  46755f:	cmp    $0x6,%esi
  467562:	je     467587 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xde7>
  467564:	lea    0x3c00(%rdx),%rcx
  46756b:	mov    %rcx,0x140(%rsp)
  467573:	cmp    $0x7,%esi
  467576:	je     467587 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xde7>
  467578:	add    $0x4600,%rdx
  46757f:	mov    %rdx,0x148(%rsp)
  467587:	mov    %r15d,%r9d
  46758a:	sub    %ebx,%r15d
  46758d:	lea    0x10(%rax),%r11
  467591:	mov    (%rax),%rdi
  467594:	mov    0x20(%rsp),%rcx
  467599:	movslq %r15d,%r15
  46759c:	mov    %esi,%edx
  46759e:	mov    %ebx,%r8d
  4675a1:	mov    %r11,%rsi
  4675a4:	add    %r15,%r12
  4675a7:	call   45a990 <_ZN5guild7kernels3cpu23s2_expert_gu_rows_multiEPKhPKPKNS1_4ActQEiPKPfii>
  4675ac:	cmp    %r12,%rbp
  4675af:	jg     467492 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xcf2>
  4675b5:	mov    %r13,%rbx
  4675b8:	jmp    466d90 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x5f0>
  4675bd:	mov    %r11,0x30(%rsp)
  4675c2:	mov    %r10,(%rsp)
  4675c6:	call   46de70 <_ZN5guild7kernels3cpu11cpu_avx2_okEv>
  4675cb:	mov    (%rsp),%r10
  4675cf:	mov    0x30(%rsp),%r11
  4675d4:	test   %al,%al
  4675d6:	jne    4675f0 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xe50>
  4675d8:	mov    0x28d8(%rbx),%rdi
  4675df:	jmp    46696a <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1ca>
  4675e4:	movb   $0x1,0x2910(%rbx)
  4675eb:	jmp    466d90 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x5f0>
  4675f0:	add    0x28c0(%rbx),%r15
  4675f7:	mov    0x8(%r15),%r8d
  4675fb:	test   %r8d,%r8d
  4675fe:	jle    4676b1 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xf11>
  467604:	lea    0x5000(%r12),%rax
  46760c:	mov    %rax,0x110(%rsp)
  467614:	cmp    $0x1,%r8d
  467618:	je     4676b1 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xf11>
  46761e:	lea    0x5e00(%r12),%rax
  467626:	mov    %rax,0x118(%rsp)
  46762e:	cmp    $0x2,%r8d
  467632:	je     4676b1 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xf11>
  467634:	lea    0x6c00(%r12),%rax
  46763c:	mov    %rax,0x120(%rsp)
  467644:	cmp    $0x3,%r8d
  467648:	je     4676b1 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xf11>
  46764a:	lea    0x7a00(%r12),%rax
  467652:	mov    %rax,0x128(%rsp)
  46765a:	cmp    $0x4,%r8d
  46765e:	je     4676b1 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xf11>
  467660:	lea    0x8800(%r12),%rax
  467668:	mov    %rax,0x130(%rsp)
  467670:	cmp    $0x5,%r8d
  467674:	je     4676b1 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xf11>
  467676:	lea    0x9600(%r12),%rax
  46767e:	mov    %rax,0x138(%rsp)
  467686:	cmp    $0x6,%r8d
  46768a:	je     4676b1 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xf11>
  46768c:	lea    0xa400(%r12),%rax
  467694:	mov    %rax,0x140(%rsp)
  46769c:	cmp    $0x7,%r8d
  4676a0:	je     4676b1 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xf11>
  4676a2:	add    $0xb200,%r12
  4676a9:	mov    %r12,0x148(%rsp)
  4676b1:	mov    0x28d8(%rbx),%rax
  4676b8:	mov    %r11,0x30(%rsp)
  4676bd:	lea    0x50(%r15),%r9
  4676c1:	mov    0x18(%rax),%rcx
  4676c5:	mov    0x28(%rax),%rsi
  4676c9:	mov    0x38(%rax),%rdi
  4676cd:	test   %rcx,%rcx
  4676d0:	lea    0x3f(%rcx),%rdx
  4676d4:	cmovns %rcx,%rdx
  4676d8:	add    (%r15),%rdi
  4676db:	push   %rbp
  4676dc:	push   %r10
  4676de:	mov    0x30(%rsp),%rcx
  4676e3:	sar    $0x6,%rdx
  4676e7:	mov    %r10,0x10(%rsp)
  4676ec:	call   46e140 <_ZN5guild7kernels3cpu11q2_rows_anyEPKhmiPKPKNS1_4ActQEiPKPfii>
  4676f1:	pop    %rcx
  4676f2:	pop    %rsi
  4676f3:	mov    (%rsp),%r10
  4676f7:	mov    0x30(%rsp),%r11
  4676fc:	jmp    466a51 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x2b1>
  467701:	mov    %r11,0x30(%rsp)
  467706:	mov    %rdx,(%rsp)
  46770a:	call   46de70 <_ZN5guild7kernels3cpu11cpu_avx2_okEv>
  46770f:	mov    (%rsp),%r10
  467713:	mov    0x30(%rsp),%r11
  467718:	test   %al,%al
  46771a:	je     467a07 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1267>
  467720:	mov    0x28c0(%rbx),%rdi
  467727:	add    %r15,%rdi
  46772a:	mov    0x8(%rdi),%r8d
  46772e:	test   %r8d,%r8d
  467731:	jle    467877 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x10d7>
  467737:	mov    %fs:0x0,%rax
  467740:	lea    -0x1d000(%rax),%rcx
  467747:	lea    -0x22000(%rax),%rdx
  46774e:	mov    %rcx,0xd0(%rsp)
  467756:	mov    %rdx,0x110(%rsp)
  46775e:	cmp    $0x1,%r8d
  467762:	je     467877 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x10d7>
  467768:	lea    -0x1c600(%rax),%rsi
  46776f:	mov    %rsi,0xd8(%rsp)
  467777:	lea    -0x21600(%rax),%rsi
  46777e:	mov    %rsi,0x118(%rsp)
  467786:	cmp    $0x2,%r8d
  46778a:	je     467877 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x10d7>
  467790:	lea    -0x1bc00(%rax),%rsi
  467797:	mov    %rsi,0xe0(%rsp)
  46779f:	lea    -0x20c00(%rax),%rsi
  4677a6:	mov    %rsi,0x120(%rsp)
  4677ae:	cmp    $0x3,%r8d
  4677b2:	je     467877 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x10d7>
  4677b8:	lea    -0x1b200(%rax),%rsi
  4677bf:	mov    %rsi,0xe8(%rsp)
  4677c7:	lea    -0x20200(%rax),%rsi
  4677ce:	mov    %rsi,0x128(%rsp)
  4677d6:	cmp    $0x4,%r8d
  4677da:	je     467877 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x10d7>
  4677e0:	lea    -0x1a800(%rax),%rsi
  4677e7:	mov    %rsi,0xf0(%rsp)
  4677ef:	lea    -0x1f800(%rax),%rsi
  4677f6:	mov    %rsi,0x130(%rsp)
  4677fe:	cmp    $0x5,%r8d
  467802:	je     467877 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x10d7>
  467804:	lea    -0x19e00(%rax),%rcx
  46780b:	lea    -0x1ee00(%rax),%rdx
  467812:	mov    %rcx,0xf8(%rsp)
  46781a:	mov    %rdx,0x138(%rsp)
  467822:	cmp    $0x6,%r8d
  467826:	je     467877 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x10d7>
  467828:	lea    -0x19400(%rax),%rdx
  46782f:	lea    -0x1d000(%rax),%rcx
  467836:	mov    %rdx,0x100(%rsp)
  46783e:	lea    -0x22000(%rax),%rdx
  467845:	add    $0xfffffffffffe1c00,%rax
  46784b:	mov    %rax,0x140(%rsp)
  467853:	cmp    $0x7,%r8d
  467857:	je     467877 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x10d7>
  467859:	lea    0x4600(%rcx),%rax
  467860:	mov    %rax,0x108(%rsp)
  467868:	lea    0x4600(%rdx),%rax
  46786f:	mov    %rax,0x148(%rsp)
  467877:	mov    0x28d8(%rbx),%rax
  46787e:	mov    %r11,0x30(%rsp)
  467883:	lea    0x10(%rdi),%rcx
  467887:	mov    (%rdi),%rdi
  46788a:	mov    0x10(%rax),%rdx
  46788e:	mov    0x20(%rax),%rsi
  467892:	push   %rbp
  467893:	push   %r10
  467895:	test   %rdx,%rdx
  467898:	lea    0x3f(%rdx),%r13
  46789c:	cmovns %rdx,%r13
  4678a0:	lea    0xe0(%rsp),%r9
  4678a8:	mov    %r10,0x10(%rsp)
  4678ad:	sar    $0x6,%r13
  4678b1:	mov    %r13d,%edx
  4678b4:	call   46e140 <_ZN5guild7kernels3cpu11q2_rows_anyEPKhmiPKPKNS1_4ActQEiPKPfii>
  4678b9:	mov    0x28c0(%rbx),%rax
  4678c0:	mov    0x28d8(%rbx),%rdx
  4678c7:	pop    %r9
  4678c9:	pop    %r10
  4678cb:	add    %r15,%rax
  4678ce:	mov    0x30(%rdx),%rdi
  4678d2:	mov    0x20(%rdx),%rsi
  4678d6:	mov    %r13d,%edx
  4678d9:	add    (%rax),%rdi
  4678dc:	mov    0x8(%rax),%r8d
  4678e0:	push   %rbp
  4678e1:	lea    0x10(%rax),%rcx
  4678e5:	mov    0x8(%rsp),%r10
  4678ea:	push   %r10
  4678ec:	mov    0x30(%rsp),%r9
  4678f1:	mov    %r10,0x10(%rsp)
  4678f6:	call   46e140 <_ZN5guild7kernels3cpu11q2_rows_anyEPKhmiPKPKNS1_4ActQEiPKPfii>
  4678fb:	mov    0x28c0(%rbx),%rax
  467902:	pop    %r11
  467904:	pop    %r13
  467906:	mov    (%rsp),%r10
  46790a:	movslq 0x8(%rax,%r15,1),%rax
  46790f:	mov    0x30(%rsp),%r11
  467914:	test   %eax,%eax
  467916:	jle    466a51 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x2b1>
  46791c:	cmp    %ebp,%r10d
  46791f:	jge    466a51 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x2b1>
  467925:	lea    (%rax,%rax,4),%rax
  467929:	mov    %ebp,%ecx
  46792b:	mov    %r14,0x58(%rsp)
  467930:	lea    0x0(,%r10,4),%rdx
  467938:	shl    $0x7,%rax
  46793c:	sub    %r10d,%ecx
  46793f:	mov    %r11,0x60(%rsp)
  467944:	add    %r10,%rax
  467947:	sub    $0x1,%ecx
  46794a:	mov    %r10,0x78(%rsp)
  46794f:	shl    $0x2,%rax
  467953:	lea    0x1(%r10,%rcx,1),%r13
  467958:	mov    %rbp,0x38(%rsp)
  46795d:	mov    %rax,0x30(%rsp)
  467962:	shl    $0x2,%r13
  467966:	mov    %fs:0x0,%rax
  46796f:	mov    %rbx,0x68(%rsp)
  467974:	mov    %rdx,%rbx
  467977:	lea    -0x22000(%rax),%r15
  46797e:	mov    %rax,%rbp
  467981:	nopl   0x0(%rax)
  467988:	mov    %rbx,%r14
  46798b:	nopl   0x0(%rax,%rax,1)
  467990:	movss  -0x1d000(%r14,%rbp,1),%xmm1
  46799a:	movaps %xmm1,%xmm0
  46799d:	xorps  0x218dcc(%rip),%xmm0        # 680770 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0x2950>
  4679a4:	movss  %xmm1,(%rsp)
  4679a9:	call   13b60 <expf@plt>
  4679ae:	addss  0x1d064e(%rip),%xmm0        # 638004 <_IO_stdin_used+0x4>
  4679b6:	movss  (%rsp),%xmm1
  4679bb:	divss  %xmm0,%xmm1
  4679bf:	mulss  (%r14,%r15,1),%xmm1
  4679c5:	movss  %xmm1,(%r12,%r14,1)
  4679cb:	add    $0x4,%r14
  4679cf:	cmp    %r13,%r14
  4679d2:	jne    467990 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x11f0>
  4679d4:	add    $0xa00,%rbx
  4679db:	add    $0xa00,%r13
  4679e2:	cmp    %rbx,0x30(%rsp)
  4679e7:	jne    467988 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x11e8>
  4679e9:	mov    0x38(%rsp),%rbp
  4679ee:	mov    0x58(%rsp),%r14
  4679f3:	mov    0x60(%rsp),%r11
  4679f8:	mov    0x68(%rsp),%rbx
  4679fd:	mov    0x78(%rsp),%r10
  467a02:	jmp    466a51 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x2b1>
  467a07:	cmpl   $0x5,0x2898(%rbx)
  467a0e:	mov    0x28d8(%rbx),%rdi
  467a15:	jne    466960 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1c0>
  467a1b:	jmp    466ac2 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x322>
  467a20:	call   13650 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  467a25:	mov    0x28d8(%rbx),%rdi
  467a2c:	mov    %rax,0x78(%rsp)
  467a31:	cmpl   $0x2a,(%rdi)
  467a34:	jne    466e1a <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x67a>
  467a3a:	call   46de70 <_ZN5guild7kernels3cpu11cpu_avx2_okEv>
  467a3f:	test   %al,%al
  467a41:	je     467fba <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x181a>
  467a47:	mov    0x28d8(%rbx),%rsi
  467a4e:	mov    0x10(%rsi),%rdx
  467a52:	test   %rdx,%rdx
  467a55:	lea    0x3f(%rdx),%rax
  467a59:	cmovns %rdx,%rax
  467a5d:	sar    $0x6,%rax
  467a61:	mov    %eax,0xa0(%rsp)
  467a68:	test   %ebp,%ebp
  467a6a:	jle    466ef4 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x754>
  467a70:	mov    0x28(%rsp),%rcx
  467a75:	lea    -0x8(%rbp),%edi
  467a78:	mov    0x18(%rsi),%rdx
  467a7c:	mov    %ebp,%eax
  467a7e:	mov    %fs:0x0,%r11
  467a87:	mov    %ebp,0xb8(%rsp)
  467a8e:	add    $0x10,%rcx
  467a92:	mov    %r13d,0xbc(%rsp)
  467a9a:	mov    %rcx,0x30(%rsp)
  467a9f:	lea    -0x5000(%r11),%rcx
  467aa6:	lea    -0x13000(%r11),%r10
  467aad:	mov    %rcx,0x60(%rsp)
  467ab2:	lea    -0x1(%rbp),%ecx
  467ab5:	and    $0xfffffff8,%ecx
  467ab8:	mov    %r11,0x68(%rsp)
  467abd:	sub    %ecx,%edi
  467abf:	lea    0xd0(%rsp),%rcx
  467ac7:	mov    %rcx,0xa8(%rsp)
  467acf:	lea    0x110(%rsp),%rcx
  467ad7:	mov    %rcx,0x20(%rsp)
  467adc:	lea    -0x12600(%r11),%rcx
  467ae3:	mov    %edi,0xa4(%rsp)
  467aea:	mov    %rcx,0xb0(%rsp)
  467af2:	nopw   0x0(%rax,%rax,1)
  467af8:	mov    $0x8,%ebp
  467afd:	mov    0x68(%rsp),%rcx
  467b02:	mov    %r10,0xd0(%rsp)
  467b0a:	cmp    %ebp,%eax
  467b0c:	cmovle %eax,%ebp
  467b0f:	add    $0xfffffffffffe8000,%rcx
  467b16:	mov    %rcx,0x110(%rsp)
  467b1e:	cmp    $0x1,%eax
  467b21:	jle    467c24 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1484>
  467b27:	mov    0xb0(%rsp),%rdi
  467b2f:	mov    %rdi,0xd8(%rsp)
  467b37:	lea    0xa00(%rcx),%rdi
  467b3e:	mov    %rdi,0x118(%rsp)
  467b46:	cmp    $0x2,%eax
  467b49:	jle    467c24 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1484>
  467b4f:	lea    0x1400(%r10),%rdi
  467b56:	mov    %rdi,0xe0(%rsp)
  467b5e:	lea    0x1400(%rcx),%rdi
  467b65:	mov    %rdi,0x120(%rsp)
  467b6d:	cmp    $0x3,%eax
  467b70:	jle    467c24 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1484>
  467b76:	lea    0x1e00(%r10),%rdi
  467b7d:	mov    %rdi,0xe8(%rsp)
  467b85:	lea    0x1e00(%rcx),%rdi
  467b8c:	mov    %rdi,0x128(%rsp)
  467b94:	cmp    $0x4,%eax
  467b97:	jle    467c24 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1484>
  467b9d:	lea    0x2800(%r10),%rdi
  467ba4:	mov    %rdi,0xf0(%rsp)
  467bac:	lea    0x2800(%rcx),%rdi
  467bb3:	mov    %rdi,0x130(%rsp)
  467bbb:	cmp    $0x5,%eax
  467bbe:	jle    467c24 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1484>
  467bc0:	lea    0x3200(%r10),%rdi
  467bc7:	mov    %rdi,0xf8(%rsp)
  467bcf:	lea    0x3200(%rcx),%rdi
  467bd6:	mov    %rdi,0x138(%rsp)
  467bde:	cmp    $0x6,%eax
  467be1:	jle    467c24 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1484>
  467be3:	lea    0x3c00(%r10),%rdi
  467bea:	mov    %rdi,0x100(%rsp)
  467bf2:	lea    0x3c00(%rcx),%rdi
  467bf9:	mov    %rdi,0x140(%rsp)
  467c01:	cmp    $0x7,%eax
  467c04:	jle    467c24 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1484>
  467c06:	lea    0x4600(%r10),%rdi
  467c0d:	add    $0x4600,%rcx
  467c14:	mov    %rdi,0x108(%rsp)
  467c1c:	mov    %rcx,0x148(%rsp)
  467c24:	mov    0x28(%rsp),%r14
  467c29:	mov    0x20(%rsi),%rsi
  467c2d:	mov    %eax,(%rsp)
  467c30:	mov    %ebp,%r8d
  467c33:	mov    %r10,0x58(%rsp)
  467c38:	mov    (%r14),%rdi
  467c3b:	push   %rdx
  467c3c:	push   $0x0
  467c3e:	mov    0x40(%rsp),%r13
  467c43:	mov    0xb0(%rsp),%r15d
  467c4b:	mov    0xb8(%rsp),%r9
  467c53:	mov    %r13,%rcx
  467c56:	mov    %r15d,%edx
  467c59:	call   46e140 <_ZN5guild7kernels3cpu11q2_rows_anyEPKhmiPKPKNS1_4ActQEiPKPfii>
  467c5e:	pop    %r12
  467c60:	pop    %rax
  467c61:	mov    %r13,%rcx
  467c64:	mov    0x28d8(%rbx),%rdx
  467c6b:	mov    %ebp,%r8d
  467c6e:	mov    0x20(%rdx),%rsi
  467c72:	mov    0x30(%rdx),%rdi
  467c76:	mov    0x18(%rdx),%rdx
  467c7a:	add    (%r14),%rdi
  467c7d:	push   %rdx
  467c7e:	mov    %r15d,%edx
  467c81:	push   $0x0
  467c83:	mov    0x30(%rsp),%r9
  467c88:	call   46e140 <_ZN5guild7kernels3cpu11q2_rows_anyEPKhmiPKPKNS1_4ActQEiPKPfii>
  467c8d:	mov    0x28d8(%rbx),%rsi
  467c94:	pop    %rax
  467c95:	pop    %rcx
  467c96:	mov    (%rsp),%eax
  467c99:	mov    0x18(%rsi),%rdx
  467c9d:	mov    0x58(%rsp),%r10
  467ca2:	test   %edx,%edx
  467ca4:	jle    467d8a <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x15ea>
  467caa:	mov    0x68(%rsp),%rdi
  467caf:	movslq %ebp,%rbp
  467cb2:	mov    %eax,0x58(%rsp)
  467cb6:	lea    -0x1(%rdx),%r13d
  467cba:	lea    0x0(%rbp,%rbp,4),%r14
  467cbf:	mov    %rsi,0x80(%rsp)
  467cc7:	mov    0x60(%rsp),%rbp
  467ccc:	mov    %rdx,0x88(%rsp)
  467cd4:	lea    -0x13000(%rdi),%r15
  467cdb:	shl    $0x9,%r14
  467cdf:	lea    -0x18000(%rdi),%r12
  467ce6:	mov    %r10,0x98(%rsp)
  467cee:	add    %r15,%r14
  467cf1:	mov    %rbx,0x90(%rsp)
  467cf9:	mov    %r15,%rbx
  467cfc:	nopl   0x0(%rax)
  467d00:	xor    %r15d,%r15d
  467d03:	nopl   0x0(%rax,%rax,1)
  467d08:	movss  (%rbx,%r15,4),%xmm1
  467d0e:	movaps %xmm1,%xmm0
  467d11:	xorps  0x218a58(%rip),%xmm0        # 680770 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0x2950>
  467d18:	movss  %xmm1,(%rsp)
  467d1d:	call   13b60 <expf@plt>
  467d22:	addss  0x1d02da(%rip),%xmm0        # 638004 <_IO_stdin_used+0x4>
  467d2a:	movss  (%rsp),%xmm1
  467d2f:	mov    %r15,%rax
  467d32:	divss  %xmm0,%xmm1
  467d36:	mulss  (%r12,%r15,4),%xmm1
  467d3c:	movss  %xmm1,0x0(%rbp,%r15,4)
  467d43:	add    $0x1,%r15
  467d47:	cmp    %rax,%r13
  467d4a:	jne    467d08 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1568>
  467d4c:	add    $0xa00,%rbx
  467d53:	add    $0xa00,%r12
  467d5a:	add    $0xa00,%rbp
  467d61:	cmp    %rbx,%r14
  467d64:	jne    467d00 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1560>
  467d66:	mov    0x58(%rsp),%eax
  467d6a:	mov    0x80(%rsp),%rsi
  467d72:	mov    0x88(%rsp),%rdx
  467d7a:	mov    0x90(%rsp),%rbx
  467d82:	mov    0x98(%rsp),%r10
  467d8a:	mov    0xa4(%rsp),%ecx
  467d91:	sub    $0x8,%eax
  467d94:	addq   $0x40,0x30(%rsp)
  467d9a:	addq   $0x5000,0x60(%rsp)
  467da3:	cmp    %ecx,%eax
  467da5:	jne    467af8 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1358>
  467dab:	mov    0xb8(%rsp),%ebp
  467db2:	mov    0xbc(%rsp),%r13d
  467dba:	jmp    466ef4 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x754>
  467dbf:	call   13650 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  467dc4:	mov    0x58(%rsp),%rsi
  467dc9:	mov    0x78(%rsp),%rcx
  467dce:	pxor   %xmm0,%xmm0
  467dd2:	pxor   %xmm1,%xmm1
  467dd6:	movupd 0x18(%rbx),%xmm6
  467ddb:	sub    %r12,%rax
  467dde:	mov    %rsi,%rdx
  467de1:	sub    %rcx,%rdx
  467de4:	cvtsi2sd %rdx,%xmm0
  467de9:	mov    %r12,%rdx
  467dec:	sub    %rsi,%rdx
  467def:	cvtsi2sd %rdx,%xmm1
  467df4:	unpcklpd %xmm1,%xmm0
  467df8:	movsd  0x219120(%rip),%xmm1        # 680f20 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0x3100>
  467e00:	unpcklpd %xmm1,%xmm1
  467e04:	divpd  %xmm1,%xmm0
  467e08:	addpd  %xmm6,%xmm0
  467e0c:	movups %xmm0,0x18(%rbx)
  467e10:	pxor   %xmm0,%xmm0
  467e14:	cvtsi2sd %rax,%xmm0
  467e19:	divsd  0x2190ff(%rip),%xmm0        # 680f20 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0x3100>
  467e21:	addsd  0x28(%rbx),%xmm0
  467e26:	movsd  %xmm0,0x28(%rbx)
  467e2b:	jmp    46709b <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x8fb>
  467e30:	call   13650 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  467e35:	mov    %rax,0x58(%rsp)
  467e3a:	jmp    466f0b <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x76b>
  467e3f:	call   46de70 <_ZN5guild7kernels3cpu11cpu_avx2_okEv>
  467e44:	test   %al,%al
  467e46:	je     466f80 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x7e0>
  467e4c:	test   %ebp,%ebp
  467e4e:	jle    467ebe <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x171e>
  467e50:	mov    %fs:0x0,%r11
  467e59:	movslq %ebp,%rdx
  467e5c:	lea    0x110(%rsp),%r14
  467e64:	mov    %ebp,(%rsp)
  467e67:	lea    (%r14,%rdx,8),%r15
  467e6b:	mov    %rbx,%rbp
  467e6e:	lea    -0x5000(%r11),%rcx
  467e75:	mov    %r15,%rbx
  467e78:	lea    -0xc000(%r11),%r12
  467e7f:	mov    %rcx,%r15
  467e82:	nopw   0x0(%rax,%rax,1)
  467e88:	mov    0x28d8(%rbp),%rax
  467e8f:	mov    %r12,%rdx
  467e92:	mov    %r15,%rdi
  467e95:	add    $0x8,%r14
  467e99:	add    $0xa00,%r15
  467ea0:	mov    0x18(%rax),%esi
  467ea3:	call   46e030 <_ZN5guild7kernels3cpu13act_quant_anyEPKfiRNS1_4ActQE>
  467ea8:	mov    %r12,-0x8(%r14)
  467eac:	add    $0xe00,%r12
  467eb3:	cmp    %r14,%rbx
  467eb6:	jne    467e88 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x16e8>
  467eb8:	mov    %rbp,%rbx
  467ebb:	mov    (%rsp),%ebp
  467ebe:	mov    0x1c(%rsp),%r8d
  467ec3:	xor    %r12d,%r12d
  467ec6:	test   %r8d,%r8d
  467ec9:	js     467fc6 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1826>
  467ecf:	cmpq   $0x0,0x50(%rbx)
  467ed4:	movq   $0x0,(%rsp)
  467edc:	je     467f31 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1791>
  467ede:	pxor   %xmm0,%xmm0
  467ee2:	lea    0xc0(%rsp),%rsi
  467eea:	mov    $0x3,%edi
  467eef:	movaps %xmm0,0xc0(%rsp)
  467ef7:	call   131a0 <clock_gettime@plt>
  467efc:	pxor   %xmm1,%xmm1
  467f00:	pxor   %xmm0,%xmm0
  467f04:	cvtsi2sdq 0xc0(%rsp),%xmm1
  467f0e:	mulsd  0x215faa(%rip),%xmm1        # 67dec0 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0xa0>
  467f16:	cvtsi2sdq 0xc8(%rsp),%xmm0
  467f20:	divsd  0x218ff8(%rip),%xmm0        # 680f20 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0x3100>
  467f28:	addsd  %xmm0,%xmm1
  467f2c:	movsd  %xmm1,(%rsp)
  467f31:	xor    %edx,%edx
  467f33:	lea    0x110(%rsp),%r14
  467f3b:	cmp    %ebp,%edx
  467f3d:	jge    46708d <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x8ed>
  467f43:	mov    0x28d8(%rbx),%rax
  467f4a:	mov    %ebp,%r8d
  467f4d:	mov    0x18(%rax),%rcx
  467f51:	mov    0x28(%rax),%rsi
  467f55:	mov    0x38(%rax),%rdi
  467f59:	mov    0x10(%rax),%rax
  467f5d:	test   %rcx,%rcx
  467f60:	lea    0x3f(%rcx),%rdx
  467f64:	cmovns %rcx,%rdx
  467f68:	mov    0x28(%rsp),%rcx
  467f6d:	add    (%rcx),%rdi
  467f70:	push   %rax
  467f71:	sar    $0x6,%rdx
  467f75:	mov    %r14,%rcx
  467f78:	push   $0x0
  467f7a:	mov    0x48(%rsp),%r9
  467f7f:	call   46e140 <_ZN5guild7kernels3cpu11q2_rows_anyEPKhmiPKPKNS1_4ActQEiPKPfii>
  467f84:	mov    %r13d,%edx
  467f87:	pop    %rsi
  467f88:	pop    %rdi
  467f89:	cmp    %ebp,%edx
  467f8b:	jl     467f43 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x17a3>
  467f8d:	jmp    46708d <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x8ed>
  467f92:	mov    0x18(%rdi),%ecx
  467f95:	jmp    466906 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x166>
  467f9a:	call   13650 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  467f9f:	cmpq   $0x0,0x50(%rbx)
  467fa4:	movq   $0x0,(%rsp)
  467fac:	mov    %rax,%r12
  467faf:	jne    46700c <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x86c>
  467fb5:	jmp    46705f <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x8bf>
  467fba:	mov    0x28d8(%rbx),%rdi
  467fc1:	jmp    466e1a <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x67a>
  467fc6:	call   13650 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  467fcb:	mov    %rax,%r12
  467fce:	jmp    467ecf <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x172f>
  467fd3:	lea    0x110(%rsp),%rax
  467fdb:	mov    %rax,0x20(%rsp)
  467fe0:	jmp    466fee <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x84e>
  467fe5:	call   13390 <__stack_chk_fail@plt>

Disassembly of section .fini:
