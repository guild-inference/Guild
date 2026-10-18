
/home/ubuntu/Guild/build-cuda12/ornith_prefill_bench:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

00000000004f07a0 <_ZN5guild7kernels3cpu13kq256_gu_rowsEiPKhmmiPKPKviPKPfii>:
  4f07a0:	endbr64
  4f07a4:	push   %r15
  4f07a6:	push   %r14
  4f07a8:	push   %r13
  4f07aa:	push   %r12
  4f07ac:	push   %rbp
  4f07ad:	push   %rbx
  4f07ae:	mov    %rdx,%rbx
  4f07b1:	sub    $0xb8,%rsp
  4f07b8:	mov    0xf8(%rsp),%rax
  4f07c0:	mov    %edi,0x30(%rsp)
  4f07c4:	mov    %rsi,%rdi
  4f07c7:	mov    %rcx,0x40(%rsp)
  4f07cc:	mov    0x100(%rsp),%esi
  4f07d3:	mov    %rax,0x50(%rsp)
  4f07d8:	mov    0x108(%rsp),%eax
  4f07df:	mov    %r8d,0x34(%rsp)
  4f07e4:	mov    %r9,0x48(%rsp)
  4f07e9:	mov    %fs:0x28,%rdx
  4f07f2:	mov    %rdx,0xa8(%rsp)
  4f07fa:	xor    %edx,%edx
  4f07fc:	cmp    %eax,%esi
  4f07fe:	jge    4f0963 <_ZN5guild7kernels3cpu13kq256_gu_rowsEiPKhmmiPKPKviPKPfii+0x1c3>
  4f0804:	mov    0xf0(%rsp),%ecx
  4f080b:	test   %ecx,%ecx
  4f080d:	jle    4f0963 <_ZN5guild7kernels3cpu13kq256_gu_rowsEiPKhmmiPKPKviPKPfii+0x1c3>
  4f0813:	movslq %esi,%rdx
  4f0816:	sub    %esi,%eax
  4f0818:	mov    %rbx,%rcx
  4f081b:	mov    %rbx,0x58(%rsp)
  4f0820:	add    %rdx,%rax
  4f0823:	imul   %rdx,%rcx
  4f0827:	lea    0x0(,%rdx,4),%r14
  4f082f:	shl    $0x2,%rax
  4f0833:	lea    0x80(%rsp),%r12
  4f083b:	lea    0x60(%rsp),%r15
  4f0840:	mov    %rax,0x38(%rsp)
  4f0845:	mov    0xf0(%rsp),%eax
  4f084c:	add    %rdi,%rcx
  4f084f:	sub    $0x1,%eax
  4f0852:	and    $0xfffffff8,%eax
  4f0855:	add    $0x8,%eax
  4f0858:	mov    %eax,0x2c(%rsp)
  4f085c:	nopl   0x0(%rax)
  4f0860:	mov    0x40(%rsp),%rax
  4f0865:	mov    0x50(%rsp),%rbp
  4f086a:	mov    %rcx,0x18(%rsp)
  4f086f:	movl   $0x0,0x28(%rsp)
  4f0877:	add    %rcx,%rax
  4f087a:	mov    %rax,0x20(%rsp)
  4f087f:	mov    0x48(%rsp),%rax
  4f0884:	mov    %rax,0x10(%rsp)
  4f0889:	nopl   0x0(%rax)
  4f0890:	mov    0x28(%rsp),%eax
  4f0894:	mov    0xf0(%rsp),%ebx
  4f089b:	push   %r12
  4f089d:	push   %r15
  4f089f:	mov    0x20(%rsp),%r8
  4f08a4:	sub    %eax,%ebx
  4f08a6:	mov    $0x8,%eax
  4f08ab:	mov    0x44(%rsp),%ecx
  4f08af:	mov    0x30(%rsp),%rdx
  4f08b4:	cmp    %eax,%ebx
  4f08b6:	mov    0x28(%rsp),%rsi
  4f08bb:	mov    0x40(%rsp),%edi
  4f08bf:	cmovg  %eax,%ebx
  4f08c2:	xor    %r13d,%r13d
  4f08c5:	mov    %ebx,%r9d
  4f08c8:	movslq %ebx,%rbx
  4f08cb:	call   4efba0 <_ZN5guild7kernels3cpu12_GLOBAL__N_19dot_2rowsEiPKhS4_iPKPKviPfS9_>
  4f08d0:	shl    $0x2,%rbx
  4f08d4:	pop    %rax
  4f08d5:	pop    %rdx
  4f08d6:	cs nopw 0x0(%rax,%rax,1)
  4f08e0:	vmovss (%r15,%r13,1),%xmm1
  4f08e6:	vxorps 0x1acd02(%rip),%xmm1,%xmm0        # 69d5f0 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0x2a90>
  4f08ee:	vmovss %xmm1,0xc(%rsp)
  4f08f4:	call   14bd0 <expf@plt>
  4f08f9:	vmovss 0xc(%rsp),%xmm1
  4f08ff:	mov    0x0(%rbp,%r13,2),%rax
  4f0904:	vaddss 0x1616f8(%rip),%xmm0,%xmm0        # 652004 <_IO_stdin_used+0x4>
  4f090c:	vdivss %xmm0,%xmm1,%xmm0
  4f0910:	vmulss (%r12,%r13,1),%xmm0,%xmm0
  4f0916:	add    $0x4,%r13
  4f091a:	vmovss %xmm0,(%rax,%r14,1)
  4f0920:	cmp    %r13,%rbx
  4f0923:	jne    4f08e0 <_ZN5guild7kernels3cpu13kq256_gu_rowsEiPKhmmiPKPKviPKPfii+0x140>
  4f0925:	addl   $0x8,0x28(%rsp)
  4f092a:	mov    0x2c(%rsp),%esi
  4f092e:	add    $0x40,%rbp
  4f0932:	mov    0x28(%rsp),%eax
  4f0936:	addq   $0x40,0x10(%rsp)
  4f093c:	cmp    %esi,%eax
  4f093e:	jne    4f0890 <_ZN5guild7kernels3cpu13kq256_gu_rowsEiPKhmmiPKPKviPKPfii+0xf0>
  4f0944:	mov    0x58(%rsp),%rax
  4f0949:	mov    0x18(%rsp),%rcx
  4f094e:	add    $0x4,%r14
  4f0952:	add    %rax,%rcx
  4f0955:	mov    0x38(%rsp),%rax
  4f095a:	cmp    %rax,%r14
  4f095d:	jne    4f0860 <_ZN5guild7kernels3cpu13kq256_gu_rowsEiPKhmmiPKPKviPKPfii+0xc0>
  4f0963:	mov    0xa8(%rsp),%rax
  4f096b:	sub    %fs:0x28,%rax
  4f0974:	jne    4f0988 <_ZN5guild7kernels3cpu13kq256_gu_rowsEiPKhmmiPKPKviPKPfii+0x1e8>
  4f0976:	add    $0xb8,%rsp
  4f097d:	pop    %rbx
  4f097e:	pop    %rbp
  4f097f:	pop    %r12
  4f0981:	pop    %r13
  4f0983:	pop    %r14
  4f0985:	pop    %r15
  4f0987:	ret
  4f0988:	call   143e0 <__stack_chk_fail@plt>

Disassembly of section .fini:
