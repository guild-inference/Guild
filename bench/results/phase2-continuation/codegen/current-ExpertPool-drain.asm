
build-cuda12/ornith_prefill_bench:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

0000000000467bd0 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj>:
  467bd0:	endbr64
  467bd4:	push   %r15
  467bd6:	push   %r14
  467bd8:	lea    0x1(%rsi),%r14d
  467bdc:	push   %r13
  467bde:	push   %r12
  467be0:	push   %rbp
  467be1:	push   %rbx
  467be2:	mov    %rdi,%rbx
  467be5:	sub    $0x168,%rsp
  467bec:	mov    %rdx,0x50(%rsp)
  467bf1:	mov    %ecx,0x70(%rsp)
  467bf5:	mov    %fs:0x28,%rax
  467bfe:	mov    %rax,0x158(%rsp)
  467c06:	xor    %eax,%eax
  467c08:	movslq %esi,%rax
  467c0b:	mov    %esi,0x1c(%rsp)
  467c0f:	shl    $0x2,%rax
  467c13:	mov    %r14d,0x74(%rsp)
  467c18:	mov    %rax,0x40(%rsp)
  467c1d:	movslq %r14d,%rax
  467c20:	shl    $0x6,%rax
  467c24:	mov    %rax,0x48(%rsp)
  467c29:	lea    0x2700(%rdi),%rax
  467c30:	mov    %rax,0x8(%rsp)
  467c35:	nopl   (%rax)
  467c38:	mov    0x8(%rsp),%rax
  467c3d:	mov    (%rax),%rax
  467c40:	mov    %rax,%rdx
  467c43:	shr    $0x20,%rdx
  467c47:	cmp    %edx,0x70(%rsp)
  467c4b:	jne    46870a <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xb3a>
  467c51:	mov    %eax,%edx
  467c53:	movzwl %ax,%ebp
  467c56:	shr    $0x10,%edx
  467c59:	cmp    %edx,%ebp
  467c5b:	jae    46870a <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xb3a>
  467c61:	mov    0x8(%rsp),%rsi
  467c66:	lea    0x1(%rax),%rdx
  467c6a:	lock cmpxchg %rdx,(%rsi)
  467c6f:	jne    467c40 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x70>
  467c71:	mov    0x1c(%rsp),%r8d
  467c76:	test   %r8d,%r8d
  467c79:	js     4686d8 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xb08>
  467c7f:	mov    0x26d8(%rbx),%rax
  467c86:	mov    0x40(%rsp),%rdi
  467c8b:	mov    %ebp,(%rax,%rdi,1)
  467c8e:	cmpq   $0x0,0x80(%rbx)
  467c96:	movq   $0x0,0x10(%rsp)
  467c9f:	je     467cf5 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x125>
  467ca1:	pxor   %xmm0,%xmm0
  467ca5:	lea    0xc0(%rsp),%rsi
  467cad:	mov    $0x3,%edi
  467cb2:	movaps %xmm0,0xc0(%rsp)
  467cba:	call   131a0 <clock_gettime@plt>
  467cbf:	pxor   %xmm1,%xmm1
  467cc3:	pxor   %xmm0,%xmm0
  467cc7:	cvtsi2sdq 0xc0(%rsp),%xmm1
  467cd1:	mulsd  0x2173c7(%rip),%xmm1        # 67f0a0 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0xa0>
  467cd9:	cvtsi2sdq 0xc8(%rsp),%xmm0
  467ce3:	divsd  0x21a365(%rip),%xmm0        # 682050 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0x3050>
  467ceb:	addsd  %xmm0,%xmm1
  467cef:	movsd  %xmm1,0x10(%rsp)
  467cf5:	mov    0x28d8(%rbx),%r13d
  467cfc:	cmp    $0xa,%r13d
  467d00:	ja     467d20 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x150>
  467d02:	lea    0x27fc57(%rip),%rdx        # 6e7960 <_ZN5guild4spec12_GLOBAL__N_1L6kShapeE+0xc80>
  467d09:	mov    %r13d,%eax
  467d0c:	movslq (%rdx,%rax,4),%rax
  467d10:	add    %rdx,%rax
  467d13:	notrack jmp *%rax
  467d16:	cs nopw 0x0(%rax,%rax,1)
  467d20:	cmp    $0x4,%r13d
  467d24:	jle    468779 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xba9>
  467d2a:	mov    0x2918(%rbx),%rdi
  467d31:	cmp    $0x5,%r13d
  467d35:	je     4693b5 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x17e5>
  467d3b:	mov    0x10(%rdi),%ecx
  467d3e:	mov    0x2908(%rbx),%r8
  467d45:	mov    %ebp,%eax
  467d47:	movslq 0x2910(%rbx),%rsi
  467d4e:	imul   %r8,%rax
  467d52:	cqto
  467d54:	idiv   %rsi
  467d57:	mov    %rax,%r10
  467d5a:	lea    0x1(%rbp),%eax
  467d5d:	imul   %r8,%rax
  467d61:	cqto
  467d63:	idiv   %rsi
  467d66:	mov    %rax,%r9
  467d69:	cmp    %rax,%r10
  467d6c:	jge    4680d8 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x508>
  467d72:	lea    0x110(%rsp),%rax
  467d7a:	mov    %r9,0x28(%rsp)
  467d7f:	movslq %ecx,%r11
  467d82:	mov    %r10,%r14
  467d85:	mov    %rax,0x20(%rsp)
  467d8a:	jmp    467ea1 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x2d1>
  467d8f:	nop
  467d90:	cmpl   $0x2a,0x4(%rdi)
  467d94:	je     4689dd <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xe0d>
  467d9a:	add    0x2900(%rbx),%r15
  467da1:	mov    0x8(%r15),%ecx
  467da5:	test   %ecx,%ecx
  467da7:	jle    467e53 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x283>
  467dad:	lea    0xc000(%r12),%rax
  467db5:	mov    %rax,0x110(%rsp)
  467dbd:	cmp    $0x1,%ecx
  467dc0:	je     467e53 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x283>
  467dc6:	lea    0xc400(%r12),%rax
  467dce:	mov    %rax,0x118(%rsp)
  467dd6:	cmp    $0x2,%ecx
  467dd9:	je     467e53 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x283>
  467ddb:	lea    0xc800(%r12),%rax
  467de3:	mov    %rax,0x120(%rsp)
  467deb:	cmp    $0x3,%ecx
  467dee:	je     467e53 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x283>
  467df0:	lea    0xcc00(%r12),%rax
  467df8:	mov    %rax,0x128(%rsp)
  467e00:	cmp    $0x4,%ecx
  467e03:	je     467e53 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x283>
  467e05:	lea    0xd000(%r12),%rax
  467e0d:	mov    %rax,0x130(%rsp)
  467e15:	cmp    $0x5,%ecx
  467e18:	je     467e53 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x283>
  467e1a:	lea    0xd400(%r12),%rax
  467e22:	mov    %rax,0x138(%rsp)
  467e2a:	cmp    $0x6,%ecx
  467e2d:	je     467e53 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x283>
  467e2f:	lea    0xd800(%r12),%rax
  467e37:	mov    %rax,0x140(%rsp)
  467e3f:	cmp    $0x7,%ecx
  467e42:	je     467e53 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x283>
  467e44:	add    $0xdc00,%r12
  467e4b:	mov    %r12,0x148(%rsp)
  467e53:	mov    %r11,0x30(%rsp)
  467e58:	sub    $0x8,%rsp
  467e5c:	mov    (%r15),%rsi
  467e5f:	mov    %r10d,%r9d
  467e62:	push   %rbp
  467e63:	lea    0x50(%r15),%r8
  467e67:	mov    0x30(%rsp),%rdx
  467e6c:	mov    %r10,0x10(%rsp)
  467e71:	call   *0x60(%rbx)
  467e74:	pop    %rax
  467e75:	pop    %rdx
  467e76:	mov    (%rsp),%r10
  467e7a:	mov    0x30(%rsp),%r11
  467e7f:	sub    %r10d,%ebp
  467e82:	movslq %ebp,%rbp
  467e85:	add    %rbp,%r14
  467e88:	cmp    %r14,0x28(%rsp)
  467e8d:	jle    4681a8 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x5d8>
  467e93:	mov    0x28d8(%rbx),%r13d
  467e9a:	mov    0x2918(%rbx),%rdi
  467ea1:	mov    %r14,%rax
  467ea4:	mov    0x28(%rsp),%rbp
  467ea9:	cqto
  467eab:	idiv   %r11
  467eae:	sub    %r14,%rbp
  467eb1:	add    %rdx,%rbp
  467eb4:	cltq
  467eb6:	mov    %rdx,%r10
  467eb9:	cmp    %r11,%rbp
  467ebc:	lea    0x0(,%rax,8),%r15
  467ec4:	cmovg  %r11,%rbp
  467ec8:	sub    %rax,%r15
  467ecb:	mov    %r15,%r12
  467ece:	shl    $0x5,%r15
  467ed2:	shl    $0xd,%r12
  467ed6:	add    0x2958(%rbx),%r12
  467edd:	cmp    $0x5,%r13d
  467ee1:	jne    467d90 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1c0>
  467ee7:	cmpl   $0x2a,(%rdi)
  467eea:	je     468b21 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xf51>
  467ef0:	add    0x2900(%rbx),%r15
  467ef7:	mov    0x8(%r15),%ecx
  467efb:	test   %ecx,%ecx
  467efd:	jle    467fa1 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x3d1>
  467f03:	mov    %r12,0x110(%rsp)
  467f0b:	cmp    $0x1,%ecx
  467f0e:	je     467fa1 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x3d1>
  467f14:	lea    0xa00(%r12),%rax
  467f1c:	mov    %rax,0x118(%rsp)
  467f24:	cmp    $0x2,%ecx
  467f27:	je     467fa1 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x3d1>
  467f29:	lea    0x1400(%r12),%rax
  467f31:	mov    %rax,0x120(%rsp)
  467f39:	cmp    $0x3,%ecx
  467f3c:	je     467fa1 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x3d1>
  467f3e:	lea    0x1e00(%r12),%rax
  467f46:	mov    %rax,0x128(%rsp)
  467f4e:	cmp    $0x4,%ecx
  467f51:	je     467fa1 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x3d1>
  467f53:	lea    0x2800(%r12),%rax
  467f5b:	mov    %rax,0x130(%rsp)
  467f63:	cmp    $0x5,%ecx
  467f66:	je     467fa1 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x3d1>
  467f68:	lea    0x3200(%r12),%rax
  467f70:	mov    %rax,0x138(%rsp)
  467f78:	cmp    $0x6,%ecx
  467f7b:	je     467fa1 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x3d1>
  467f7d:	lea    0x3c00(%r12),%rax
  467f85:	mov    %rax,0x140(%rsp)
  467f8d:	cmp    $0x7,%ecx
  467f90:	je     467fa1 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x3d1>
  467f92:	add    $0x4600,%r12
  467f99:	mov    %r12,0x148(%rsp)
  467fa1:	mov    %r11,0x30(%rsp)
  467fa6:	sub    $0x8,%rsp
  467faa:	mov    (%r15),%rsi
  467fad:	mov    %r10d,%r9d
  467fb0:	push   %rbp
  467fb1:	lea    0x90(%r15),%rdx
  467fb8:	mov    0x30(%rsp),%r8
  467fbd:	mov    %r10,0x10(%rsp)
  467fc2:	call   *0x58(%rbx)
  467fc5:	pop    %rdi
  467fc6:	pop    %r8
  467fc8:	mov    (%rsp),%r10
  467fcc:	mov    0x30(%rsp),%r11
  467fd1:	jmp    467e7f <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x2af>
  467fd6:	movslq 0x2948(%rbx),%r8
  467fdd:	mov    %ebp,%edi
  467fdf:	movslq 0x294c(%rbx),%rax
  467fe6:	mov    %rdi,%r9
  467fe9:	mov    0x2930(%rbx),%rsi
  467ff0:	mov    0x2938(%rbx),%r10
  467ff7:	imul   %r8,%r9
  467ffb:	mov    0x2940(%rbx),%rdx
  468002:	mov    %r8,%rcx
  468005:	lea    0x0(,%rax,4),%r13
  46800d:	mov    %r9,%r12
  468010:	imul   %rax,%r12
  468014:	test   %eax,%eax
  468016:	jle    4680d8 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x508>
  46801c:	add    %r9,%r8
  46801f:	imul   %r13,%rdi
  468023:	add    %r12,%rax
  468026:	mov    %rsi,0x38(%rsp)
  46802b:	lea    (%r10,%r8,4),%rbp
  46802f:	mov    %r12,%r8
  468032:	mov    %rax,0x20(%rsp)
  468037:	lea    (%r10,%r9,4),%rax
  46803b:	neg    %r8
  46803e:	mov    %rax,0x30(%rsp)
  468043:	xor    %r14d,%r14d
  468046:	mov    %r12,%r15
  468049:	lea    (%rdi,%r8,4),%rdi
  46804d:	mov    %ecx,0x28(%rsp)
  468051:	lea    (%rdx,%rdi,1),%rax
  468055:	mov    %rbx,0x58(%rsp)
  46805a:	mov    %rax,(%rsp)
  46805e:	xchg   %ax,%ax
  468060:	mov    0x28(%rsp),%edi
  468064:	pxor   %xmm2,%xmm2
  468068:	test   %edi,%edi
  46806a:	jle    4680b2 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x4e2>
  46806c:	mov    0x38(%rsp),%rax
  468071:	mov    0x30(%rsp),%r12
  468076:	pxor   %xmm2,%xmm2
  46807a:	lea    (%rax,%r15,4),%rbx
  46807e:	xchg   %ax,%ax
  468080:	movss  (%rbx),%xmm0
  468084:	movss  (%r12),%xmm1
  46808a:	movd   %xmm0,%eax
  46808e:	not    %eax
  468090:	test   $0x7f800000,%eax
  468095:	sete   %al
  468098:	add    $0x4,%r12
  46809c:	add    %r13,%rbx
  46809f:	movzbl %al,%eax
  4680a2:	or     %eax,%r14d
  4680a5:	call   13480 <fmaf@plt>
  4680aa:	movaps %xmm0,%xmm2
  4680ad:	cmp    %rbp,%r12
  4680b0:	jne    468080 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x4b0>
  4680b2:	mov    (%rsp),%rax
  4680b6:	movss  %xmm2,(%rax,%r15,4)
  4680bc:	mov    0x20(%rsp),%rax
  4680c1:	add    $0x1,%r15
  4680c5:	cmp    %rax,%r15
  4680c8:	jne    468060 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x490>
  4680ca:	mov    0x58(%rsp),%rbx
  4680cf:	test   %r14d,%r14d
  4680d2:	jne    468a04 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xe34>
  4680d8:	cmpq   $0x0,0x80(%rbx)
  4680e0:	je     468170 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x5a0>
  4680e6:	pxor   %xmm0,%xmm0
  4680ea:	mov    $0x3,%edi
  4680ef:	mov    0x48(%rsp),%rbp
  4680f4:	lea    0xc0(%rsp),%rsi
  4680fc:	movaps %xmm0,0xc0(%rsp)
  468104:	add    0x68(%rbx),%rbp
  468108:	call   131a0 <clock_gettime@plt>
  46810d:	pxor   %xmm0,%xmm0
  468111:	mov    0x28d8(%rbx),%eax
  468117:	pxor   %xmm1,%xmm1
  46811b:	cvtsi2sdq 0xc0(%rsp),%xmm0
  468125:	mulsd  0x216f73(%rip),%xmm0        # 67f0a0 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0xa0>
  46812d:	cvtsi2sdq 0xc8(%rsp),%xmm1
  468137:	divsd  0x219f11(%rip),%xmm1        # 682050 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0x3050>
  46813f:	addsd  %xmm1,%xmm0
  468143:	subsd  0x10(%rsp),%xmm0
  468149:	cmp    $0x5,%eax
  46814c:	je     468738 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xb68>
  468152:	cmp    $0x6,%eax
  468155:	je     46876a <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xb9a>
  46815b:	sub    $0x7,%eax
  46815e:	and    $0xfffffffd,%eax
  468161:	jne    468170 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x5a0>
  468163:	addsd  0x8(%rbp),%xmm0
  468168:	movsd  %xmm0,0x8(%rbp)
  46816d:	nopl   (%rax)
  468170:	lock addl $0x1,0x2740(%rbx)
  468178:	jmp    467c38 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x68>
  46817d:	mov    0x2918(%rbx),%rdi
  468184:	mov    %rbp,%rdx
  468187:	mov    0x2920(%rbx),%rax
  46818e:	shl    $0xc,%rdx
  468192:	add    0x2928(%rbx),%rdx
  468199:	imul   0x10(%rdi),%rbp
  46819e:	lea    (%rax,%rbp,4),%rsi
  4681a2:	call   *0x48(%rbx)
  4681a5:	nopl   (%rax)
  4681a8:	cmpq   $0x0,0x80(%rbx)
  4681b0:	je     468170 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x5a0>
  4681b2:	cmpl   $0x8,0x28d8(%rbx)
  4681b9:	je     468170 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x5a0>
  4681bb:	jmp    4680e6 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x516>
  4681c0:	lea    0x0(,%rbp,8),%rax
  4681c8:	movq   $0x0,0x78(%rsp)
  4681d1:	sub    %rbp,%rax
  4681d4:	shl    $0x5,%rax
  4681d8:	add    0x2900(%rbx),%rax
  4681df:	mov    0xd0(%rax),%r12
  4681e6:	mov    %rax,%rcx
  4681e9:	mov    0x8(%rax),%ebp
  4681ec:	mov    %rax,0x28(%rsp)
  4681f1:	mov    0xd8(%rcx),%rsi
  4681f8:	lea    0x90(%rax),%rax
  4681ff:	test   %r12,%r12
  468202:	cmove  %rax,%r12
  468206:	mov    %rcx,%rax
  468209:	add    $0x50,%rax
  46820d:	test   %rsi,%rsi
  468210:	cmovne %rsi,%rax
  468214:	mov    0x1c(%rsp),%esi
  468218:	mov    %rax,0x38(%rsp)
  46821d:	test   %esi,%esi
  46821f:	js     468e40 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1270>
  468225:	mov    0x2918(%rbx),%rdi
  46822c:	cmpl   $0x2a,(%rdi)
  46822f:	je     468e5a <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x128a>
  468235:	test   %ebp,%ebp
  468237:	jle    4682e5 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x715>
  46823d:	mov    %fs:0x0,%r11
  468246:	add    $0xffffffffffffb000,%r11
  46824d:	mov    %r11,0x110(%rsp)
  468255:	cmp    $0x1,%ebp
  468258:	je     4682e5 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x715>
  46825e:	lea    0xa00(%r11),%rax
  468265:	mov    %rax,0x118(%rsp)
  46826d:	cmp    $0x2,%ebp
  468270:	je     4682e5 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x715>
  468272:	lea    0x1400(%r11),%rax
  468279:	mov    %rax,0x120(%rsp)
  468281:	cmp    $0x3,%ebp
  468284:	je     4682e5 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x715>
  468286:	lea    0x1e00(%r11),%rax
  46828d:	mov    %rax,0x128(%rsp)
  468295:	cmp    $0x4,%ebp
  468298:	je     4682e5 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x715>
  46829a:	lea    0x2800(%r11),%rax
  4682a1:	mov    %rax,0x130(%rsp)
  4682a9:	cmp    $0x5,%ebp
  4682ac:	je     4682e5 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x715>
  4682ae:	lea    0x3200(%r11),%rax
  4682b5:	mov    %rax,0x138(%rsp)
  4682bd:	cmp    $0x6,%ebp
  4682c0:	je     4682e5 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x715>
  4682c2:	lea    0x3c00(%r11),%rax
  4682c9:	mov    %rax,0x140(%rsp)
  4682d1:	cmp    $0x7,%ebp
  4682d4:	je     4682e5 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x715>
  4682d6:	add    $0x4600,%r11
  4682dd:	mov    %r11,0x148(%rsp)
  4682e5:	mov    0x28(%rsp),%rax
  4682ea:	sub    $0x8,%rsp
  4682ee:	xor    %r9d,%r9d
  4682f1:	mov    %ebp,%ecx
  4682f3:	mov    %r12,%rdx
  4682f6:	mov    (%rax),%rsi
  4682f9:	mov    0x18(%rdi),%rax
  4682fd:	push   %rax
  4682fe:	lea    0x120(%rsp),%r8
  468306:	call   *0x58(%rbx)
  468309:	pop    %r10
  46830b:	pop    %r11
  46830d:	movq   $0x0,0x58(%rsp)
  468316:	mov    0x1c(%rsp),%r9d
  46831b:	test   %r9d,%r9d
  46831e:	js     469250 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1680>
  468324:	cmpq   $0x0,0x80(%rbx)
  46832c:	movq   $0x0,0x30(%rsp)
  468335:	je     46838b <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x7bb>
  468337:	pxor   %xmm0,%xmm0
  46833b:	lea    0xc0(%rsp),%rsi
  468343:	mov    $0x3,%edi
  468348:	movaps %xmm0,0xc0(%rsp)
  468350:	call   131a0 <clock_gettime@plt>
  468355:	pxor   %xmm1,%xmm1
  468359:	pxor   %xmm0,%xmm0
  46835d:	cvtsi2sdq 0xc0(%rsp),%xmm1
  468367:	mulsd  0x216d31(%rip),%xmm1        # 67f0a0 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0xa0>
  46836f:	cvtsi2sdq 0xc8(%rsp),%xmm0
  468379:	divsd  0x219ccf(%rip),%xmm0        # 682050 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0x3050>
  468381:	addsd  %xmm0,%xmm1
  468385:	movsd  %xmm1,0x30(%rsp)
  46838b:	mov    0x2918(%rbx),%rax
  468392:	cmpl   $0x2a,0x4(%rax)
  468396:	je     46925f <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x168f>
  46839c:	test   %ebp,%ebp
  46839e:	jle    4693e3 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1813>
  4683a4:	lea    0x110(%rsp),%rax
  4683ac:	movslq %ebp,%rdx
  4683af:	mov    %fs:0x0,%r11
  4683b8:	mov    %rax,0x20(%rsp)
  4683bd:	mov    %rax,%r12
  4683c0:	lea    (%rax,%rdx,8),%r15
  4683c4:	lea    -0x5000(%r11),%r14
  4683cb:	lea    -0xe000(%r11),%r13
  4683d2:	nopw   0x0(%rax,%rax,1)
  4683d8:	mov    %r13,%rdx
  4683db:	mov    %r14,%rsi
  4683de:	mov    0x2918(%rbx),%rdi
  4683e5:	add    $0x8,%r12
  4683e9:	call   *0x50(%rbx)
  4683ec:	add    $0xa00,%r14
  4683f3:	mov    %r13,-0x8(%r12)
  4683f8:	add    $0x400,%r13
  4683ff:	cmp    %r12,%r15
  468402:	jne    4683d8 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x808>
  468404:	mov    0x1c(%rsp),%ecx
  468408:	xor    %r12d,%r12d
  46840b:	test   %ecx,%ecx
  46840d:	js     4693bd <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x17ed>
  468413:	cmpq   $0x0,0x80(%rbx)
  46841b:	movq   $0x0,(%rsp)
  468423:	je     468478 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x8a8>
  468425:	pxor   %xmm0,%xmm0
  468429:	lea    0xc0(%rsp),%rsi
  468431:	mov    $0x3,%edi
  468436:	movaps %xmm0,0xc0(%rsp)
  46843e:	call   131a0 <clock_gettime@plt>
  468443:	pxor   %xmm1,%xmm1
  468447:	pxor   %xmm0,%xmm0
  46844b:	cvtsi2sdq 0xc0(%rsp),%xmm1
  468455:	mulsd  0x216c43(%rip),%xmm1        # 67f0a0 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0xa0>
  46845d:	cvtsi2sdq 0xc8(%rsp),%xmm0
  468467:	divsd  0x219be1(%rip),%xmm0        # 682050 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0x3050>
  46846f:	addsd  %xmm0,%xmm1
  468473:	movsd  %xmm1,(%rsp)
  468478:	mov    0x28(%rsp),%rax
  46847d:	sub    $0x8,%rsp
  468481:	xor    %r9d,%r9d
  468484:	mov    %ebp,%ecx
  468486:	mov    0x2918(%rbx),%rdi
  46848d:	mov    (%rax),%rsi
  468490:	mov    0x10(%rdi),%rax
  468494:	push   %rax
  468495:	mov    0x48(%rsp),%r8
  46849a:	mov    0x30(%rsp),%rdx
  46849f:	call   *0x60(%rbx)
  4684a2:	pop    %rax
  4684a3:	pop    %rdx
  4684a4:	mov    0x1c(%rsp),%r15d
  4684a9:	test   %r15d,%r15d
  4684ac:	js     4691df <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x160f>
  4684b2:	cmpq   $0x0,0x80(%rbx)
  4684ba:	je     468170 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x5a0>
  4684c0:	movsd  0x30(%rsp),%xmm4
  4684c6:	movsd  (%rsp),%xmm1
  4684cb:	lea    0xc0(%rsp),%rsi
  4684d3:	mov    $0x3,%edi
  4684d8:	movslq 0x74(%rsp),%rbp
  4684dd:	subsd  %xmm4,%xmm1
  4684e1:	movapd %xmm4,%xmm0
  4684e5:	subsd  0x10(%rsp),%xmm0
  4684eb:	shl    $0x6,%rbp
  4684ef:	add    0x68(%rbx),%rbp
  4684f3:	unpcklpd %xmm1,%xmm0
  4684f7:	addpd  0x0(%rbp),%xmm0
  4684fc:	movaps %xmm0,0x0(%rbp)
  468500:	pxor   %xmm0,%xmm0
  468504:	movaps %xmm0,0xc0(%rsp)
  46850c:	call   131a0 <clock_gettime@plt>
  468511:	pxor   %xmm0,%xmm0
  468515:	pxor   %xmm1,%xmm1
  468519:	cvtsi2sdq 0xc0(%rsp),%xmm0
  468523:	mulsd  0x216b75(%rip),%xmm0        # 67f0a0 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0xa0>
  46852b:	cvtsi2sdq 0xc8(%rsp),%xmm1
  468535:	divsd  0x219b13(%rip),%xmm1        # 682050 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0x3050>
  46853d:	addsd  %xmm1,%xmm0
  468541:	subsd  (%rsp),%xmm0
  468546:	addsd  0x10(%rbp),%xmm0
  46854b:	movsd  %xmm0,0x10(%rbp)
  468550:	jmp    4681a8 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x5d8>
  468555:	lea    0x0(,%rbp,8),%rdx
  46855d:	mov    0x2900(%rbx),%rax
  468564:	sub    %rbp,%rdx
  468567:	mov    %rdx,%r12
  46856a:	shl    $0x5,%rdx
  46856e:	mov    0x8(%rax,%rdx,1),%r14d
  468573:	shl    $0xd,%r12
  468577:	add    0x2958(%rbx),%r12
  46857e:	test   %r14d,%r14d
  468581:	jle    4680d8 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x508>
  468587:	lea    0x5000(%r12),%rbp
  46858f:	lea    0xc000(%r12),%r14
  468597:	xor    %r13d,%r13d
  46859a:	mov    %rdx,%r15
  46859d:	jmp    4685d4 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xa04>
  46859f:	nop
  4685a0:	mov    %r14,%rdx
  4685a3:	mov    %r12,%rsi
  4685a6:	call   *0x50(%rbx)
  4685a9:	mov    0x2900(%rbx),%rax
  4685b0:	add    $0x1,%r13d
  4685b4:	add    $0xa00,%r12
  4685bb:	add    $0xe00,%rbp
  4685c2:	add    $0x400,%r14
  4685c9:	cmp    %r13d,0x8(%rax,%r15,1)
  4685ce:	jle    4681a8 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x5d8>
  4685d4:	mov    0x2918(%rbx),%rdi
  4685db:	cmpl   $0x2a,0x4(%rdi)
  4685df:	jne    4685a0 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x9d0>
  4685e1:	call   46f3d0 <_ZN5guild7kernels3cpu11cpu_avx2_okEv>
  4685e6:	test   %al,%al
  4685e8:	jne    468750 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xb80>
  4685ee:	mov    0x2918(%rbx),%rdi
  4685f5:	jmp    4685a0 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x9d0>
  4685f7:	mov    0x28dc(%rbx),%esi
  4685fd:	mov    %ebp,%eax
  4685ff:	cltd
  468600:	idiv   %esi
  468602:	lea    (%rdx,%rdx,4),%ecx
  468605:	movslq %eax,%r9
  468608:	shl    $0x7,%ecx
  46860b:	mov    %r9,%rdi
  46860e:	lea    0x280(%rcx),%eax
  468614:	shl    $0x5,%rdi
  468618:	add    0x90(%rbx),%rdi
  46861f:	cltd
  468620:	idiv   %esi
  468622:	mov    %eax,%r8d
  468625:	mov    %ecx,%eax
  468627:	cltd
  468628:	idiv   %esi
  46862a:	mov    0x8(%rdi),%rsi
  46862e:	mov    (%rdi),%rdi
  468631:	lea    (%r9,%r9,2),%rdx
  468635:	shl    $0xb,%rdx
  468639:	add    0x28e8(%rbx),%rdx
  468640:	mov    %eax,%ecx
  468642:	call   45aca0 <_ZN5guild7kernels3cpu17s2_expert_gu_rowsEPKhRKNS1_4ActQEPfii>
  468647:	jmp    4681a8 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x5d8>
  46864c:	mov    0x28e0(%rbx),%r9d
  468653:	mov    %ebp,%eax
  468655:	cltd
  468656:	idiv   %r9d
  468659:	lea    (%rdx,%rdx,4),%ecx
  46865c:	movslq %eax,%rsi
  46865f:	shl    $0x9,%ecx
  468662:	mov    %rsi,%rdi
  468665:	lea    (%rsi,%rsi,2),%rsi
  468669:	lea    0xa00(%rcx),%eax
  46866f:	shl    $0x5,%rdi
  468673:	shl    $0xb,%rsi
  468677:	add    0x90(%rbx),%rdi
  46867e:	cltd
  46867f:	add    0x28e8(%rbx),%rsi
  468686:	idiv   %r9d
  468689:	add    $0xa00,%rsi
  468690:	mov    %eax,%r8d
  468693:	mov    %ecx,%eax
  468695:	cltd
  468696:	idiv   %r9d
  468699:	mov    0x10(%rdi),%rdx
  46869d:	mov    (%rdi),%rdi
  4686a0:	mov    %eax,%ecx
  4686a2:	call   45b6e0 <_ZN5guild7kernels3cpu19s2_expert_down_rowsEPKhRKNS1_4ActQEPfii>
  4686a7:	jmp    4681a8 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x5d8>
  4686ac:	shl    $0x5,%rbp
  4686b0:	add    0x90(%rbx),%rbp
  4686b7:	mov    0x50(%rsp),%rcx
  4686bc:	mov    0x10(%rbp),%rdx
  4686c0:	mov    0x8(%rbp),%rsi
  4686c4:	mov    0x0(%rbp),%rdi
  4686c8:	call   459540 <_ZN5guild7kernels3cpu16s2_expert_vnni_qEPKhRKNS1_4ActQEPfRNS1_13ExpertScratchE>
  4686cd:	jmp    4681a8 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x5d8>
  4686d2:	nopw   0x0(%rax,%rax,1)
  4686d8:	mov    %ebp,0x26e0(%rbx)
  4686de:	call   13650 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  4686e3:	mov    %rax,%rcx
  4686e6:	movabs $0x431bde82d7b634db,%rax
  4686f0:	imul   %rcx
  4686f3:	sar    $0x3f,%rcx
  4686f7:	sar    $0x12,%rdx
  4686fb:	sub    %rcx,%rdx
  4686fe:	mov    %rdx,0x26e8(%rbx)
  468705:	jmp    467c8e <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xbe>
  46870a:	mov    0x158(%rsp),%rax
  468712:	sub    %fs:0x28,%rax
  46871b:	jne    4693f5 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1825>
  468721:	add    $0x168,%rsp
  468728:	pop    %rbx
  468729:	pop    %rbp
  46872a:	pop    %r12
  46872c:	pop    %r13
  46872e:	pop    %r14
  468730:	pop    %r15
  468732:	ret
  468733:	nopl   0x0(%rax,%rax,1)
  468738:	addsd  0x0(%rbp),%xmm0
  46873d:	movsd  %xmm0,0x0(%rbp)
  468742:	jmp    468170 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x5a0>
  468747:	nopw   0x0(%rax,%rax,1)
  468750:	mov    0x2918(%rbx),%rax
  468757:	mov    %rbp,%rdx
  46875a:	mov    %r12,%rdi
  46875d:	mov    0x18(%rax),%esi
  468760:	call   46f590 <_ZN5guild7kernels3cpu13act_quant_anyEPKfiRNS1_4ActQE>
  468765:	jmp    4685a9 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x9d9>
  46876a:	addsd  0x10(%rbp),%xmm0
  46876f:	movsd  %xmm0,0x10(%rbp)
  468774:	jmp    468170 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x5a0>
  468779:	cmp    $0x3,%r13d
  46877d:	mov    0x2908(%rbx),%rsi
  468784:	mov    $0x280,%eax
  468789:	mov    $0xa00,%r14d
  46878f:	cmove  %rax,%r14
  468793:	mov    %ebp,%eax
  468795:	movslq 0x2910(%rbx),%rcx
  46879c:	imul   %rsi,%rax
  4687a0:	cqto
  4687a2:	idiv   %rcx
  4687a5:	mov    %rax,%r12
  4687a8:	lea    0x1(%rbp),%eax
  4687ab:	imul   %rsi,%rax
  4687af:	cqto
  4687b1:	idiv   %rcx
  4687b4:	mov    %rax,%rbp
  4687b7:	cmp    %rax,%r12
  4687ba:	jge    4680d8 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x508>
  4687c0:	lea    0x110(%rsp),%rax
  4687c8:	mov    %r13d,%ecx
  4687cb:	mov    %rbx,%r13
  4687ce:	mov    %rax,0x20(%rsp)
  4687d3:	jmp    4688b9 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xce9>
  4687d8:	nopl   0x0(%rax,%rax,1)
  4687e0:	test   %esi,%esi
  4687e2:	jle    468887 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xcb7>
  4687e8:	lea    0x5000(%rdx),%rcx
  4687ef:	mov    %rcx,0x110(%rsp)
  4687f7:	cmp    $0x1,%esi
  4687fa:	je     468887 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xcb7>
  468800:	lea    0x5e00(%rdx),%rcx
  468807:	mov    %rcx,0x118(%rsp)
  46880f:	cmp    $0x2,%esi
  468812:	je     468887 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xcb7>
  468814:	lea    0x6c00(%rdx),%rcx
  46881b:	mov    %rcx,0x120(%rsp)
  468823:	cmp    $0x3,%esi
  468826:	je     468887 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xcb7>
  468828:	lea    0x7a00(%rdx),%rcx
  46882f:	mov    %rcx,0x128(%rsp)
  468837:	cmp    $0x4,%esi
  46883a:	je     468887 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xcb7>
  46883c:	lea    0x8800(%rdx),%rcx
  468843:	mov    %rcx,0x130(%rsp)
  46884b:	cmp    $0x5,%esi
  46884e:	je     468887 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xcb7>
  468850:	lea    0x9600(%rdx),%rcx
  468857:	mov    %rcx,0x138(%rsp)
  46885f:	cmp    $0x6,%esi
  468862:	je     468887 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xcb7>
  468864:	lea    0xa400(%rdx),%rcx
  46886b:	mov    %rcx,0x140(%rsp)
  468873:	cmp    $0x7,%esi
  468876:	je     468887 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xcb7>
  468878:	add    $0xb200,%rdx
  46887f:	mov    %rdx,0x148(%rsp)
  468887:	mov    %r15d,%r9d
  46888a:	sub    %ebx,%r15d
  46888d:	mov    %esi,%edx
  46888f:	mov    (%rax),%rdi
  468892:	mov    0x20(%rsp),%rsi
  468897:	movslq %r15d,%r15
  46889a:	lea    0x50(%rax),%rcx
  46889e:	mov    %ebx,%r8d
  4688a1:	add    %r15,%r12
  4688a4:	call   45d130 <_ZN5guild7kernels3cpu25s2_expert_down_rows_multiEPKhPKPKNS1_4ActQEiPKPfii>
  4688a9:	cmp    %r12,%rbp
  4688ac:	jle    4689d5 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xe05>
  4688b2:	mov    0x28d8(%r13),%ecx
  4688b9:	mov    %r12,%rax
  4688bc:	mov    %rbp,%r15
  4688bf:	cqto
  4688c1:	sub    %r12,%r15
  4688c4:	idiv   %r14
  4688c7:	add    %rdx,%r15
  4688ca:	mov    %rdx,%rbx
  4688cd:	movslq %eax,%rdx
  4688d0:	cmp    %r14,%r15
  4688d3:	lea    0x0(,%rdx,8),%rax
  4688db:	cmovg  %r14,%r15
  4688df:	sub    %rdx,%rax
  4688e2:	mov    %rax,%rdx
  4688e5:	shl    $0x5,%rax
  4688e9:	add    0x2900(%r13),%rax
  4688f0:	shl    $0xd,%rdx
  4688f4:	mov    0x8(%rax),%esi
  4688f7:	add    0x2958(%r13),%rdx
  4688fe:	cmp    $0x3,%ecx
  468901:	jne    4687e0 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xc10>
  468907:	test   %esi,%esi
  468909:	jle    4689a7 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xdd7>
  46890f:	mov    %rdx,0x110(%rsp)
  468917:	cmp    $0x1,%esi
  46891a:	je     4689a7 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xdd7>
  468920:	lea    0xa00(%rdx),%rcx
  468927:	mov    %rcx,0x118(%rsp)
  46892f:	cmp    $0x2,%esi
  468932:	je     4689a7 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xdd7>
  468934:	lea    0x1400(%rdx),%rcx
  46893b:	mov    %rcx,0x120(%rsp)
  468943:	cmp    $0x3,%esi
  468946:	je     4689a7 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xdd7>
  468948:	lea    0x1e00(%rdx),%rcx
  46894f:	mov    %rcx,0x128(%rsp)
  468957:	cmp    $0x4,%esi
  46895a:	je     4689a7 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xdd7>
  46895c:	lea    0x2800(%rdx),%rcx
  468963:	mov    %rcx,0x130(%rsp)
  46896b:	cmp    $0x5,%esi
  46896e:	je     4689a7 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xdd7>
  468970:	lea    0x3200(%rdx),%rcx
  468977:	mov    %rcx,0x138(%rsp)
  46897f:	cmp    $0x6,%esi
  468982:	je     4689a7 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xdd7>
  468984:	lea    0x3c00(%rdx),%rcx
  46898b:	mov    %rcx,0x140(%rsp)
  468993:	cmp    $0x7,%esi
  468996:	je     4689a7 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xdd7>
  468998:	add    $0x4600,%rdx
  46899f:	mov    %rdx,0x148(%rsp)
  4689a7:	mov    %r15d,%r9d
  4689aa:	sub    %ebx,%r15d
  4689ad:	lea    0x10(%rax),%r11
  4689b1:	mov    (%rax),%rdi
  4689b4:	mov    0x20(%rsp),%rcx
  4689b9:	movslq %r15d,%r15
  4689bc:	mov    %esi,%edx
  4689be:	mov    %ebx,%r8d
  4689c1:	mov    %r11,%rsi
  4689c4:	add    %r15,%r12
  4689c7:	call   45bda0 <_ZN5guild7kernels3cpu23s2_expert_gu_rows_multiEPKhPKPKNS1_4ActQEiPKPfii>
  4689cc:	cmp    %r12,%rbp
  4689cf:	jg     4688b2 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xce2>
  4689d5:	mov    %r13,%rbx
  4689d8:	jmp    4681a8 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x5d8>
  4689dd:	mov    %r11,0x30(%rsp)
  4689e2:	mov    %r10,(%rsp)
  4689e6:	call   46f3d0 <_ZN5guild7kernels3cpu11cpu_avx2_okEv>
  4689eb:	mov    (%rsp),%r10
  4689ef:	mov    0x30(%rsp),%r11
  4689f4:	test   %al,%al
  4689f6:	jne    468a10 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xe40>
  4689f8:	mov    0x2918(%rbx),%rdi
  4689ff:	jmp    467d9a <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1ca>
  468a04:	movb   $0x1,0x2950(%rbx)
  468a0b:	jmp    4681a8 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x5d8>
  468a10:	add    0x2900(%rbx),%r15
  468a17:	mov    0x8(%r15),%r8d
  468a1b:	test   %r8d,%r8d
  468a1e:	jle    468ad1 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xf01>
  468a24:	lea    0x5000(%r12),%rax
  468a2c:	mov    %rax,0x110(%rsp)
  468a34:	cmp    $0x1,%r8d
  468a38:	je     468ad1 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xf01>
  468a3e:	lea    0x5e00(%r12),%rax
  468a46:	mov    %rax,0x118(%rsp)
  468a4e:	cmp    $0x2,%r8d
  468a52:	je     468ad1 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xf01>
  468a54:	lea    0x6c00(%r12),%rax
  468a5c:	mov    %rax,0x120(%rsp)
  468a64:	cmp    $0x3,%r8d
  468a68:	je     468ad1 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xf01>
  468a6a:	lea    0x7a00(%r12),%rax
  468a72:	mov    %rax,0x128(%rsp)
  468a7a:	cmp    $0x4,%r8d
  468a7e:	je     468ad1 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xf01>
  468a80:	lea    0x8800(%r12),%rax
  468a88:	mov    %rax,0x130(%rsp)
  468a90:	cmp    $0x5,%r8d
  468a94:	je     468ad1 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xf01>
  468a96:	lea    0x9600(%r12),%rax
  468a9e:	mov    %rax,0x138(%rsp)
  468aa6:	cmp    $0x6,%r8d
  468aaa:	je     468ad1 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xf01>
  468aac:	lea    0xa400(%r12),%rax
  468ab4:	mov    %rax,0x140(%rsp)
  468abc:	cmp    $0x7,%r8d
  468ac0:	je     468ad1 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xf01>
  468ac2:	add    $0xb200,%r12
  468ac9:	mov    %r12,0x148(%rsp)
  468ad1:	mov    0x2918(%rbx),%rax
  468ad8:	mov    %r11,0x30(%rsp)
  468add:	lea    0x50(%r15),%r9
  468ae1:	mov    0x18(%rax),%rcx
  468ae5:	mov    0x28(%rax),%rsi
  468ae9:	mov    0x38(%rax),%rdi
  468aed:	test   %rcx,%rcx
  468af0:	lea    0x3f(%rcx),%rdx
  468af4:	cmovns %rcx,%rdx
  468af8:	add    (%r15),%rdi
  468afb:	push   %rbp
  468afc:	push   %r10
  468afe:	mov    0x30(%rsp),%rcx
  468b03:	sar    $0x6,%rdx
  468b07:	mov    %r10,0x10(%rsp)
  468b0c:	call   46f6a0 <_ZN5guild7kernels3cpu11q2_rows_anyEPKhmiPKPKNS1_4ActQEiPKPfii>
  468b11:	pop    %rcx
  468b12:	pop    %rsi
  468b13:	mov    (%rsp),%r10
  468b17:	mov    0x30(%rsp),%r11
  468b1c:	jmp    467e7f <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x2af>
  468b21:	mov    %r11,0x30(%rsp)
  468b26:	mov    %rdx,(%rsp)
  468b2a:	call   46f3d0 <_ZN5guild7kernels3cpu11cpu_avx2_okEv>
  468b2f:	mov    (%rsp),%r10
  468b33:	mov    0x30(%rsp),%r11
  468b38:	test   %al,%al
  468b3a:	je     468e27 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1257>
  468b40:	mov    0x2900(%rbx),%rdi
  468b47:	add    %r15,%rdi
  468b4a:	mov    0x8(%rdi),%r8d
  468b4e:	test   %r8d,%r8d
  468b51:	jle    468c97 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x10c7>
  468b57:	mov    %fs:0x0,%rax
  468b60:	lea    -0x1d000(%rax),%rcx
  468b67:	lea    -0x22000(%rax),%rdx
  468b6e:	mov    %rcx,0xd0(%rsp)
  468b76:	mov    %rdx,0x110(%rsp)
  468b7e:	cmp    $0x1,%r8d
  468b82:	je     468c97 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x10c7>
  468b88:	lea    -0x1c600(%rax),%rsi
  468b8f:	mov    %rsi,0xd8(%rsp)
  468b97:	lea    -0x21600(%rax),%rsi
  468b9e:	mov    %rsi,0x118(%rsp)
  468ba6:	cmp    $0x2,%r8d
  468baa:	je     468c97 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x10c7>
  468bb0:	lea    -0x1bc00(%rax),%rsi
  468bb7:	mov    %rsi,0xe0(%rsp)
  468bbf:	lea    -0x20c00(%rax),%rsi
  468bc6:	mov    %rsi,0x120(%rsp)
  468bce:	cmp    $0x3,%r8d
  468bd2:	je     468c97 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x10c7>
  468bd8:	lea    -0x1b200(%rax),%rsi
  468bdf:	mov    %rsi,0xe8(%rsp)
  468be7:	lea    -0x20200(%rax),%rsi
  468bee:	mov    %rsi,0x128(%rsp)
  468bf6:	cmp    $0x4,%r8d
  468bfa:	je     468c97 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x10c7>
  468c00:	lea    -0x1a800(%rax),%rsi
  468c07:	mov    %rsi,0xf0(%rsp)
  468c0f:	lea    -0x1f800(%rax),%rsi
  468c16:	mov    %rsi,0x130(%rsp)
  468c1e:	cmp    $0x5,%r8d
  468c22:	je     468c97 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x10c7>
  468c24:	lea    -0x19e00(%rax),%rcx
  468c2b:	lea    -0x1ee00(%rax),%rdx
  468c32:	mov    %rcx,0xf8(%rsp)
  468c3a:	mov    %rdx,0x138(%rsp)
  468c42:	cmp    $0x6,%r8d
  468c46:	je     468c97 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x10c7>
  468c48:	lea    -0x19400(%rax),%rdx
  468c4f:	lea    -0x1d000(%rax),%rcx
  468c56:	mov    %rdx,0x100(%rsp)
  468c5e:	lea    -0x22000(%rax),%rdx
  468c65:	add    $0xfffffffffffe1c00,%rax
  468c6b:	mov    %rax,0x140(%rsp)
  468c73:	cmp    $0x7,%r8d
  468c77:	je     468c97 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x10c7>
  468c79:	lea    0x4600(%rcx),%rax
  468c80:	mov    %rax,0x108(%rsp)
  468c88:	lea    0x4600(%rdx),%rax
  468c8f:	mov    %rax,0x148(%rsp)
  468c97:	mov    0x2918(%rbx),%rax
  468c9e:	mov    %r11,0x30(%rsp)
  468ca3:	lea    0x10(%rdi),%rcx
  468ca7:	mov    (%rdi),%rdi
  468caa:	mov    0x10(%rax),%rdx
  468cae:	mov    0x20(%rax),%rsi
  468cb2:	push   %rbp
  468cb3:	push   %r10
  468cb5:	test   %rdx,%rdx
  468cb8:	lea    0x3f(%rdx),%r13
  468cbc:	cmovns %rdx,%r13
  468cc0:	lea    0xe0(%rsp),%r9
  468cc8:	mov    %r10,0x10(%rsp)
  468ccd:	sar    $0x6,%r13
  468cd1:	mov    %r13d,%edx
  468cd4:	call   46f6a0 <_ZN5guild7kernels3cpu11q2_rows_anyEPKhmiPKPKNS1_4ActQEiPKPfii>
  468cd9:	mov    0x2900(%rbx),%rax
  468ce0:	mov    0x2918(%rbx),%rdx
  468ce7:	pop    %r9
  468ce9:	pop    %r10
  468ceb:	add    %r15,%rax
  468cee:	mov    0x30(%rdx),%rdi
  468cf2:	mov    0x20(%rdx),%rsi
  468cf6:	mov    %r13d,%edx
  468cf9:	add    (%rax),%rdi
  468cfc:	mov    0x8(%rax),%r8d
  468d00:	push   %rbp
  468d01:	lea    0x10(%rax),%rcx
  468d05:	mov    0x8(%rsp),%r10
  468d0a:	push   %r10
  468d0c:	mov    0x30(%rsp),%r9
  468d11:	mov    %r10,0x10(%rsp)
  468d16:	call   46f6a0 <_ZN5guild7kernels3cpu11q2_rows_anyEPKhmiPKPKNS1_4ActQEiPKPfii>
  468d1b:	mov    0x2900(%rbx),%rax
  468d22:	pop    %r11
  468d24:	pop    %r13
  468d26:	mov    (%rsp),%r10
  468d2a:	movslq 0x8(%rax,%r15,1),%rax
  468d2f:	mov    0x30(%rsp),%r11
  468d34:	test   %eax,%eax
  468d36:	jle    467e7f <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x2af>
  468d3c:	cmp    %ebp,%r10d
  468d3f:	jge    467e7f <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x2af>
  468d45:	lea    (%rax,%rax,4),%rax
  468d49:	mov    %ebp,%ecx
  468d4b:	mov    %r14,0x58(%rsp)
  468d50:	lea    0x0(,%r10,4),%rdx
  468d58:	shl    $0x7,%rax
  468d5c:	sub    %r10d,%ecx
  468d5f:	mov    %r11,0x60(%rsp)
  468d64:	add    %r10,%rax
  468d67:	sub    $0x1,%ecx
  468d6a:	mov    %r10,0x78(%rsp)
  468d6f:	shl    $0x2,%rax
  468d73:	lea    0x1(%r10,%rcx,1),%r13
  468d78:	mov    %rbp,0x38(%rsp)
  468d7d:	mov    %rax,0x30(%rsp)
  468d82:	shl    $0x2,%r13
  468d86:	mov    %fs:0x0,%rax
  468d8f:	mov    %rbx,0x68(%rsp)
  468d94:	mov    %rdx,%rbx
  468d97:	lea    -0x22000(%rax),%r15
  468d9e:	mov    %rax,%rbp
  468da1:	nopl   0x0(%rax)
  468da8:	mov    %rbx,%r14
  468dab:	nopl   0x0(%rax,%rax,1)
  468db0:	movss  -0x1d000(%r14,%rbp,1),%xmm1
  468dba:	movaps %xmm1,%xmm0
  468dbd:	xorps  0x218adc(%rip),%xmm0        # 6818a0 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0x28a0>
  468dc4:	movss  %xmm1,(%rsp)
  468dc9:	call   13b60 <expf@plt>
  468dce:	addss  0x1d022e(%rip),%xmm0        # 639004 <_IO_stdin_used+0x4>
  468dd6:	movss  (%rsp),%xmm1
  468ddb:	divss  %xmm0,%xmm1
  468ddf:	mulss  (%r14,%r15,1),%xmm1
  468de5:	movss  %xmm1,(%r12,%r14,1)
  468deb:	add    $0x4,%r14
  468def:	cmp    %r13,%r14
  468df2:	jne    468db0 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x11e0>
  468df4:	add    $0xa00,%rbx
  468dfb:	add    $0xa00,%r13
  468e02:	cmp    %rbx,0x30(%rsp)
  468e07:	jne    468da8 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x11d8>
  468e09:	mov    0x38(%rsp),%rbp
  468e0e:	mov    0x58(%rsp),%r14
  468e13:	mov    0x60(%rsp),%r11
  468e18:	mov    0x68(%rsp),%rbx
  468e1d:	mov    0x78(%rsp),%r10
  468e22:	jmp    467e7f <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x2af>
  468e27:	cmpl   $0x5,0x28d8(%rbx)
  468e2e:	mov    0x2918(%rbx),%rdi
  468e35:	jne    467d90 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1c0>
  468e3b:	jmp    467ef0 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x320>
  468e40:	call   13650 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  468e45:	mov    0x2918(%rbx),%rdi
  468e4c:	mov    %rax,0x78(%rsp)
  468e51:	cmpl   $0x2a,(%rdi)
  468e54:	jne    468235 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x665>
  468e5a:	call   46f3d0 <_ZN5guild7kernels3cpu11cpu_avx2_okEv>
  468e5f:	test   %al,%al
  468e61:	je     4693ca <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x17fa>
  468e67:	mov    0x2918(%rbx),%rsi
  468e6e:	mov    0x10(%rsi),%rdx
  468e72:	test   %rdx,%rdx
  468e75:	lea    0x3f(%rdx),%rax
  468e79:	cmovns %rdx,%rax
  468e7d:	sar    $0x6,%rax
  468e81:	mov    %eax,0xa0(%rsp)
  468e88:	test   %ebp,%ebp
  468e8a:	jle    46830d <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x73d>
  468e90:	mov    0x28(%rsp),%rcx
  468e95:	lea    -0x8(%rbp),%edi
  468e98:	mov    0x18(%rsi),%rdx
  468e9c:	mov    %ebp,%eax
  468e9e:	mov    %fs:0x0,%r11
  468ea7:	mov    %ebp,0xb8(%rsp)
  468eae:	add    $0x10,%rcx
  468eb2:	mov    %r13d,0xbc(%rsp)
  468eba:	mov    %rcx,0x30(%rsp)
  468ebf:	lea    -0x5000(%r11),%rcx
  468ec6:	lea    -0x13000(%r11),%r10
  468ecd:	mov    %rcx,0x60(%rsp)
  468ed2:	lea    -0x1(%rbp),%ecx
  468ed5:	and    $0xfffffff8,%ecx
  468ed8:	mov    %r11,0x68(%rsp)
  468edd:	sub    %ecx,%edi
  468edf:	lea    0xd0(%rsp),%rcx
  468ee7:	mov    %rcx,0xa8(%rsp)
  468eef:	lea    0x110(%rsp),%rcx
  468ef7:	mov    %rcx,0x20(%rsp)
  468efc:	lea    -0x12600(%r11),%rcx
  468f03:	mov    %edi,0xa4(%rsp)
  468f0a:	mov    %rcx,0xb0(%rsp)
  468f12:	nopw   0x0(%rax,%rax,1)
  468f18:	mov    $0x8,%ebp
  468f1d:	mov    0x68(%rsp),%rcx
  468f22:	mov    %r10,0xd0(%rsp)
  468f2a:	cmp    %ebp,%eax
  468f2c:	cmovle %eax,%ebp
  468f2f:	add    $0xfffffffffffe8000,%rcx
  468f36:	mov    %rcx,0x110(%rsp)
  468f3e:	cmp    $0x1,%eax
  468f41:	jle    469044 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1474>
  468f47:	mov    0xb0(%rsp),%rdi
  468f4f:	mov    %rdi,0xd8(%rsp)
  468f57:	lea    0xa00(%rcx),%rdi
  468f5e:	mov    %rdi,0x118(%rsp)
  468f66:	cmp    $0x2,%eax
  468f69:	jle    469044 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1474>
  468f6f:	lea    0x1400(%r10),%rdi
  468f76:	mov    %rdi,0xe0(%rsp)
  468f7e:	lea    0x1400(%rcx),%rdi
  468f85:	mov    %rdi,0x120(%rsp)
  468f8d:	cmp    $0x3,%eax
  468f90:	jle    469044 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1474>
  468f96:	lea    0x1e00(%r10),%rdi
  468f9d:	mov    %rdi,0xe8(%rsp)
  468fa5:	lea    0x1e00(%rcx),%rdi
  468fac:	mov    %rdi,0x128(%rsp)
  468fb4:	cmp    $0x4,%eax
  468fb7:	jle    469044 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1474>
  468fbd:	lea    0x2800(%r10),%rdi
  468fc4:	mov    %rdi,0xf0(%rsp)
  468fcc:	lea    0x2800(%rcx),%rdi
  468fd3:	mov    %rdi,0x130(%rsp)
  468fdb:	cmp    $0x5,%eax
  468fde:	jle    469044 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1474>
  468fe0:	lea    0x3200(%r10),%rdi
  468fe7:	mov    %rdi,0xf8(%rsp)
  468fef:	lea    0x3200(%rcx),%rdi
  468ff6:	mov    %rdi,0x138(%rsp)
  468ffe:	cmp    $0x6,%eax
  469001:	jle    469044 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1474>
  469003:	lea    0x3c00(%r10),%rdi
  46900a:	mov    %rdi,0x100(%rsp)
  469012:	lea    0x3c00(%rcx),%rdi
  469019:	mov    %rdi,0x140(%rsp)
  469021:	cmp    $0x7,%eax
  469024:	jle    469044 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1474>
  469026:	lea    0x4600(%r10),%rdi
  46902d:	add    $0x4600,%rcx
  469034:	mov    %rdi,0x108(%rsp)
  46903c:	mov    %rcx,0x148(%rsp)
  469044:	mov    0x28(%rsp),%r14
  469049:	mov    0x20(%rsi),%rsi
  46904d:	mov    %eax,(%rsp)
  469050:	mov    %ebp,%r8d
  469053:	mov    %r10,0x58(%rsp)
  469058:	mov    (%r14),%rdi
  46905b:	push   %rdx
  46905c:	push   $0x0
  46905e:	mov    0x40(%rsp),%r13
  469063:	mov    0xb0(%rsp),%r15d
  46906b:	mov    0xb8(%rsp),%r9
  469073:	mov    %r13,%rcx
  469076:	mov    %r15d,%edx
  469079:	call   46f6a0 <_ZN5guild7kernels3cpu11q2_rows_anyEPKhmiPKPKNS1_4ActQEiPKPfii>
  46907e:	pop    %r12
  469080:	pop    %rax
  469081:	mov    %r13,%rcx
  469084:	mov    0x2918(%rbx),%rdx
  46908b:	mov    %ebp,%r8d
  46908e:	mov    0x20(%rdx),%rsi
  469092:	mov    0x30(%rdx),%rdi
  469096:	mov    0x18(%rdx),%rdx
  46909a:	add    (%r14),%rdi
  46909d:	push   %rdx
  46909e:	mov    %r15d,%edx
  4690a1:	push   $0x0
  4690a3:	mov    0x30(%rsp),%r9
  4690a8:	call   46f6a0 <_ZN5guild7kernels3cpu11q2_rows_anyEPKhmiPKPKNS1_4ActQEiPKPfii>
  4690ad:	mov    0x2918(%rbx),%rsi
  4690b4:	pop    %rax
  4690b5:	pop    %rcx
  4690b6:	mov    (%rsp),%eax
  4690b9:	mov    0x18(%rsi),%rdx
  4690bd:	mov    0x58(%rsp),%r10
  4690c2:	test   %edx,%edx
  4690c4:	jle    4691aa <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x15da>
  4690ca:	mov    0x68(%rsp),%rdi
  4690cf:	movslq %ebp,%rbp
  4690d2:	mov    %eax,0x58(%rsp)
  4690d6:	lea    -0x1(%rdx),%r13d
  4690da:	lea    0x0(%rbp,%rbp,4),%r14
  4690df:	mov    %rsi,0x80(%rsp)
  4690e7:	mov    0x60(%rsp),%rbp
  4690ec:	mov    %rdx,0x88(%rsp)
  4690f4:	lea    -0x13000(%rdi),%r15
  4690fb:	shl    $0x9,%r14
  4690ff:	lea    -0x18000(%rdi),%r12
  469106:	mov    %r10,0x98(%rsp)
  46910e:	add    %r15,%r14
  469111:	mov    %rbx,0x90(%rsp)
  469119:	mov    %r15,%rbx
  46911c:	nopl   0x0(%rax)
  469120:	xor    %r15d,%r15d
  469123:	nopl   0x0(%rax,%rax,1)
  469128:	movss  (%rbx,%r15,4),%xmm1
  46912e:	movaps %xmm1,%xmm0
  469131:	xorps  0x218768(%rip),%xmm0        # 6818a0 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0x28a0>
  469138:	movss  %xmm1,(%rsp)
  46913d:	call   13b60 <expf@plt>
  469142:	addss  0x1cfeba(%rip),%xmm0        # 639004 <_IO_stdin_used+0x4>
  46914a:	movss  (%rsp),%xmm1
  46914f:	mov    %r15,%rax
  469152:	divss  %xmm0,%xmm1
  469156:	mulss  (%r12,%r15,4),%xmm1
  46915c:	movss  %xmm1,0x0(%rbp,%r15,4)
  469163:	add    $0x1,%r15
  469167:	cmp    %rax,%r13
  46916a:	jne    469128 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1558>
  46916c:	add    $0xa00,%rbx
  469173:	add    $0xa00,%r12
  46917a:	add    $0xa00,%rbp
  469181:	cmp    %rbx,%r14
  469184:	jne    469120 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1550>
  469186:	mov    0x58(%rsp),%eax
  46918a:	mov    0x80(%rsp),%rsi
  469192:	mov    0x88(%rsp),%rdx
  46919a:	mov    0x90(%rsp),%rbx
  4691a2:	mov    0x98(%rsp),%r10
  4691aa:	mov    0xa4(%rsp),%ecx
  4691b1:	sub    $0x8,%eax
  4691b4:	addq   $0x40,0x30(%rsp)
  4691ba:	addq   $0x5000,0x60(%rsp)
  4691c3:	cmp    %ecx,%eax
  4691c5:	jne    468f18 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1348>
  4691cb:	mov    0xb8(%rsp),%ebp
  4691d2:	mov    0xbc(%rsp),%r13d
  4691da:	jmp    46830d <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x73d>
  4691df:	call   13650 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  4691e4:	mov    0x58(%rsp),%rsi
  4691e9:	mov    0x78(%rsp),%rcx
  4691ee:	pxor   %xmm0,%xmm0
  4691f2:	pxor   %xmm1,%xmm1
  4691f6:	movupd 0x18(%rbx),%xmm6
  4691fb:	sub    %r12,%rax
  4691fe:	mov    %rsi,%rdx
  469201:	sub    %rcx,%rdx
  469204:	cvtsi2sd %rdx,%xmm0
  469209:	mov    %r12,%rdx
  46920c:	sub    %rsi,%rdx
  46920f:	cvtsi2sd %rdx,%xmm1
  469214:	unpcklpd %xmm1,%xmm0
  469218:	movsd  0x218e30(%rip),%xmm1        # 682050 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0x3050>
  469220:	unpcklpd %xmm1,%xmm1
  469224:	divpd  %xmm1,%xmm0
  469228:	addpd  %xmm6,%xmm0
  46922c:	movups %xmm0,0x18(%rbx)
  469230:	pxor   %xmm0,%xmm0
  469234:	cvtsi2sd %rax,%xmm0
  469239:	divsd  0x218e0f(%rip),%xmm0        # 682050 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0x3050>
  469241:	addsd  0x28(%rbx),%xmm0
  469246:	movsd  %xmm0,0x28(%rbx)
  46924b:	jmp    4684b2 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x8e2>
  469250:	call   13650 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  469255:	mov    %rax,0x58(%rsp)
  46925a:	jmp    468324 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x754>
  46925f:	call   46f3d0 <_ZN5guild7kernels3cpu11cpu_avx2_okEv>
  469264:	test   %al,%al
  469266:	je     46839c <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x7cc>
  46926c:	test   %ebp,%ebp
  46926e:	jle    4692de <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x170e>
  469270:	mov    %fs:0x0,%r11
  469279:	movslq %ebp,%rdx
  46927c:	lea    0x110(%rsp),%r14
  469284:	mov    %ebp,(%rsp)
  469287:	lea    (%r14,%rdx,8),%r15
  46928b:	mov    %rbx,%rbp
  46928e:	lea    -0x5000(%r11),%rcx
  469295:	mov    %r15,%rbx
  469298:	lea    -0xc000(%r11),%r12
  46929f:	mov    %rcx,%r15
  4692a2:	nopw   0x0(%rax,%rax,1)
  4692a8:	mov    0x2918(%rbp),%rax
  4692af:	mov    %r12,%rdx
  4692b2:	mov    %r15,%rdi
  4692b5:	add    $0x8,%r14
  4692b9:	add    $0xa00,%r15
  4692c0:	mov    0x18(%rax),%esi
  4692c3:	call   46f590 <_ZN5guild7kernels3cpu13act_quant_anyEPKfiRNS1_4ActQE>
  4692c8:	mov    %r12,-0x8(%r14)
  4692cc:	add    $0xe00,%r12
  4692d3:	cmp    %r14,%rbx
  4692d6:	jne    4692a8 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x16d8>
  4692d8:	mov    %rbp,%rbx
  4692db:	mov    (%rsp),%ebp
  4692de:	mov    0x1c(%rsp),%r8d
  4692e3:	xor    %r12d,%r12d
  4692e6:	test   %r8d,%r8d
  4692e9:	js     4693d6 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1806>
  4692ef:	cmpq   $0x0,0x80(%rbx)
  4692f7:	movq   $0x0,(%rsp)
  4692ff:	je     469354 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1784>
  469301:	pxor   %xmm0,%xmm0
  469305:	lea    0xc0(%rsp),%rsi
  46930d:	mov    $0x3,%edi
  469312:	movaps %xmm0,0xc0(%rsp)
  46931a:	call   131a0 <clock_gettime@plt>
  46931f:	pxor   %xmm1,%xmm1
  469323:	pxor   %xmm0,%xmm0
  469327:	cvtsi2sdq 0xc0(%rsp),%xmm1
  469331:	mulsd  0x215d67(%rip),%xmm1        # 67f0a0 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0xa0>
  469339:	cvtsi2sdq 0xc8(%rsp),%xmm0
  469343:	divsd  0x218d05(%rip),%xmm0        # 682050 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0x3050>
  46934b:	addsd  %xmm0,%xmm1
  46934f:	movsd  %xmm1,(%rsp)
  469354:	xor    %edx,%edx
  469356:	lea    0x110(%rsp),%r14
  46935e:	cmp    %ebp,%edx
  469360:	jge    4684a4 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x8d4>
  469366:	mov    0x2918(%rbx),%rax
  46936d:	mov    %ebp,%r8d
  469370:	mov    0x18(%rax),%rcx
  469374:	mov    0x28(%rax),%rsi
  469378:	mov    0x38(%rax),%rdi
  46937c:	mov    0x10(%rax),%rax
  469380:	test   %rcx,%rcx
  469383:	lea    0x3f(%rcx),%rdx
  469387:	cmovns %rcx,%rdx
  46938b:	mov    0x28(%rsp),%rcx
  469390:	add    (%rcx),%rdi
  469393:	push   %rax
  469394:	sar    $0x6,%rdx
  469398:	mov    %r14,%rcx
  46939b:	push   $0x0
  46939d:	mov    0x48(%rsp),%r9
  4693a2:	call   46f6a0 <_ZN5guild7kernels3cpu11q2_rows_anyEPKhmiPKPKNS1_4ActQEiPKPfii>
  4693a7:	mov    %r13d,%edx
  4693aa:	pop    %rsi
  4693ab:	pop    %rdi
  4693ac:	cmp    %ebp,%edx
  4693ae:	jl     469366 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1796>
  4693b0:	jmp    4684a4 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x8d4>
  4693b5:	mov    0x18(%rdi),%ecx
  4693b8:	jmp    467d3e <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x16e>
  4693bd:	call   13650 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  4693c2:	mov    %rax,%r12
  4693c5:	jmp    468413 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x843>
  4693ca:	mov    0x2918(%rbx),%rdi
  4693d1:	jmp    468235 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x665>
  4693d6:	call   13650 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  4693db:	mov    %rax,%r12
  4693de:	jmp    4692ef <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x171f>
  4693e3:	lea    0x110(%rsp),%rax
  4693eb:	mov    %rax,0x20(%rsp)
  4693f0:	jmp    468404 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x834>
  4693f5:	call   13390 <__stack_chk_fail@plt>

Disassembly of section .fini:
