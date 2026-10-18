
/home/ubuntu/Guild/build-cuda12/ornith_prefill_bench:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

0000000000464b00 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj>:
  464b00:	endbr64
  464b04:	push   %r15
  464b06:	mov    %rdi,%r15
  464b09:	push   %r14
  464b0b:	push   %r13
  464b0d:	push   %r12
  464b0f:	push   %rbp
  464b10:	push   %rbx
  464b11:	sub    $0x138,%rsp
  464b18:	mov    %rdx,0x48(%rsp)
  464b1d:	mov    %ecx,0x60(%rsp)
  464b21:	mov    %fs:0x28,%rax
  464b2a:	mov    %rax,0x128(%rsp)
  464b32:	xor    %eax,%eax
  464b34:	movslq %esi,%rax
  464b37:	mov    %esi,0x14(%rsp)
  464b3b:	shl    $0x2,%rax
  464b3f:	mov    %rax,0x28(%rsp)
  464b44:	lea    0x1(%rsi),%eax
  464b47:	mov    %eax,0x70(%rsp)
  464b4b:	cltq
  464b4d:	shl    $0x6,%rax
  464b51:	mov    %rax,0x30(%rsp)
  464b56:	lea    0x2780(%rdi),%rax
  464b5d:	mov    %rax,(%rsp)
  464b61:	nopl   0x0(%rax)
  464b68:	mov    (%rsp),%rax
  464b6c:	mov    (%rax),%rax
  464b6f:	mov    %rax,%rdx
  464b72:	shr    $0x20,%rdx
  464b76:	cmp    %edx,0x60(%rsp)
  464b7a:	jne    465740 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xc40>
  464b80:	mov    %eax,%edx
  464b82:	movzwl %ax,%ebp
  464b85:	shr    $0x10,%edx
  464b88:	cmp    %edx,%ebp
  464b8a:	jae    465740 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xc40>
  464b90:	mov    (%rsp),%rbx
  464b94:	lea    0x1(%rax),%rdx
  464b98:	lock cmpxchg %rdx,(%rbx)
  464b9d:	jne    464b6f <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x6f>
  464b9f:	mov    0x14(%rsp),%eax
  464ba3:	test   %eax,%eax
  464ba5:	js     465618 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xb18>
  464bab:	mov    0x2758(%r15),%rax
  464bb2:	mov    0x28(%rsp),%rdi
  464bb7:	mov    %ebp,(%rax,%rdi,1)
  464bba:	cmpq   $0x0,0xf8(%r15)
  464bc2:	movq   $0x0,0x8(%rsp)
  464bcb:	je     464c21 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x121>
  464bcd:	pxor   %xmm0,%xmm0
  464bd1:	lea    0x90(%rsp),%rsi
  464bd9:	mov    $0x3,%edi
  464bde:	movaps %xmm0,0x90(%rsp)
  464be6:	call   141e0 <clock_gettime@plt>
  464beb:	pxor   %xmm1,%xmm1
  464bef:	pxor   %xmm0,%xmm0
  464bf3:	cvtsi2sdq 0x90(%rsp),%xmm1
  464bfd:	mulsd  0x235ffb(%rip),%xmm1        # 69ac00 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0xa0>
  464c05:	cvtsi2sdq 0x98(%rsp),%xmm0
  464c0f:	divsd  0x239499(%rip),%xmm0        # 69e0b0 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0x3550>
  464c17:	addsd  %xmm0,%xmm1
  464c1b:	movsd  %xmm1,0x8(%rsp)
  464c21:	mov    0x2958(%r15),%ecx
  464c28:	cmp    $0xa,%ecx
  464c2b:	ja     464c40 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x140>
  464c2d:	lea    0x29e7cc(%rip),%rdx        # 703400 <_ZN5guild4spec12_GLOBAL__N_1L6kShapeE+0x60>
  464c34:	mov    %ecx,%eax
  464c36:	movslq (%rdx,%rax,4),%rax
  464c3a:	add    %rdx,%rax
  464c3d:	notrack jmp *%rax
  464c40:	cmp    $0x4,%ecx
  464c43:	jle    465650 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xb50>
  464c49:	mov    0x2998(%r15),%rdi
  464c50:	cmp    $0x5,%ecx
  464c53:	je     4662ad <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x17ad>
  464c59:	movslq 0x10(%rdi),%r13
  464c5d:	mov    0x2988(%r15),%r8
  464c64:	mov    %ebp,%eax
  464c66:	movslq 0x2990(%r15),%rsi
  464c6d:	imul   %r8,%rax
  464c71:	cqto
  464c73:	idiv   %rsi
  464c76:	mov    %rax,%r14
  464c79:	lea    0x1(%rbp),%eax
  464c7c:	imul   %r8,%rax
  464c80:	cqto
  464c82:	idiv   %rsi
  464c85:	mov    %rax,%r10
  464c88:	cmp    %rax,%r14
  464c8b:	jge    465696 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xb96>
  464c91:	lea    0xe0(%rsp),%rax
  464c99:	mov    %r10,0x20(%rsp)
  464c9e:	mov    %rax,0x18(%rsp)
  464ca3:	jmp    464dbe <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x2be>
  464ca8:	nopl   0x0(%rax,%rax,1)
  464cb0:	cmpl   $0x2a,0x4(%rdi)
  464cb4:	je     4659f0 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xef0>
  464cba:	add    0x2980(%r15),%rbp
  464cc1:	mov    0x8(%rbp),%ecx
  464cc4:	test   %ecx,%ecx
  464cc6:	jle    464d72 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x272>
  464ccc:	lea    0xc000(%r12),%rax
  464cd4:	mov    %rax,0xe0(%rsp)
  464cdc:	cmp    $0x1,%ecx
  464cdf:	je     464d72 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x272>
  464ce5:	lea    0xc400(%r12),%rax
  464ced:	mov    %rax,0xe8(%rsp)
  464cf5:	cmp    $0x2,%ecx
  464cf8:	je     464d72 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x272>
  464cfa:	lea    0xc800(%r12),%rax
  464d02:	mov    %rax,0xf0(%rsp)
  464d0a:	cmp    $0x3,%ecx
  464d0d:	je     464d72 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x272>
  464d0f:	lea    0xcc00(%r12),%rax
  464d17:	mov    %rax,0xf8(%rsp)
  464d1f:	cmp    $0x4,%ecx
  464d22:	je     464d72 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x272>
  464d24:	lea    0xd000(%r12),%rax
  464d2c:	mov    %rax,0x100(%rsp)
  464d34:	cmp    $0x5,%ecx
  464d37:	je     464d72 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x272>
  464d39:	lea    0xd400(%r12),%rax
  464d41:	mov    %rax,0x108(%rsp)
  464d49:	cmp    $0x6,%ecx
  464d4c:	je     464d72 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x272>
  464d4e:	lea    0xd800(%r12),%rax
  464d56:	mov    %rax,0x110(%rsp)
  464d5e:	cmp    $0x7,%ecx
  464d61:	je     464d72 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x272>
  464d63:	add    $0xdc00,%r12
  464d6a:	mov    %r12,0x118(%rsp)
  464d72:	sub    $0x8,%rsp
  464d76:	mov    0x0(%rbp),%rsi
  464d7a:	mov    %r11d,%r9d
  464d7d:	lea    0x50(%rbp),%r8
  464d81:	push   %rbx
  464d82:	mov    0x28(%rsp),%rdx
  464d87:	mov    %r11,0x48(%rsp)
  464d8c:	call   *0xd8(%r15)
  464d93:	mov    0x48(%rsp),%r11
  464d98:	mov    %ebx,%ecx
  464d9a:	pop    %rax
  464d9b:	pop    %rdx
  464d9c:	sub    %r11d,%ecx
  464d9f:	movslq %ecx,%rbx
  464da2:	add    %rbx,%r14
  464da5:	cmp    %r14,0x20(%rsp)
  464daa:	jle    464f50 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x450>
  464db0:	mov    0x2958(%r15),%ecx
  464db7:	mov    0x2998(%r15),%rdi
  464dbe:	mov    %r14,%rax
  464dc1:	mov    0x20(%rsp),%rbx
  464dc6:	cqto
  464dc8:	idiv   %r13
  464dcb:	sub    %r14,%rbx
  464dce:	add    %rdx,%rbx
  464dd1:	cltq
  464dd3:	mov    %rdx,%r11
  464dd6:	cmp    %r13,%rbx
  464dd9:	lea    0x0(,%rax,8),%rbp
  464de1:	cmovg  %r13,%rbx
  464de5:	sub    %rax,%rbp
  464de8:	mov    %rbp,%r12
  464deb:	shl    $0x5,%rbp
  464def:	shl    $0xd,%r12
  464df3:	add    0x29d8(%r15),%r12
  464dfa:	cmp    $0x5,%ecx
  464dfd:	jne    464cb0 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1b0>
  464e03:	cmpl   $0x2a,(%rdi)
  464e06:	je     465b78 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1078>
  464e0c:	add    0x2980(%r15),%rbp
  464e13:	mov    0x8(%rbp),%ecx
  464e16:	test   %ecx,%ecx
  464e18:	jle    464ebc <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x3bc>
  464e1e:	mov    %r12,0xe0(%rsp)
  464e26:	cmp    $0x1,%ecx
  464e29:	je     464ebc <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x3bc>
  464e2f:	lea    0xa00(%r12),%rax
  464e37:	mov    %rax,0xe8(%rsp)
  464e3f:	cmp    $0x2,%ecx
  464e42:	je     464ebc <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x3bc>
  464e44:	lea    0x1400(%r12),%rax
  464e4c:	mov    %rax,0xf0(%rsp)
  464e54:	cmp    $0x3,%ecx
  464e57:	je     464ebc <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x3bc>
  464e59:	lea    0x1e00(%r12),%rax
  464e61:	mov    %rax,0xf8(%rsp)
  464e69:	cmp    $0x4,%ecx
  464e6c:	je     464ebc <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x3bc>
  464e6e:	lea    0x2800(%r12),%rax
  464e76:	mov    %rax,0x100(%rsp)
  464e7e:	cmp    $0x5,%ecx
  464e81:	je     464ebc <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x3bc>
  464e83:	lea    0x3200(%r12),%rax
  464e8b:	mov    %rax,0x108(%rsp)
  464e93:	cmp    $0x6,%ecx
  464e96:	je     464ebc <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x3bc>
  464e98:	lea    0x3c00(%r12),%rax
  464ea0:	mov    %rax,0x110(%rsp)
  464ea8:	cmp    $0x7,%ecx
  464eab:	je     464ebc <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x3bc>
  464ead:	add    $0x4600,%r12
  464eb4:	mov    %r12,0x118(%rsp)
  464ebc:	sub    $0x8,%rsp
  464ec0:	mov    0x0(%rbp),%rsi
  464ec4:	mov    %r11d,%r9d
  464ec7:	lea    0x90(%rbp),%rdx
  464ece:	push   %rbx
  464ecf:	mov    0x28(%rsp),%r8
  464ed4:	mov    %r11,0x48(%rsp)
  464ed9:	call   *0xd0(%r15)
  464ee0:	mov    0x48(%rsp),%r11
  464ee5:	mov    %ebx,%ecx
  464ee7:	pop    %r9
  464ee9:	pop    %r10
  464eeb:	sub    %r11d,%ecx
  464eee:	jmp    464d9f <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x29f>
  464ef3:	nopl   0x0(%rax,%rax,1)
  464ef8:	movslq 0x29c8(%r15),%rax
  464eff:	mov    %ebp,%edx
  464f01:	movslq 0x29cc(%r15),%rdi
  464f08:	mov    0x29b8(%r15),%rsi
  464f0f:	mov    %rax,%rcx
  464f12:	imul   %rdx,%rax
  464f16:	mov    %rdi,%r8
  464f19:	shl    $0x2,%rdi
  464f1d:	imul   %rdi,%rdx
  464f21:	add    0x29c0(%r15),%rdx
  464f28:	lea    (%rsi,%rax,4),%rsi
  464f2c:	imul   %rdi,%rax
  464f30:	add    0x29b0(%r15),%rax
  464f37:	mov    %rax,%rdi
  464f3a:	call   *0xa0(%r15)
  464f41:	test   %eax,%eax
  464f43:	jne    465b48 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1048>
  464f49:	nopl   0x0(%rax)
  464f50:	cmpq   $0x0,0xf8(%r15)
  464f58:	je     464f68 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x468>
  464f5a:	cmpl   $0x8,0x2958(%r15)
  464f62:	jne    4656a8 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xba8>
  464f68:	lock addl $0x1,0x27c0(%r15)
  464f71:	jmp    464b68 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x68>
  464f76:	cs nopw 0x0(%rax,%rax,1)
  464f80:	mov    0x2998(%r15),%rdi
  464f87:	mov    %rbp,%rdx
  464f8a:	mov    0x29a0(%r15),%rax
  464f91:	shl    $0xc,%rdx
  464f95:	add    0x29a8(%r15),%rdx
  464f9c:	imul   0x10(%rdi),%rbp
  464fa1:	lea    (%rax,%rbp,4),%rsi
  464fa5:	call   *0xc0(%r15)
  464fac:	jmp    464f50 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x450>
  464fae:	xchg   %ax,%ax
  464fb0:	lea    0x0(,%rbp,8),%rbx
  464fb8:	movq   $0x0,0x68(%rsp)
  464fc1:	sub    %rbp,%rbx
  464fc4:	shl    $0x5,%rbx
  464fc8:	add    0x2980(%r15),%rbx
  464fcf:	mov    0x8(%rbx),%eax
  464fd2:	mov    0xd0(%rbx),%rbp
  464fd9:	mov    0xd8(%rbx),%rdi
  464fe0:	test   %rbp,%rbp
  464fe3:	mov    %eax,0x20(%rsp)
  464fe7:	lea    0x90(%rbx),%rax
  464fee:	cmove  %rax,%rbp
  464ff2:	lea    0x50(%rbx),%rax
  464ff6:	test   %rdi,%rdi
  464ff9:	cmovne %rdi,%rax
  464ffd:	mov    %rax,0x50(%rsp)
  465002:	mov    0x14(%rsp),%eax
  465006:	test   %eax,%eax
  465008:	js     465ef0 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x13f0>
  46500e:	mov    0x2998(%r15),%rdi
  465015:	cmpl   $0x2a,(%rdi)
  465018:	je     465f0a <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x140a>
  46501e:	mov    0x20(%rsp),%eax
  465022:	test   %eax,%eax
  465024:	jle    4650d6 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x5d6>
  46502a:	mov    %fs:0x0,%rdx
  465033:	mov    0x20(%rsp),%esi
  465037:	lea    -0x5100(%rdx),%r13
  46503e:	mov    %r13,0xe0(%rsp)
  465046:	cmp    $0x1,%esi
  465049:	je     4650d6 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x5d6>
  46504f:	lea    -0x4700(%rdx),%rax
  465056:	mov    %rax,0xe8(%rsp)
  46505e:	cmp    $0x2,%esi
  465061:	je     4650d6 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x5d6>
  465063:	lea    -0x3d00(%rdx),%rax
  46506a:	mov    %rax,0xf0(%rsp)
  465072:	cmp    $0x3,%esi
  465075:	je     4650d6 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x5d6>
  465077:	lea    -0x3300(%rdx),%rax
  46507e:	mov    %rax,0xf8(%rsp)
  465086:	cmp    $0x4,%esi
  465089:	je     4650d6 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x5d6>
  46508b:	lea    -0x2900(%rdx),%rax
  465092:	mov    %rax,0x100(%rsp)
  46509a:	cmp    $0x5,%esi
  46509d:	je     4650d6 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x5d6>
  46509f:	lea    -0x1f00(%rdx),%rax
  4650a6:	mov    %rax,0x108(%rsp)
  4650ae:	cmp    $0x6,%esi
  4650b1:	je     4650d6 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x5d6>
  4650b3:	lea    -0x1500(%rdx),%rax
  4650ba:	mov    %rax,0x110(%rsp)
  4650c2:	cmp    $0x7,%esi
  4650c5:	je     4650d6 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x5d6>
  4650c7:	add    $0xfffffffffffff500,%rdx
  4650ce:	mov    %rdx,0x118(%rsp)
  4650d6:	mov    0x18(%rdi),%rax
  4650da:	sub    $0x8,%rsp
  4650de:	mov    (%rbx),%rsi
  4650e1:	mov    %rbp,%rdx
  4650e4:	xor    %r9d,%r9d
  4650e7:	push   %rax
  4650e8:	mov    0x30(%rsp),%ecx
  4650ec:	lea    0xf0(%rsp),%r8
  4650f4:	call   *0xd0(%r15)
  4650fb:	pop    %rax
  4650fc:	pop    %rdx
  4650fd:	mov    0x14(%rsp),%eax
  465101:	xor    %r14d,%r14d
  465104:	test   %eax,%eax
  465106:	js     465e50 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1350>
  46510c:	cmpq   $0x0,0xf8(%r15)
  465114:	movq   $0x0,0x40(%rsp)
  46511d:	je     465173 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x673>
  46511f:	pxor   %xmm0,%xmm0
  465123:	lea    0x90(%rsp),%rsi
  46512b:	mov    $0x3,%edi
  465130:	movaps %xmm0,0x90(%rsp)
  465138:	call   141e0 <clock_gettime@plt>
  46513d:	pxor   %xmm1,%xmm1
  465141:	pxor   %xmm0,%xmm0
  465145:	cvtsi2sdq 0x90(%rsp),%xmm1
  46514f:	mulsd  0x235aa9(%rip),%xmm1        # 69ac00 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0xa0>
  465157:	cvtsi2sdq 0x98(%rsp),%xmm0
  465161:	divsd  0x238f47(%rip),%xmm0        # 69e0b0 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0x3550>
  465169:	addsd  %xmm0,%xmm1
  46516d:	movsd  %xmm1,0x40(%rsp)
  465173:	mov    0x2998(%r15),%rax
  46517a:	cmpl   $0x2a,0x4(%rax)
  46517e:	je     466217 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1717>
  465184:	mov    0x20(%rsp),%eax
  465188:	test   %eax,%eax
  46518a:	jle    466540 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1a40>
  465190:	mov    %fs:0x0,%rdx
  465199:	lea    0xe0(%rsp),%rax
  4651a1:	mov    %r14,0x38(%rsp)
  4651a6:	mov    %r15,%r14
  4651a9:	mov    %rax,0x18(%rsp)
  4651ae:	mov    %rax,%r12
  4651b1:	mov    %rbx,%r15
  4651b4:	lea    -0x5100(%rdx),%rbp
  4651bb:	lea    -0xe100(%rdx),%r13
  4651c2:	movslq 0x20(%rsp),%rdx
  4651c7:	lea    (%rax,%rdx,8),%rax
  4651cb:	mov    %rax,%rbx
  4651ce:	xchg   %ax,%ax
  4651d0:	mov    %r13,%rdx
  4651d3:	mov    %rbp,%rsi
  4651d6:	mov    0x2998(%r14),%rdi
  4651dd:	add    $0x8,%r12
  4651e1:	call   *0xc8(%r14)
  4651e8:	add    $0xa00,%rbp
  4651ef:	mov    %r13,-0x8(%r12)
  4651f4:	add    $0x400,%r13
  4651fb:	cmp    %rbx,%r12
  4651fe:	jne    4651d0 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x6d0>
  465200:	mov    %r15,%rbx
  465203:	mov    %r14,%r15
  465206:	mov    0x38(%rsp),%r14
  46520b:	mov    0x14(%rsp),%eax
  46520f:	xor    %ebp,%ebp
  465211:	test   %eax,%eax
  465213:	js     466533 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1a33>
  465219:	cmpq   $0x0,0xf8(%r15)
  465221:	movq   $0x0,0x38(%rsp)
  46522a:	je     465280 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x780>
  46522c:	pxor   %xmm0,%xmm0
  465230:	lea    0x90(%rsp),%rsi
  465238:	mov    $0x3,%edi
  46523d:	movaps %xmm0,0x90(%rsp)
  465245:	call   141e0 <clock_gettime@plt>
  46524a:	pxor   %xmm1,%xmm1
  46524e:	pxor   %xmm0,%xmm0
  465252:	cvtsi2sdq 0x90(%rsp),%xmm1
  46525c:	mulsd  0x23599c(%rip),%xmm1        # 69ac00 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0xa0>
  465264:	cvtsi2sdq 0x98(%rsp),%xmm0
  46526e:	divsd  0x238e3a(%rip),%xmm0        # 69e0b0 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0x3550>
  465276:	addsd  %xmm0,%xmm1
  46527a:	movsd  %xmm1,0x38(%rsp)
  465280:	mov    0x2998(%r15),%rdi
  465287:	sub    $0x8,%rsp
  46528b:	mov    (%rbx),%rsi
  46528e:	xor    %r9d,%r9d
  465291:	mov    0x10(%rdi),%rax
  465295:	push   %rax
  465296:	mov    0x60(%rsp),%r8
  46529b:	mov    0x30(%rsp),%ecx
  46529f:	mov    0x28(%rsp),%rdx
  4652a4:	call   *0xd8(%r15)
  4652ab:	pop    %r12
  4652ad:	pop    %r13
  4652af:	mov    0x14(%rsp),%ebx
  4652b3:	test   %ebx,%ebx
  4652b5:	js     465e78 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1378>
  4652bb:	cmpq   $0x0,0xf8(%r15)
  4652c3:	je     464f68 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x468>
  4652c9:	movsd  0x40(%rsp),%xmm3
  4652cf:	movsd  0x38(%rsp),%xmm1
  4652d5:	lea    0x90(%rsp),%rsi
  4652dd:	mov    $0x3,%edi
  4652e2:	movslq 0x70(%rsp),%rbx
  4652e7:	subsd  %xmm3,%xmm1
  4652eb:	movapd %xmm3,%xmm0
  4652ef:	subsd  0x8(%rsp),%xmm0
  4652f5:	shl    $0x6,%rbx
  4652f9:	add    0xe0(%r15),%rbx
  465300:	unpcklpd %xmm1,%xmm0
  465304:	addpd  (%rbx),%xmm0
  465308:	movaps %xmm0,(%rbx)
  46530b:	pxor   %xmm0,%xmm0
  46530f:	movaps %xmm0,0x90(%rsp)
  465317:	call   141e0 <clock_gettime@plt>
  46531c:	pxor   %xmm0,%xmm0
  465320:	pxor   %xmm1,%xmm1
  465324:	cvtsi2sdq 0x90(%rsp),%xmm0
  46532e:	mulsd  0x2358ca(%rip),%xmm0        # 69ac00 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0xa0>
  465336:	cvtsi2sdq 0x98(%rsp),%xmm1
  465340:	divsd  0x238d68(%rip),%xmm1        # 69e0b0 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0x3550>
  465348:	addsd  %xmm1,%xmm0
  46534c:	subsd  0x38(%rsp),%xmm0
  465352:	addsd  0x10(%rbx),%xmm0
  465357:	movsd  %xmm0,0x10(%rbx)
  46535c:	jmp    464f50 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x450>
  465361:	nopl   0x0(%rax)
  465368:	lea    0x0(,%rbp,8),%r14
  465370:	mov    0x2980(%r15),%rax
  465377:	sub    %rbp,%r14
  46537a:	mov    %r14,%rbp
  46537d:	shl    $0x5,%r14
  465381:	mov    0x8(%rax,%r14,1),%r11d
  465386:	shl    $0xd,%rbp
  46538a:	add    0x29d8(%r15),%rbp
  465391:	test   %r11d,%r11d
  465394:	jle    465696 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xb96>
  46539a:	lea    0x5000(%rbp),%rbx
  4653a1:	lea    0xc000(%rbp),%r13
  4653a8:	xor    %r12d,%r12d
  4653ab:	jmp    4653e8 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x8e8>
  4653ad:	nopl   (%rax)
  4653b0:	mov    %r13,%rdx
  4653b3:	mov    %rbp,%rsi
  4653b6:	call   *0xc8(%r15)
  4653bd:	mov    0x2980(%r15),%rax
  4653c4:	add    $0x1,%r12d
  4653c8:	add    $0xa00,%rbp
  4653cf:	add    $0xe00,%rbx
  4653d6:	add    $0x400,%r13
  4653dd:	cmp    %r12d,0x8(%rax,%r14,1)
  4653e2:	jle    464f50 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x450>
  4653e8:	mov    0x2998(%r15),%rdi
  4653ef:	cmpl   $0x2a,0x4(%rdi)
  4653f3:	jne    4653b0 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x8b0>
  4653f5:	call   4715c0 <_ZN5guild7kernels3cpu11cpu_avx2_okEv>
  4653fa:	test   %al,%al
  4653fc:	jne    465780 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xc80>
  465402:	mov    0x2998(%r15),%rdi
  465409:	jmp    4653b0 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x8b0>
  46540b:	nopl   0x0(%rax,%rax,1)
  465410:	mov    0x295c(%r15),%esi
  465417:	mov    %ebp,%eax
  465419:	mov    0x80(%r15),%r10
  465420:	cltd
  465421:	idiv   %esi
  465423:	lea    (%rdx,%rdx,4),%r9d
  465427:	movslq %eax,%rbp
  46542a:	shl    $0x7,%r9d
  46542e:	mov    %r9d,%eax
  465431:	add    $0x280,%r9d
  465438:	cltd
  465439:	idiv   %esi
  46543b:	mov    %eax,%ecx
  46543d:	mov    %r9d,%eax
  465440:	cltd
  465441:	idiv   %esi
  465443:	lea    0x0(%rbp,%rbp,2),%rdx
  465448:	shl    $0x5,%rbp
  46544c:	add    0x108(%r15),%rbp
  465453:	shl    $0xb,%rdx
  465457:	mov    0x8(%rbp),%rsi
  46545b:	mov    0x0(%rbp),%rdi
  46545f:	add    0x2968(%r15),%rdx
  465466:	mov    %eax,%r9d
  465469:	test   %r10,%r10
  46546c:	je     466400 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1900>
  465472:	lea    0xe0(%rsp),%rax
  46547a:	lea    0xa0(%rsp),%r11
  465482:	mov    %ecx,%r8d
  465485:	mov    %rsi,0xa0(%rsp)
  46548d:	mov    %rdx,0xe0(%rsp)
  465495:	mov    %rax,%rcx
  465498:	mov    %r11,%rsi
  46549b:	mov    $0x1,%edx
  4654a0:	call   *%r10
  4654a3:	jmp    464f50 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x450>
  4654a8:	nopl   0x0(%rax,%rax,1)
  4654b0:	mov    0x2960(%r15),%esi
  4654b7:	mov    %ebp,%eax
  4654b9:	mov    0x88(%r15),%r10
  4654c0:	cltd
  4654c1:	idiv   %esi
  4654c3:	lea    (%rdx,%rdx,4),%r9d
  4654c7:	movslq %eax,%rbp
  4654ca:	shl    $0x9,%r9d
  4654ce:	mov    %r9d,%eax
  4654d1:	add    $0xa00,%r9d
  4654d8:	cltd
  4654d9:	idiv   %esi
  4654db:	mov    %eax,%ecx
  4654dd:	mov    %r9d,%eax
  4654e0:	cltd
  4654e1:	idiv   %esi
  4654e3:	lea    0x0(%rbp,%rbp,2),%rsi
  4654e8:	shl    $0xb,%rsi
  4654ec:	add    0x2968(%r15),%rsi
  4654f3:	add    $0xa00,%rsi
  4654fa:	mov    %eax,%r9d
  4654fd:	mov    %rbp,%rax
  465500:	shl    $0x5,%rax
  465504:	add    0x108(%r15),%rax
  46550b:	mov    0x10(%rax),%rdx
  46550f:	mov    (%rax),%rdi
  465512:	test   %r10,%r10
  465515:	jne    465472 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x972>
  46551b:	mov    %r9d,%r8d
  46551e:	call   482190 <_ZN5guild7kernels3cpu19s2_expert_down_rowsEPKhRK25guild_cpu_planar_act_v1_sPfii>
  465523:	jmp    464f50 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x450>
  465528:	nopl   0x0(%rax,%rax,1)
  465530:	mov    %ebp,%eax
  465532:	shl    $0x5,%rax
  465536:	add    0x108(%r15),%rax
  46553d:	mov    0x10(%rax),%rbp
  465541:	mov    0x8(%rax),%rsi
  465545:	mov    (%rax),%rbx
  465548:	mov    0x80(%r15),%rax
  46554f:	test   %rax,%rax
  465552:	je     4663eb <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x18eb>
  465558:	mov    0x48(%rsp),%r14
  46555d:	xor    %r8d,%r8d
  465560:	mov    $0x1,%edx
  465565:	lea    0xa0(%rsp),%r11
  46556d:	lea    0xe0(%rsp),%rdi
  465575:	mov    %rsi,0xa0(%rsp)
  46557d:	mov    %r11,%rsi
  465580:	mov    $0x280,%r9d
  465586:	lea    0x1c00(%r14),%r12
  46558d:	mov    %rdi,0x18(%rsp)
  465592:	mov    %rdi,%rcx
  465595:	mov    %rbx,%rdi
  465598:	mov    %r11,0x20(%rsp)
  46559d:	lea    0xe00(%r14),%r13
  4655a4:	mov    %r12,0xe0(%rsp)
  4655ac:	call   *%rax
  4655ae:	mov    0x78(%r15),%rax
  4655b2:	mov    0x20(%rsp),%r11
  4655b7:	test   %rax,%rax
  4655ba:	je     46640d <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x190d>
  4655c0:	mov    %r13,%rdx
  4655c3:	mov    $0x280,%esi
  4655c8:	mov    %r12,%rdi
  4655cb:	call   *%rax
  4655cd:	mov    0x88(%r15),%rax
  4655d4:	mov    0x20(%rsp),%r11
  4655d9:	test   %rax,%rax
  4655dc:	je     466437 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1937>
  4655e2:	mov    0x18(%rsp),%rcx
  4655e7:	xor    %r8d,%r8d
  4655ea:	mov    $0x1,%edx
  4655ef:	mov    %r11,%rsi
  4655f2:	mov    %r13,0xa0(%rsp)
  4655fa:	mov    $0xa00,%r9d
  465600:	mov    %rbx,%rdi
  465603:	mov    %rbp,0xe0(%rsp)
  46560b:	call   *%rax
  46560d:	jmp    464f50 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x450>
  465612:	nopw   0x0(%rax,%rax,1)
  465618:	mov    %ebp,0x2760(%r15)
  46561f:	call   146c0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  465624:	mov    %rax,%rcx
  465627:	movabs $0x431bde82d7b634db,%rax
  465631:	imul   %rcx
  465634:	sar    $0x3f,%rcx
  465638:	sar    $0x12,%rdx
  46563c:	sub    %rcx,%rdx
  46563f:	mov    %rdx,0x2768(%r15)
  465646:	jmp    464bba <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xba>
  46564b:	nopl   0x0(%rax,%rax,1)
  465650:	cmp    $0x3,%ecx
  465653:	mov    0x2988(%r15),%rdi
  46565a:	mov    $0x280,%eax
  46565f:	mov    $0xa00,%r13d
  465665:	cmove  %rax,%r13
  465669:	mov    %ebp,%eax
  46566b:	movslq 0x2990(%r15),%rsi
  465672:	imul   %rdi,%rax
  465676:	cqto
  465678:	idiv   %rsi
  46567b:	mov    %rax,%r12
  46567e:	lea    0x1(%rbp),%eax
  465681:	imul   %rdi,%rax
  465685:	cqto
  465687:	idiv   %rsi
  46568a:	mov    %rax,%rbx
  46568d:	cmp    %rax,%r12
  465690:	jl     4657c0 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xcc0>
  465696:	cmpq   $0x0,0xf8(%r15)
  46569e:	je     464f68 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x468>
  4656a4:	nopl   0x0(%rax)
  4656a8:	pxor   %xmm0,%xmm0
  4656ac:	mov    $0x3,%edi
  4656b1:	mov    0x30(%rsp),%rbx
  4656b6:	lea    0x90(%rsp),%rsi
  4656be:	movaps %xmm0,0x90(%rsp)
  4656c6:	add    0xe0(%r15),%rbx
  4656cd:	call   141e0 <clock_gettime@plt>
  4656d2:	pxor   %xmm0,%xmm0
  4656d6:	mov    0x2958(%r15),%eax
  4656dd:	pxor   %xmm1,%xmm1
  4656e1:	cvtsi2sdq 0x90(%rsp),%xmm0
  4656eb:	mulsd  0x23550d(%rip),%xmm0        # 69ac00 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0xa0>
  4656f3:	cvtsi2sdq 0x98(%rsp),%xmm1
  4656fd:	divsd  0x2389ab(%rip),%xmm1        # 69e0b0 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0x3550>
  465705:	addsd  %xmm1,%xmm0
  465709:	subsd  0x8(%rsp),%xmm0
  46570f:	cmp    $0x5,%eax
  465712:	je     465770 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xc70>
  465714:	cmp    $0x6,%eax
  465717:	je     4657a8 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xca8>
  46571d:	sub    $0x7,%eax
  465720:	and    $0xfffffffd,%eax
  465723:	jne    464f68 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x468>
  465729:	addsd  0x8(%rbx),%xmm0
  46572e:	movsd  %xmm0,0x8(%rbx)
  465733:	jmp    464f68 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x468>
  465738:	nopl   0x0(%rax,%rax,1)
  465740:	mov    0x128(%rsp),%rax
  465748:	sub    %fs:0x28,%rax
  465751:	jne    4665f3 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1af3>
  465757:	add    $0x138,%rsp
  46575e:	pop    %rbx
  46575f:	pop    %rbp
  465760:	pop    %r12
  465762:	pop    %r13
  465764:	pop    %r14
  465766:	pop    %r15
  465768:	ret
  465769:	nopl   0x0(%rax)
  465770:	addsd  (%rbx),%xmm0
  465774:	movsd  %xmm0,(%rbx)
  465778:	jmp    464f68 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x468>
  46577d:	nopl   (%rax)
  465780:	mov    0x2998(%r15),%rax
  465787:	mov    %rbx,%rdx
  46578a:	mov    %rbp,%rdi
  46578d:	mov    0x18(%rax),%esi
  465790:	mov    0x78(%r15),%rax
  465794:	test   %rax,%rax
  465797:	je     46649f <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x199f>
  46579d:	call   *%rax
  46579f:	jmp    4653bd <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x8bd>
  4657a4:	nopl   0x0(%rax)
  4657a8:	addsd  0x10(%rbx),%xmm0
  4657ad:	movsd  %xmm0,0x10(%rbx)
  4657b2:	jmp    464f68 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x468>
  4657b7:	nopw   0x0(%rax,%rax,1)
  4657c0:	lea    0xe0(%rsp),%rax
  4657c8:	mov    %r15,%r14
  4657cb:	mov    %rax,0x18(%rsp)
  4657d0:	jmp    4658bc <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xdbc>
  4657d5:	nopl   (%rax)
  4657d8:	test   %edx,%edx
  4657da:	jle    46587f <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xd7f>
  4657e0:	lea    0x5000(%rsi),%rcx
  4657e7:	mov    %rcx,0xe0(%rsp)
  4657ef:	cmp    $0x1,%edx
  4657f2:	je     46587f <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xd7f>
  4657f8:	lea    0x5e00(%rsi),%rcx
  4657ff:	mov    %rcx,0xe8(%rsp)
  465807:	cmp    $0x2,%edx
  46580a:	je     46587f <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xd7f>
  46580c:	lea    0x6c00(%rsi),%rcx
  465813:	mov    %rcx,0xf0(%rsp)
  46581b:	cmp    $0x3,%edx
  46581e:	je     46587f <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xd7f>
  465820:	lea    0x7a00(%rsi),%rcx
  465827:	mov    %rcx,0xf8(%rsp)
  46582f:	cmp    $0x4,%edx
  465832:	je     46587f <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xd7f>
  465834:	lea    0x8800(%rsi),%rcx
  46583b:	mov    %rcx,0x100(%rsp)
  465843:	cmp    $0x5,%edx
  465846:	je     46587f <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xd7f>
  465848:	lea    0x9600(%rsi),%rcx
  46584f:	mov    %rcx,0x108(%rsp)
  465857:	cmp    $0x6,%edx
  46585a:	je     46587f <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xd7f>
  46585c:	lea    0xa400(%rsi),%rcx
  465863:	mov    %rcx,0x110(%rsp)
  46586b:	cmp    $0x7,%edx
  46586e:	je     46587f <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xd7f>
  465870:	add    $0xb200,%rsi
  465877:	mov    %rsi,0x118(%rsp)
  46587f:	mov    (%rax),%rdi
  465882:	lea    0x50(%rax),%rcx
  465886:	mov    0x88(%r14),%rax
  46588d:	test   %rax,%rax
  465890:	je     465a10 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xf10>
  465896:	mov    0x18(%rsp),%rsi
  46589b:	mov    %ebp,%r9d
  46589e:	mov    %r15d,%r8d
  4658a1:	call   *%rax
  4658a3:	sub    %r15d,%ebp
  4658a6:	movslq %ebp,%rbp
  4658a9:	add    %rbp,%r12
  4658ac:	cmp    %r12,%rbx
  4658af:	jle    4659e0 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xee0>
  4658b5:	mov    0x2958(%r14),%ecx
  4658bc:	mov    %r12,%rax
  4658bf:	mov    %rbx,%rbp
  4658c2:	cqto
  4658c4:	sub    %r12,%rbp
  4658c7:	idiv   %r13
  4658ca:	add    %rdx,%rbp
  4658cd:	mov    %rdx,%r15
  4658d0:	movslq %eax,%rdx
  4658d3:	cmp    %r13,%rbp
  4658d6:	lea    0x0(,%rdx,8),%rax
  4658de:	cmovg  %r13,%rbp
  4658e2:	sub    %rdx,%rax
  4658e5:	mov    %rax,%rsi
  4658e8:	shl    $0x5,%rax
  4658ec:	add    0x2980(%r14),%rax
  4658f3:	shl    $0xd,%rsi
  4658f7:	mov    0x8(%rax),%edx
  4658fa:	add    0x29d8(%r14),%rsi
  465901:	cmp    $0x3,%ecx
  465904:	jne    4657d8 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xcd8>
  46590a:	test   %edx,%edx
  46590c:	jle    4659aa <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xeaa>
  465912:	mov    %rsi,0xe0(%rsp)
  46591a:	cmp    $0x1,%edx
  46591d:	je     4659aa <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xeaa>
  465923:	lea    0xa00(%rsi),%rcx
  46592a:	mov    %rcx,0xe8(%rsp)
  465932:	cmp    $0x2,%edx
  465935:	je     4659aa <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xeaa>
  465937:	lea    0x1400(%rsi),%rcx
  46593e:	mov    %rcx,0xf0(%rsp)
  465946:	cmp    $0x3,%edx
  465949:	je     4659aa <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xeaa>
  46594b:	lea    0x1e00(%rsi),%rcx
  465952:	mov    %rcx,0xf8(%rsp)
  46595a:	cmp    $0x4,%edx
  46595d:	je     4659aa <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xeaa>
  46595f:	lea    0x2800(%rsi),%rcx
  465966:	mov    %rcx,0x100(%rsp)
  46596e:	cmp    $0x5,%edx
  465971:	je     4659aa <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xeaa>
  465973:	lea    0x3200(%rsi),%rcx
  46597a:	mov    %rcx,0x108(%rsp)
  465982:	cmp    $0x6,%edx
  465985:	je     4659aa <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xeaa>
  465987:	lea    0x3c00(%rsi),%rcx
  46598e:	mov    %rcx,0x110(%rsp)
  465996:	cmp    $0x7,%edx
  465999:	je     4659aa <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xeaa>
  46599b:	add    $0x4600,%rsi
  4659a2:	mov    %rsi,0x118(%rsp)
  4659aa:	mov    (%rax),%rdi
  4659ad:	lea    0x10(%rax),%rsi
  4659b1:	mov    0x80(%r14),%rax
  4659b8:	test   %rax,%rax
  4659bb:	je     466202 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1702>
  4659c1:	mov    %ebp,%r9d
  4659c4:	sub    %r15d,%ebp
  4659c7:	mov    0x18(%rsp),%rcx
  4659cc:	mov    %r15d,%r8d
  4659cf:	call   *%rax
  4659d1:	movslq %ebp,%rbp
  4659d4:	add    %rbp,%r12
  4659d7:	cmp    %r12,%rbx
  4659da:	jg     4658b5 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xdb5>
  4659e0:	mov    %r14,%r15
  4659e3:	jmp    464f50 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x450>
  4659e8:	nopl   0x0(%rax,%rax,1)
  4659f0:	mov    %r11,0x38(%rsp)
  4659f5:	call   4715c0 <_ZN5guild7kernels3cpu11cpu_avx2_okEv>
  4659fa:	mov    0x38(%rsp),%r11
  4659ff:	test   %al,%al
  465a01:	jne    465a28 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xf28>
  465a03:	mov    0x2998(%r15),%rdi
  465a0a:	jmp    464cba <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1ba>
  465a0f:	nop
  465a10:	mov    0x18(%rsp),%rsi
  465a15:	mov    %ebp,%r9d
  465a18:	mov    %r15d,%r8d
  465a1b:	call   483be0 <_ZN5guild7kernels3cpu25s2_expert_down_rows_multiEPKhPKPK25guild_cpu_planar_act_v1_siPKPfii>
  465a20:	jmp    4658a3 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xda3>
  465a25:	nopl   (%rax)
  465a28:	add    0x2980(%r15),%rbp
  465a2f:	mov    0x8(%rbp),%r8d
  465a33:	test   %r8d,%r8d
  465a36:	jle    465ae9 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xfe9>
  465a3c:	lea    0x5000(%r12),%rax
  465a44:	mov    %rax,0xe0(%rsp)
  465a4c:	cmp    $0x1,%r8d
  465a50:	je     465ae9 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xfe9>
  465a56:	lea    0x5e00(%r12),%rax
  465a5e:	mov    %rax,0xe8(%rsp)
  465a66:	cmp    $0x2,%r8d
  465a6a:	je     465ae9 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xfe9>
  465a6c:	lea    0x6c00(%r12),%rax
  465a74:	mov    %rax,0xf0(%rsp)
  465a7c:	cmp    $0x3,%r8d
  465a80:	je     465ae9 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xfe9>
  465a82:	lea    0x7a00(%r12),%rax
  465a8a:	mov    %rax,0xf8(%rsp)
  465a92:	cmp    $0x4,%r8d
  465a96:	je     465ae9 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xfe9>
  465a98:	lea    0x8800(%r12),%rax
  465aa0:	mov    %rax,0x100(%rsp)
  465aa8:	cmp    $0x5,%r8d
  465aac:	je     465ae9 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xfe9>
  465aae:	lea    0x9600(%r12),%rax
  465ab6:	mov    %rax,0x108(%rsp)
  465abe:	cmp    $0x6,%r8d
  465ac2:	je     465ae9 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xfe9>
  465ac4:	lea    0xa400(%r12),%rax
  465acc:	mov    %rax,0x110(%rsp)
  465ad4:	cmp    $0x7,%r8d
  465ad8:	je     465ae9 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xfe9>
  465ada:	add    $0xb200,%r12
  465ae1:	mov    %r12,0x118(%rsp)
  465ae9:	mov    0x2998(%r15),%rcx
  465af0:	lea    0x50(%rbp),%r9
  465af4:	mov    0x18(%rcx),%rdx
  465af8:	mov    0x38(%rcx),%rdi
  465afc:	mov    0x28(%rcx),%rsi
  465b00:	test   %rdx,%rdx
  465b03:	lea    0x3f(%rdx),%rax
  465b07:	cmovns %rdx,%rax
  465b0b:	add    0x0(%rbp),%rdi
  465b0f:	mov    0x90(%r15),%rbp
  465b16:	sar    $0x6,%rax
  465b1a:	mov    %eax,%edx
  465b1c:	test   %rbp,%rbp
  465b1f:	je     465b58 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1058>
  465b21:	push   %rbx
  465b22:	push   %r11
  465b24:	mov    0x28(%rsp),%rcx
  465b29:	mov    %r11,0x48(%rsp)
  465b2e:	call   *%rbp
  465b30:	pop    %rdi
  465b31:	pop    %r8
  465b33:	mov    0x38(%rsp),%r11
  465b38:	mov    %ebx,%ecx
  465b3a:	sub    %r11d,%ecx
  465b3d:	jmp    464d9f <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x29f>
  465b42:	nopw   0x0(%rax,%rax,1)
  465b48:	movb   $0x1,0x29d0(%r15)
  465b50:	jmp    464f50 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x450>
  465b55:	nopl   (%rax)
  465b58:	push   %rbx
  465b59:	push   %r11
  465b5b:	mov    0x28(%rsp),%rcx
  465b60:	mov    %r11,0x48(%rsp)
  465b65:	call   471980 <_ZN5guild7kernels3cpu11q2_rows_anyEPKhmiPKPK25guild_cpu_planar_act_v1_siPKPfii>
  465b6a:	pop    %rcx
  465b6b:	pop    %rsi
  465b6c:	mov    0x38(%rsp),%r11
  465b71:	jmp    465b38 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1038>
  465b73:	nopl   0x0(%rax,%rax,1)
  465b78:	mov    %rdx,0x38(%rsp)
  465b7d:	call   4715c0 <_ZN5guild7kernels3cpu11cpu_avx2_okEv>
  465b82:	mov    0x38(%rsp),%r11
  465b87:	test   %al,%al
  465b89:	je     465e31 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1331>
  465b8f:	mov    0x2980(%r15),%rdx
  465b96:	add    %rbp,%rdx
  465b99:	mov    0x8(%rdx),%r8d
  465b9d:	test   %r8d,%r8d
  465ba0:	jle    465cd9 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x11d9>
  465ba6:	mov    %fs:0x0,%rdi
  465baf:	lea    -0x1d100(%rdi),%rcx
  465bb6:	lea    -0x22100(%rdi),%rax
  465bbd:	mov    %rcx,0xa0(%rsp)
  465bc5:	mov    %rax,0xe0(%rsp)
  465bcd:	cmp    $0x1,%r8d
  465bd1:	je     465cd9 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x11d9>
  465bd7:	lea    -0x1c700(%rdi),%rsi
  465bde:	mov    %rsi,0xa8(%rsp)
  465be6:	lea    -0x21700(%rdi),%rsi
  465bed:	mov    %rsi,0xe8(%rsp)
  465bf5:	cmp    $0x2,%r8d
  465bf9:	je     465cd9 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x11d9>
  465bff:	lea    -0x1bd00(%rdi),%rsi
  465c06:	mov    %rsi,0xb0(%rsp)
  465c0e:	lea    -0x20d00(%rdi),%rsi
  465c15:	mov    %rsi,0xf0(%rsp)
  465c1d:	cmp    $0x3,%r8d
  465c21:	je     465cd9 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x11d9>
  465c27:	lea    -0x1b300(%rdi),%rsi
  465c2e:	mov    %rsi,0xb8(%rsp)
  465c36:	lea    -0x20300(%rdi),%rsi
  465c3d:	mov    %rsi,0xf8(%rsp)
  465c45:	cmp    $0x4,%r8d
  465c49:	je     465cd9 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x11d9>
  465c4f:	lea    -0x1a900(%rdi),%rsi
  465c56:	mov    %rsi,0xc0(%rsp)
  465c5e:	lea    -0x1f900(%rdi),%rsi
  465c65:	mov    %rsi,0x100(%rsp)
  465c6d:	cmp    $0x5,%r8d
  465c71:	je     465cd9 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x11d9>
  465c73:	lea    -0x19f00(%rdi),%rcx
  465c7a:	lea    -0x1ef00(%rdi),%rax
  465c81:	mov    %rcx,0xc8(%rsp)
  465c89:	mov    %rax,0x108(%rsp)
  465c91:	cmp    $0x6,%r8d
  465c95:	je     465cd9 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x11d9>
  465c97:	lea    -0x19500(%rdi),%rax
  465c9e:	lea    -0x1e500(%rdi),%rsi
  465ca5:	mov    %rax,0xd0(%rsp)
  465cad:	mov    %rsi,0x110(%rsp)
  465cb5:	cmp    $0x7,%r8d
  465cb9:	je     465cd9 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x11d9>
  465cbb:	lea    -0x18b00(%rdi),%rcx
  465cc2:	lea    -0x1db00(%rdi),%rax
  465cc9:	mov    %rcx,0xd8(%rsp)
  465cd1:	mov    %rax,0x118(%rsp)
  465cd9:	mov    0x2998(%r15),%rax
  465ce0:	mov    (%rdx),%rdi
  465ce3:	mov    0x10(%rax),%rcx
  465ce7:	lea    0x3f(%rcx),%rsi
  465ceb:	test   %rcx,%rcx
  465cee:	cmovs  %rsi,%rcx
  465cf2:	mov    0x20(%rax),%rsi
  465cf6:	mov    0x90(%r15),%rax
  465cfd:	sar    $0x6,%rcx
  465d01:	mov    %ecx,%r10d
  465d04:	lea    0x10(%rdx),%rcx
  465d08:	test   %rax,%rax
  465d0b:	je     4665a2 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1aa2>
  465d11:	push   %rbx
  465d12:	mov    %r10d,%edx
  465d15:	push   %r11
  465d17:	mov    %r11,0x50(%rsp)
  465d1c:	lea    0xb0(%rsp),%r9
  465d24:	mov    %r10d,0x48(%rsp)
  465d29:	call   *%rax
  465d2b:	pop    %rdi
  465d2c:	pop    %r8
  465d2e:	mov    0x38(%rsp),%r10d
  465d33:	mov    0x40(%rsp),%r11
  465d38:	mov    0x2980(%r15),%rax
  465d3f:	mov    0x2998(%r15),%rdx
  465d46:	add    %rbp,%rax
  465d49:	mov    0x30(%rdx),%rdi
  465d4d:	mov    0x20(%rdx),%rsi
  465d51:	mov    0x8(%rax),%r8d
  465d55:	lea    0x10(%rax),%rcx
  465d59:	add    (%rax),%rdi
  465d5c:	mov    0x90(%r15),%rax
  465d63:	test   %rax,%rax
  465d66:	je     46657f <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1a7f>
  465d6c:	push   %rbx
  465d6d:	mov    %r10d,%edx
  465d70:	push   %r11
  465d72:	mov    0x28(%rsp),%r9
  465d77:	mov    %r11,0x48(%rsp)
  465d7c:	call   *%rax
  465d7e:	pop    %rax
  465d7f:	pop    %rdx
  465d80:	mov    0x38(%rsp),%r11
  465d85:	mov    0x2980(%r15),%rdx
  465d8c:	mov    %ebx,%ecx
  465d8e:	xor    %eax,%eax
  465d90:	lea    0x0(,%r11,4),%rbx
  465d98:	mov    %fs:0x0,%rdi
  465da1:	sub    %r11d,%ecx
  465da4:	mov    0x8(%rdx,%rbp,1),%r9d
  465da9:	mov    %rdi,0x40(%rsp)
  465dae:	add    $0xfffffffffffddf00,%rdi
  465db5:	mov    %rdi,0x38(%rsp)
  465dba:	test   %r9d,%r9d
  465dbd:	jle    464d9f <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x29f>
  465dc3:	mov    %r14,0x50(%rsp)
  465dc8:	mov    %rbp,%r14
  465dcb:	mov    %eax,%ebp
  465dcd:	mov    %r13,0x58(%rsp)
  465dd2:	mov    %ecx,%r13d
  465dd5:	nopl   (%rax)
  465dd8:	mov    0x38(%rsp),%rax
  465ddd:	lea    (%r12,%rbx,1),%rdx
  465de1:	mov    %r13d,%ecx
  465de4:	lea    (%rbx,%rax,1),%rsi
  465de8:	mov    0x40(%rsp),%rax
  465ded:	lea    -0x1d100(%rbx,%rax,1),%rdi
  465df5:	mov    0xa8(%r15),%rax
  465dfc:	test   %rax,%rax
  465dff:	je     4663c5 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x18c5>
  465e05:	call   *%rax
  465e07:	mov    0x2980(%r15),%rax
  465e0e:	add    $0x1,%ebp
  465e11:	add    $0xa00,%rbx
  465e18:	cmp    0x8(%rax,%r14,1),%ebp
  465e1d:	jl     465dd8 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x12d8>
  465e1f:	mov    %r13d,%ecx
  465e22:	mov    0x50(%rsp),%r14
  465e27:	mov    0x58(%rsp),%r13
  465e2c:	jmp    464d9f <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x29f>
  465e31:	cmpl   $0x5,0x2958(%r15)
  465e39:	mov    0x2998(%r15),%rdi
  465e40:	jne    464cb0 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1b0>
  465e46:	jmp    464e0c <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x30c>
  465e4b:	nopl   0x0(%rax,%rax,1)
  465e50:	call   146c0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  465e55:	mov    %rax,%r14
  465e58:	jmp    46510c <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x60c>
  465e5d:	movq   $0x0,0x38(%rsp)
  465e66:	mov    0x20(%rsp),%r12d
  465e6b:	test   %r12d,%r12d
  465e6e:	jg     466521 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1a21>
  465e74:	nopl   0x0(%rax)
  465e78:	call   146c0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  465e7d:	mov    %r14,%rdx
  465e80:	pxor   %xmm0,%xmm0
  465e84:	pxor   %xmm1,%xmm1
  465e88:	mov    0x68(%rsp),%rbx
  465e8d:	movupd 0x18(%r15),%xmm5
  465e93:	sub    %rbp,%rax
  465e96:	sub    %rbx,%rdx
  465e99:	cvtsi2sd %rdx,%xmm0
  465e9e:	mov    %rbp,%rdx
  465ea1:	sub    %r14,%rdx
  465ea4:	cvtsi2sd %rdx,%xmm1
  465ea9:	unpcklpd %xmm1,%xmm0
  465ead:	movsd  0x2381fb(%rip),%xmm1        # 69e0b0 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0x3550>
  465eb5:	unpcklpd %xmm1,%xmm1
  465eb9:	divpd  %xmm1,%xmm0
  465ebd:	addpd  %xmm5,%xmm0
  465ec1:	movups %xmm0,0x18(%r15)
  465ec6:	pxor   %xmm0,%xmm0
  465eca:	cvtsi2sd %rax,%xmm0
  465ecf:	divsd  0x2381d9(%rip),%xmm0        # 69e0b0 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0x3550>
  465ed7:	addsd  0x28(%r15),%xmm0
  465edd:	movsd  %xmm0,0x28(%r15)
  465ee3:	jmp    4652bb <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x7bb>
  465ee8:	nopl   0x0(%rax,%rax,1)
  465ef0:	call   146c0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  465ef5:	mov    0x2998(%r15),%rdi
  465efc:	mov    %rax,0x68(%rsp)
  465f01:	cmpl   $0x2a,(%rdi)
  465f04:	jne    46501e <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x51e>
  465f0a:	call   4715c0 <_ZN5guild7kernels3cpu11cpu_avx2_okEv>
  465f0f:	test   %al,%al
  465f11:	je     4662b6 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x17b6>
  465f17:	mov    0x2998(%r15),%rsi
  465f1e:	mov    0x20(%rsp),%edi
  465f22:	mov    0x10(%rsi),%rdx
  465f26:	test   %rdx,%rdx
  465f29:	lea    0x3f(%rdx),%rax
  465f2d:	cmovns %rdx,%rax
  465f31:	sar    $0x6,%rax
  465f35:	mov    %eax,%ecx
  465f37:	test   %edi,%edi
  465f39:	jle    4650fd <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x5fd>
  465f3f:	lea    -0x8(%rdi),%r8d
  465f43:	mov    %edi,%eax
  465f45:	sub    $0x1,%edi
  465f48:	mov    %ecx,0x58(%rsp)
  465f4c:	mov    %fs:0x0,%rdx
  465f55:	and    $0xfffffff8,%edi
  465f58:	lea    0x10(%rbx),%r10
  465f5c:	lea    0xa0(%rsp),%r11
  465f64:	sub    %edi,%r8d
  465f67:	lea    0xe0(%rsp),%rdi
  465f6f:	mov    %r10,0x38(%rsp)
  465f74:	mov    %rdi,0x18(%rsp)
  465f79:	lea    -0x12700(%rdx),%rdi
  465f80:	lea    -0x5100(%rdx),%r12
  465f87:	mov    %rdi,0x80(%rsp)
  465f8f:	lea    -0x17700(%rdx),%rdi
  465f96:	lea    -0x13100(%rdx),%r13
  465f9d:	mov    %r8d,0x74(%rsp)
  465fa2:	lea    -0x18100(%rdx),%rbp
  465fa9:	mov    %rdi,0x88(%rsp)
  465fb1:	mov    %r11,0x78(%rsp)
  465fb6:	mov    %rbx,0x40(%rsp)
  465fbb:	mov    %r15,%rbx
  465fbe:	xchg   %ax,%ax
  465fc0:	mov    $0x8,%r14d
  465fc6:	mov    %r13,0xa0(%rsp)
  465fce:	cmp    %r14d,%eax
  465fd1:	mov    %rbp,0xe0(%rsp)
  465fd9:	cmovle %eax,%r14d
  465fdd:	cmp    $0x1,%eax
  465fe0:	jle    4660e4 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x15e4>
  465fe6:	mov    0x80(%rsp),%rdi
  465fee:	mov    %rdi,0xa8(%rsp)
  465ff6:	mov    0x88(%rsp),%rdi
  465ffe:	mov    %rdi,0xe8(%rsp)
  466006:	cmp    $0x2,%eax
  466009:	jle    4660e4 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x15e4>
  46600f:	lea    0x1400(%r13),%rdx
  466016:	mov    %rdx,0xb0(%rsp)
  46601e:	lea    0x1400(%rbp),%rdx
  466025:	mov    %rdx,0xf0(%rsp)
  46602d:	cmp    $0x3,%eax
  466030:	jle    4660e4 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x15e4>
  466036:	lea    0x1e00(%r13),%rdx
  46603d:	mov    %rdx,0xb8(%rsp)
  466045:	lea    0x1e00(%rbp),%rdx
  46604c:	mov    %rdx,0xf8(%rsp)
  466054:	cmp    $0x4,%eax
  466057:	jle    4660e4 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x15e4>
  46605d:	lea    0x2800(%r13),%rdx
  466064:	mov    %rdx,0xc0(%rsp)
  46606c:	lea    0x2800(%rbp),%rdx
  466073:	mov    %rdx,0x100(%rsp)
  46607b:	cmp    $0x5,%eax
  46607e:	jle    4660e4 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x15e4>
  466080:	lea    0x3200(%r13),%rdx
  466087:	mov    %rdx,0xc8(%rsp)
  46608f:	lea    0x3200(%rbp),%rdx
  466096:	mov    %rdx,0x108(%rsp)
  46609e:	cmp    $0x6,%eax
  4660a1:	jle    4660e4 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x15e4>
  4660a3:	lea    0x3c00(%r13),%rdx
  4660aa:	mov    %rdx,0xd0(%rsp)
  4660b2:	lea    0x3c00(%rbp),%rdx
  4660b9:	mov    %rdx,0x110(%rsp)
  4660c1:	cmp    $0x7,%eax
  4660c4:	jle    4660e4 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x15e4>
  4660c6:	lea    0x4600(%r13),%rdx
  4660cd:	mov    %rdx,0xd8(%rsp)
  4660d5:	lea    0x4600(%rbp),%rdx
  4660dc:	mov    %rdx,0x118(%rsp)
  4660e4:	mov    0x40(%rsp),%rdi
  4660e9:	mov    0x90(%rbx),%r11
  4660f0:	mov    %eax,0x64(%rsp)
  4660f4:	mov    0x18(%rsi),%rdx
  4660f8:	mov    0x20(%rsi),%rsi
  4660fc:	mov    (%rdi),%rdi
  4660ff:	test   %r11,%r11
  466102:	je     466476 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1976>
  466108:	push   %rdx
  466109:	mov    %r14d,%r8d
  46610c:	push   $0x0
  46610e:	mov    0x88(%rsp),%r9
  466116:	mov    0x48(%rsp),%rcx
  46611b:	mov    0x68(%rsp),%edx
  46611f:	call   *%r11
  466122:	pop    %r11
  466124:	pop    %r15
  466126:	mov    0x64(%rsp),%eax
  46612a:	mov    0x2998(%rbx),%rdx
  466131:	mov    0x90(%rbx),%r11
  466138:	mov    %eax,0x64(%rsp)
  46613c:	mov    0x30(%rdx),%rdi
  466140:	mov    0x18(%rdx),%rcx
  466144:	mov    0x20(%rdx),%rsi
  466148:	mov    0x40(%rsp),%rdx
  46614d:	add    (%rdx),%rdi
  466150:	test   %r11,%r11
  466153:	je     466452 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1952>
  466159:	push   %rcx
  46615a:	mov    %r14d,%r8d
  46615d:	push   $0x0
  46615f:	mov    0x28(%rsp),%r9
  466164:	mov    0x48(%rsp),%rcx
  466169:	mov    0x68(%rsp),%edx
  46616d:	call   *%r11
  466170:	pop    %rdi
  466171:	pop    %r8
  466173:	mov    0x64(%rsp),%eax
  466177:	movslq %r14d,%r14
  46617a:	mov    %eax,0x64(%rsp)
  46617e:	xor    %r15d,%r15d
  466181:	lea    (%r14,%r14,4),%r14
  466185:	shl    $0x9,%r14
  466189:	jmp    46619e <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x169e>
  46618b:	nopl   0x0(%rax,%rax,1)
  466190:	call   *%rax
  466192:	add    $0xa00,%r15
  466199:	cmp    %r14,%r15
  46619c:	je     4661d8 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x16d8>
  46619e:	mov    0x2998(%rbx),%rax
  4661a5:	lea    (%r15,%r12,1),%rdx
  4661a9:	lea    (%r15,%rbp,1),%rsi
  4661ad:	lea    (%r15,%r13,1),%rdi
  4661b1:	mov    0x18(%rax),%ecx
  4661b4:	mov    0xa8(%rbx),%rax
  4661bb:	test   %rax,%rax
  4661be:	jne    466190 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1690>
  4661c0:	call   48cff0 <_ZN5guild7kernels3cpu6swigluEPKfS3_Pfi>
  4661c5:	add    $0xa00,%r15
  4661cc:	cmp    %r14,%r15
  4661cf:	jne    46619e <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x169e>
  4661d1:	nopl   0x0(%rax)
  4661d8:	mov    0x64(%rsp),%eax
  4661dc:	addq   $0x40,0x38(%rsp)
  4661e2:	add    $0x5000,%r12
  4661e9:	sub    $0x8,%eax
  4661ec:	cmp    %eax,0x74(%rsp)
  4661f0:	je     466552 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1a52>
  4661f6:	mov    0x2998(%rbx),%rsi
  4661fd:	jmp    465fc0 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x14c0>
  466202:	mov    0x18(%rsp),%rcx
  466207:	mov    %ebp,%r9d
  46620a:	mov    %r15d,%r8d
  46620d:	call   482850 <_ZN5guild7kernels3cpu23s2_expert_gu_rows_multiEPKhPKPK25guild_cpu_planar_act_v1_siPKPfii>
  466212:	jmp    4658a3 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xda3>
  466217:	call   4715c0 <_ZN5guild7kernels3cpu11cpu_avx2_okEv>
  46621c:	test   %al,%al
  46621e:	je     465184 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x684>
  466224:	mov    0x20(%rsp),%edi
  466228:	test   %edi,%edi
  46622a:	jle    4665d0 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1ad0>
  466230:	mov    %fs:0x0,%rdx
  466239:	lea    0xe0(%rsp),%rax
  466241:	mov    %r14,0x38(%rsp)
  466246:	mov    %r15,%r14
  466249:	mov    %rax,0x18(%rsp)
  46624e:	mov    %rax,%r13
  466251:	mov    %rbx,%r15
  466254:	lea    -0xc100(%rdx),%rbp
  46625b:	lea    -0x5100(%rdx),%r12
  466262:	movslq %edi,%rdx
  466265:	lea    (%rax,%rdx,8),%rax
  466269:	mov    %rax,%rbx
  46626c:	jmp    46628d <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x178d>
  46626e:	xchg   %ax,%ax
  466270:	call   *%rax
  466272:	mov    %rbp,0x0(%r13)
  466276:	add    $0x8,%r13
  46627a:	add    $0xe00,%rbp
  466281:	add    $0xa00,%r12
  466288:	cmp    %rbx,%r13
  46628b:	je     4662c8 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x17c8>
  46628d:	mov    0x2998(%r14),%rax
  466294:	mov    %rbp,%rdx
  466297:	mov    %r12,%rdi
  46629a:	mov    0x18(%rax),%esi
  46629d:	mov    0x78(%r14),%rax
  4662a1:	test   %rax,%rax
  4662a4:	jne    466270 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1770>
  4662a6:	call   471870 <_ZN5guild7kernels3cpu13act_quant_anyEPKfiR25guild_cpu_planar_act_v1_s>
  4662ab:	jmp    466272 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1772>
  4662ad:	movslq 0x18(%rdi),%r13
  4662b1:	jmp    464c5d <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x15d>
  4662b6:	mov    0x2998(%r15),%rdi
  4662bd:	jmp    46501e <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x51e>
  4662c2:	nopw   0x0(%rax,%rax,1)
  4662c8:	mov    0x14(%rsp),%ebp
  4662cc:	mov    %r15,%rbx
  4662cf:	mov    %r14,%r15
  4662d2:	mov    0x38(%rsp),%r14
  4662d7:	test   %ebp,%ebp
  4662d9:	js     4664a9 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x19a9>
  4662df:	cmpq   $0x0,0xf8(%r15)
  4662e7:	je     4662f0 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x17f0>
  4662e9:	xor    %ebp,%ebp
  4662eb:	jmp    4664bf <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x19bf>
  4662f0:	movq   $0x0,0x38(%rsp)
  4662f9:	xor    %ebp,%ebp
  4662fb:	cmpl   $0x8,0x20(%rsp)
  466300:	jle    46636d <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x186d>
  466302:	mov    0x20(%rsp),%r12d
  466307:	jmp    46631e <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x181e>
  466309:	push   %r8
  46630b:	mov    %r12d,%r8d
  46630e:	push   $0x0
  466310:	mov    0x60(%rsp),%r9
  466315:	mov    0x28(%rsp),%rcx
  46631a:	call   *%rax
  46631c:	pop    %rcx
  46631d:	pop    %rsi
  46631e:	mov    0x2998(%r15),%rax
  466325:	mov    0x18(%rax),%rcx
  466329:	mov    0x38(%rax),%rdi
  46632d:	mov    0x10(%rax),%r8
  466331:	mov    0x28(%rax),%rsi
  466335:	test   %rcx,%rcx
  466338:	lea    0x3f(%rcx),%rdx
  46633c:	mov    0x90(%r15),%rax
  466343:	cmovns %rcx,%rdx
  466347:	add    (%rbx),%rdi
  46634a:	sar    $0x6,%rdx
  46634e:	test   %rax,%rax
  466351:	jne    466309 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1809>
  466353:	push   %r8
  466355:	mov    %r12d,%r8d
  466358:	push   $0x0
  46635a:	mov    0x60(%rsp),%r9
  46635f:	mov    0x28(%rsp),%rcx
  466364:	call   471980 <_ZN5guild7kernels3cpu11q2_rows_anyEPKhmiPKPK25guild_cpu_planar_act_v1_siPKPfii>
  466369:	pop    %rax
  46636a:	pop    %rdx
  46636b:	jmp    46631e <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x181e>
  46636d:	mov    0x2998(%r15),%rdi
  466374:	mov    0x90(%r15),%r10
  46637b:	mov    0x18(%rdi),%rdx
  46637f:	mov    0x10(%rdi),%rcx
  466383:	mov    0x28(%rdi),%rsi
  466387:	mov    0x38(%rdi),%rdi
  46638b:	test   %rdx,%rdx
  46638e:	lea    0x3f(%rdx),%rax
  466392:	cmovns %rdx,%rax
  466396:	add    (%rbx),%rdi
  466399:	sar    $0x6,%rax
  46639d:	mov    %eax,%edx
  46639f:	test   %r10,%r10
  4663a2:	je     46655f <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1a5f>
  4663a8:	push   %rcx
  4663a9:	push   $0x0
  4663ab:	mov    0x60(%rsp),%r9
  4663b0:	mov    0x30(%rsp),%r8d
  4663b5:	mov    0x28(%rsp),%rcx
  4663ba:	call   *%r10
  4663bd:	pop    %rdi
  4663be:	pop    %r8
  4663c0:	jmp    4652af <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x7af>
  4663c5:	call   48cff0 <_ZN5guild7kernels3cpu6swigluEPKfS3_Pfi>
  4663ca:	mov    0x2980(%r15),%rax
  4663d1:	add    $0x1,%ebp
  4663d4:	add    $0xa00,%rbx
  4663db:	cmp    %ebp,0x8(%rax,%r14,1)
  4663e0:	jg     465dd8 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x12d8>
  4663e6:	jmp    465e1f <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x131f>
  4663eb:	mov    0x48(%rsp),%rcx
  4663f0:	mov    %rbp,%rdx
  4663f3:	mov    %rbx,%rdi
  4663f6:	call   47fff0 <_ZN5guild7kernels3cpu16s2_expert_vnni_qEPKhRK25guild_cpu_planar_act_v1_sPfRNS1_13ExpertScratchE>
  4663fb:	jmp    464f50 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x450>
  466400:	mov    %eax,%r8d
  466403:	call   481750 <_ZN5guild7kernels3cpu17s2_expert_gu_rowsEPKhRK25guild_cpu_planar_act_v1_sPfii>
  466408:	jmp    464f50 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x450>
  46640d:	mov    %r13,%rdx
  466410:	mov    $0x280,%esi
  466415:	mov    %r12,%rdi
  466418:	mov    %r11,0x20(%rsp)
  46641d:	call   47fde0 <_ZN5guild7kernels3cpu14act_quant_q8_1EPKfiR25guild_cpu_planar_act_v1_s>
  466422:	mov    0x88(%r15),%rax
  466429:	mov    0x20(%rsp),%r11
  46642e:	test   %rax,%rax
  466431:	jne    4655e2 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0xae2>
  466437:	mov    $0xa00,%r8d
  46643d:	xor    %ecx,%ecx
  46643f:	mov    %rbp,%rdx
  466442:	mov    %r13,%rsi
  466445:	mov    %rbx,%rdi
  466448:	call   482190 <_ZN5guild7kernels3cpu19s2_expert_down_rowsEPKhRK25guild_cpu_planar_act_v1_sPfii>
  46644d:	jmp    464f50 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x450>
  466452:	push   %rcx
  466453:	mov    %r14d,%r8d
  466456:	push   $0x0
  466458:	mov    0x28(%rsp),%r9
  46645d:	mov    0x48(%rsp),%rcx
  466462:	mov    0x68(%rsp),%edx
  466466:	call   471980 <_ZN5guild7kernels3cpu11q2_rows_anyEPKhmiPKPK25guild_cpu_planar_act_v1_siPKPfii>
  46646b:	pop    %rcx
  46646c:	pop    %rsi
  46646d:	mov    0x64(%rsp),%eax
  466471:	jmp    466177 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1677>
  466476:	push   %rdx
  466477:	mov    %r14d,%r8d
  46647a:	push   $0x0
  46647c:	mov    0x88(%rsp),%r9
  466484:	mov    0x48(%rsp),%rcx
  466489:	mov    0x68(%rsp),%edx
  46648d:	call   471980 <_ZN5guild7kernels3cpu11q2_rows_anyEPKhmiPKPK25guild_cpu_planar_act_v1_siPKPfii>
  466492:	pop    %r9
  466494:	pop    %r10
  466496:	mov    0x64(%rsp),%eax
  46649a:	jmp    46612a <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x162a>
  46649f:	call   471870 <_ZN5guild7kernels3cpu13act_quant_anyEPKfiR25guild_cpu_planar_act_v1_s>
  4664a4:	jmp    4653bd <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x8bd>
  4664a9:	call   146c0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  4664ae:	cmpq   $0x0,0xf8(%r15)
  4664b6:	mov    %rax,%rbp
  4664b9:	je     465e5d <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x135d>
  4664bf:	pxor   %xmm0,%xmm0
  4664c3:	lea    0x90(%rsp),%rsi
  4664cb:	mov    $0x3,%edi
  4664d0:	movaps %xmm0,0x90(%rsp)
  4664d8:	call   141e0 <clock_gettime@plt>
  4664dd:	pxor   %xmm1,%xmm1
  4664e1:	pxor   %xmm0,%xmm0
  4664e5:	mov    0x20(%rsp),%r11d
  4664ea:	cvtsi2sdq 0x90(%rsp),%xmm1
  4664f4:	mulsd  0x234704(%rip),%xmm1        # 69ac00 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0xa0>
  4664fc:	cvtsi2sdq 0x98(%rsp),%xmm0
  466506:	divsd  0x237ba2(%rip),%xmm0        # 69e0b0 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0x3550>
  46650e:	addsd  %xmm0,%xmm1
  466512:	movsd  %xmm1,0x38(%rsp)
  466518:	test   %r11d,%r11d
  46651b:	jle    4652af <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x7af>
  466521:	lea    0xe0(%rsp),%rax
  466529:	mov    %rax,0x18(%rsp)
  46652e:	jmp    4662fb <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x17fb>
  466533:	call   146c0 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
  466538:	mov    %rax,%rbp
  46653b:	jmp    465219 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x719>
  466540:	lea    0xe0(%rsp),%rax
  466548:	mov    %rax,0x18(%rsp)
  46654d:	jmp    46520b <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x70b>
  466552:	mov    %rbx,%r15
  466555:	mov    0x40(%rsp),%rbx
  46655a:	jmp    4650fd <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x5fd>
  46655f:	push   %rcx
  466560:	push   $0x0
  466562:	mov    0x60(%rsp),%r9
  466567:	mov    0x30(%rsp),%r8d
  46656c:	mov    0x28(%rsp),%rcx
  466571:	call   471980 <_ZN5guild7kernels3cpu11q2_rows_anyEPKhmiPKPK25guild_cpu_planar_act_v1_siPKPfii>
  466576:	pop    %r9
  466578:	pop    %r10
  46657a:	jmp    4652af <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x7af>
  46657f:	push   %rbx
  466580:	mov    %r10d,%edx
  466583:	push   %r11
  466585:	mov    0x28(%rsp),%r9
  46658a:	mov    %r11,0x48(%rsp)
  46658f:	call   471980 <_ZN5guild7kernels3cpu11q2_rows_anyEPKhmiPKPK25guild_cpu_planar_act_v1_siPKPfii>
  466594:	pop    %r10
  466596:	pop    %r11
  466598:	mov    0x38(%rsp),%r11
  46659d:	jmp    465d85 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1285>
  4665a2:	push   %rbx
  4665a3:	mov    %r10d,%edx
  4665a6:	push   %r11
  4665a8:	lea    0xb0(%rsp),%r9
  4665b0:	mov    %r11,0x50(%rsp)
  4665b5:	mov    %r10d,0x48(%rsp)
  4665ba:	call   471980 <_ZN5guild7kernels3cpu11q2_rows_anyEPKhmiPKPK25guild_cpu_planar_act_v1_siPKPfii>
  4665bf:	pop    %rcx
  4665c0:	pop    %rsi
  4665c1:	mov    0x40(%rsp),%r11
  4665c6:	mov    0x38(%rsp),%r10d
  4665cb:	jmp    465d38 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x1238>
  4665d0:	mov    0x14(%rsp),%r13d
  4665d5:	test   %r13d,%r13d
  4665d8:	js     4664a9 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x19a9>
  4665de:	cmpq   $0x0,0xf8(%r15)
  4665e6:	je     464f68 <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x468>
  4665ec:	xor    %ebp,%ebp
  4665ee:	jmp    4664bf <_ZN5guild7kernels3cpu10ExpertPool5drainEiRNS1_13ExpertScratchEj+0x19bf>
  4665f3:	call   143e0 <__stack_chk_fail@plt>

Disassembly of section .fini:
