
/home/ubuntu/guild-kernels-cpu/build/libguild_cpu_plugin.so:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

0000000000080ff0 <_ZN5guild7kernels3cpu13kq256_gu_rowsEiPKhmmiPKPKviPKPfii>:
   80ff0:	endbr64
   80ff4:	push   %r15
   80ff6:	push   %r14
   80ff8:	push   %r13
   80ffa:	push   %r12
   80ffc:	push   %rbp
   80ffd:	push   %rbx
   80ffe:	mov    %rdx,%rbx
   81001:	sub    $0xb8,%rsp
   81008:	mov    0xf8(%rsp),%rax
   81010:	mov    %edi,0x30(%rsp)
   81014:	mov    %rsi,%rdi
   81017:	mov    %rcx,0x40(%rsp)
   8101c:	mov    0x100(%rsp),%esi
   81023:	mov    %rax,0x50(%rsp)
   81028:	mov    0x108(%rsp),%eax
   8102f:	mov    %r8d,0x34(%rsp)
   81034:	mov    %r9,0x48(%rsp)
   81039:	mov    %fs:0x28,%rdx
   81042:	mov    %rdx,0xa8(%rsp)
   8104a:	xor    %edx,%edx
   8104c:	cmp    %eax,%esi
   8104e:	jge    811b3 <_ZN5guild7kernels3cpu13kq256_gu_rowsEiPKhmmiPKPKviPKPfii+0x1c3>
   81054:	mov    0xf0(%rsp),%ecx
   8105b:	test   %ecx,%ecx
   8105d:	jle    811b3 <_ZN5guild7kernels3cpu13kq256_gu_rowsEiPKhmmiPKPKviPKPfii+0x1c3>
   81063:	movslq %esi,%rdx
   81066:	sub    %esi,%eax
   81068:	mov    %rbx,%rcx
   8106b:	mov    %rbx,0x58(%rsp)
   81070:	add    %rdx,%rax
   81073:	imul   %rdx,%rcx
   81077:	lea    0x0(,%rdx,4),%r14
   8107f:	shl    $0x2,%rax
   81083:	lea    0x80(%rsp),%r12
   8108b:	lea    0x60(%rsp),%r15
   81090:	mov    %rax,0x38(%rsp)
   81095:	mov    0xf0(%rsp),%eax
   8109c:	add    %rdi,%rcx
   8109f:	sub    $0x1,%eax
   810a2:	and    $0xfffffff8,%eax
   810a5:	add    $0x8,%eax
   810a8:	mov    %eax,0x2c(%rsp)
   810ac:	nopl   0x0(%rax)
   810b0:	mov    0x40(%rsp),%rax
   810b5:	mov    0x50(%rsp),%rbp
   810ba:	mov    %rcx,0x18(%rsp)
   810bf:	movl   $0x0,0x28(%rsp)
   810c7:	add    %rcx,%rax
   810ca:	mov    %rax,0x20(%rsp)
   810cf:	mov    0x48(%rsp),%rax
   810d4:	mov    %rax,0x10(%rsp)
   810d9:	nopl   0x0(%rax)
   810e0:	mov    0x28(%rsp),%eax
   810e4:	mov    0xf0(%rsp),%ebx
   810eb:	push   %r12
   810ed:	push   %r15
   810ef:	mov    0x20(%rsp),%r8
   810f4:	sub    %eax,%ebx
   810f6:	mov    $0x8,%eax
   810fb:	mov    0x44(%rsp),%ecx
   810ff:	mov    0x30(%rsp),%rdx
   81104:	cmp    %eax,%ebx
   81106:	mov    0x28(%rsp),%rsi
   8110b:	mov    0x40(%rsp),%edi
   8110f:	cmovg  %eax,%ebx
   81112:	xor    %r13d,%r13d
   81115:	mov    %ebx,%r9d
   81118:	movslq %ebx,%rbx
   8111b:	call   803f0 <_ZN5guild7kernels3cpu12_GLOBAL__N_19dot_2rowsEiPKhS4_iPKPKviPfS9_>
   81120:	shl    $0x2,%rbx
   81124:	pop    %rax
   81125:	pop    %rdx
   81126:	cs nopw 0x0(%rax,%rax,1)
   81130:	vmovss (%r15,%r13,1),%xmm1
   81136:	vxorps 0x171002(%rip),%xmm1,%xmm0        # 1f2140 <_fini+0x9904>
   8113e:	vmovss %xmm1,0xc(%rsp)
   81144:	call   1cda0 <expf@plt>
   81149:	vmovss 0xc(%rsp),%xmm1
   8114f:	mov    0x0(%rbp,%r13,2),%rax
   81154:	vaddss 0x171694(%rip),%xmm0,%xmm0        # 1f27f0 <_fini+0x9fb4>
   8115c:	vdivss %xmm0,%xmm1,%xmm0
   81160:	vmulss (%r12,%r13,1),%xmm0,%xmm0
   81166:	add    $0x4,%r13
   8116a:	vmovss %xmm0,(%rax,%r14,1)
   81170:	cmp    %r13,%rbx
   81173:	jne    81130 <_ZN5guild7kernels3cpu13kq256_gu_rowsEiPKhmmiPKPKviPKPfii+0x140>
   81175:	addl   $0x8,0x28(%rsp)
   8117a:	mov    0x2c(%rsp),%esi
   8117e:	add    $0x40,%rbp
   81182:	mov    0x28(%rsp),%eax
   81186:	addq   $0x40,0x10(%rsp)
   8118c:	cmp    %esi,%eax
   8118e:	jne    810e0 <_ZN5guild7kernels3cpu13kq256_gu_rowsEiPKhmmiPKPKviPKPfii+0xf0>
   81194:	mov    0x58(%rsp),%rax
   81199:	mov    0x18(%rsp),%rcx
   8119e:	add    $0x4,%r14
   811a2:	add    %rax,%rcx
   811a5:	mov    0x38(%rsp),%rax
   811aa:	cmp    %rax,%r14
   811ad:	jne    810b0 <_ZN5guild7kernels3cpu13kq256_gu_rowsEiPKhmmiPKPKviPKPfii+0xc0>
   811b3:	mov    0xa8(%rsp),%rax
   811bb:	sub    %fs:0x28,%rax
   811c4:	jne    811d8 <_ZN5guild7kernels3cpu13kq256_gu_rowsEiPKhmmiPKPKviPKPfii+0x1e8>
   811c6:	add    $0xb8,%rsp
   811cd:	pop    %rbx
   811ce:	pop    %rbp
   811cf:	pop    %r12
   811d1:	pop    %r13
   811d3:	pop    %r14
   811d5:	pop    %r15
   811d7:	ret
   811d8:	call   1cae0 <__stack_chk_fail@plt>

Disassembly of section .fini:
