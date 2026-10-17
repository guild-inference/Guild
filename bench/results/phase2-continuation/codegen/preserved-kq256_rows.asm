
bench/results/external-kernels/phase1/ornith_prefill_bench:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

00000000004d6e80 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii>:
  4d6e80:	endbr64
  4d6e84:	push   %r15
  4d6e86:	mov    %rsi,%r10
  4d6e89:	mov    %rdx,%rsi
  4d6e8c:	push   %r14
  4d6e8e:	push   %r13
  4d6e90:	push   %r12
  4d6e92:	push   %rbp
  4d6e93:	push   %rbx
  4d6e94:	sub    $0xa8,%rsp
  4d6e9b:	mov    %ecx,0xc(%rsp)
  4d6e9f:	mov    0xe8(%rsp),%ebx
  4d6ea6:	mov    %edi,0x8(%rsp)
  4d6eaa:	mov    0xf0(%rsp),%ecx
  4d6eb1:	mov    %r8,0x38(%rsp)
  4d6eb6:	mov    0xe0(%rsp),%r13
  4d6ebe:	mov    %fs:0x28,%rax
  4d6ec7:	mov    %rax,0x98(%rsp)
  4d6ecf:	xor    %eax,%eax
  4d6ed1:	lea    0x1(%rbx),%eax
  4d6ed4:	cmp    %eax,%ecx
  4d6ed6:	jle    4d7145 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x2c5>
  4d6edc:	mov    %rdx,%rdi
  4d6edf:	lea    -0x2(%rcx),%edx
  4d6ee2:	lea    -0x1(%r9),%r14d
  4d6ee6:	movslq %ebx,%rax
  4d6ee9:	sub    %ebx,%edx
  4d6eeb:	imul   %rax,%rdi
  4d6eef:	lea    0x0(,%rax,4),%rbp
  4d6ef7:	shr    $0x3,%r14d
  4d6efb:	shr    $1,%edx
  4d6efd:	add    $0x1,%r14d
  4d6f01:	lea    (%rdx,%rdx,1),%r8d
  4d6f05:	lea    0x2(%rax,%r8,1),%rax
  4d6f0a:	add    %r10,%rdi
  4d6f0d:	shl    $0x2,%rax
  4d6f11:	mov    %rax,0x30(%rsp)
  4d6f16:	lea    0x0(,%r14,8),%rax
  4d6f1e:	mov    %rax,0x18(%rsp)
  4d6f23:	lea    (%rsi,%rdi,1),%r14
  4d6f27:	test   %r9d,%r9d
  4d6f2a:	jle    4d71a5 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x325>
  4d6f30:	lea    0x70(%rsp),%rax
  4d6f35:	mov    %edx,0x44(%rsp)
  4d6f39:	mov    %rbp,%r12
  4d6f3c:	mov    %r10,0x48(%rsp)
  4d6f41:	mov    %r9d,0x40(%rsp)
  4d6f46:	mov    %ebx,0xe8(%rsp)
  4d6f4d:	mov    %ecx,0xf0(%rsp)
  4d6f54:	mov    %rax,(%rsp)
  4d6f58:	lea    0x50(%rsp),%rax
  4d6f5d:	mov    %rax,0x20(%rsp)
  4d6f62:	mov    %rdi,%rax
  4d6f65:	nopl   (%rax)
  4d6f68:	mov    0x38(%rsp),%rbp
  4d6f6d:	mov    %rsi,0x28(%rsp)
  4d6f72:	xor    %r15d,%r15d
  4d6f75:	mov    %rax,0x10(%rsp)
  4d6f7a:	mov    0x40(%rsp),%ebx
  4d6f7e:	mov    %r13,%rax
  4d6f81:	mov    %r15,%r13
  4d6f84:	mov    %rbp,%r15
  4d6f87:	mov    %rax,%rbp
  4d6f8a:	nopw   0x0(%rax,%rax,1)
  4d6f90:	mov    $0x8,%r9d
  4d6f96:	push   (%rsp)
  4d6f99:	mov    %r14,%rdx
  4d6f9c:	mov    %r15,%r8
  4d6f9f:	push   0x28(%rsp)
  4d6fa3:	cmp    %r9d,%ebx
  4d6fa6:	mov    0x1c(%rsp),%ecx
  4d6faa:	cmovle %ebx,%r9d
  4d6fae:	mov    0x20(%rsp),%rsi
  4d6fb3:	mov    0x18(%rsp),%edi
  4d6fb7:	call   4d6090 <_ZN5guild7kernels3cpu12_GLOBAL__N_19dot_2rowsEiPKhS4_iPKPKviPfS9_>
  4d6fbc:	vmovss 0x60(%rsp),%xmm0
  4d6fc2:	mov    0x0(%rbp,%r13,8),%rdx
  4d6fc7:	vinsertps $0x10,0x80(%rsp),%xmm0,%xmm0
  4d6fd2:	vmovlps %xmm0,(%rdx,%r12,1)
  4d6fd8:	pop    %rax
  4d6fd9:	pop    %rdx
  4d6fda:	cmp    $0x1,%ebx
  4d6fdd:	jle    4d70e6 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x266>
  4d6fe3:	lea    0x1(%r13),%ecx
  4d6fe7:	vmovss 0x54(%rsp),%xmm0
  4d6fed:	mov    %r13d,%edx
  4d6ff0:	vinsertps $0x10,0x74(%rsp),%xmm0,%xmm0
  4d6ff8:	mov    0x0(%rbp,%rcx,8),%rcx
  4d6ffd:	vmovlps %xmm0,(%rcx,%r12,1)
  4d7003:	cmp    $0x2,%ebx
  4d7006:	jle    4d70e6 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x266>
  4d700c:	lea    0x2(%r13),%ecx
  4d7010:	vmovss 0x58(%rsp),%xmm0
  4d7016:	vinsertps $0x10,0x78(%rsp),%xmm0,%xmm0
  4d701e:	mov    0x0(%rbp,%rcx,8),%rcx
  4d7023:	vmovlps %xmm0,(%rcx,%r12,1)
  4d7029:	cmp    $0x3,%ebx
  4d702c:	jle    4d70e6 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x266>
  4d7032:	lea    0x3(%r13),%ecx
  4d7036:	vmovss 0x5c(%rsp),%xmm0
  4d703c:	vinsertps $0x10,0x7c(%rsp),%xmm0,%xmm0
  4d7044:	mov    0x0(%rbp,%rcx,8),%rcx
  4d7049:	vmovlps %xmm0,(%rcx,%r12,1)
  4d704f:	cmp    $0x4,%ebx
  4d7052:	jle    4d70e6 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x266>
  4d7058:	lea    0x4(%r13),%ecx
  4d705c:	vmovss 0x60(%rsp),%xmm0
  4d7062:	vinsertps $0x10,0x80(%rsp),%xmm0,%xmm0
  4d706d:	mov    0x0(%rbp,%rcx,8),%rcx
  4d7072:	vmovlps %xmm0,(%rcx,%r12,1)
  4d7078:	cmp    $0x5,%ebx
  4d707b:	jle    4d70e6 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x266>
  4d707d:	lea    0x5(%r13),%ecx
  4d7081:	vmovss 0x64(%rsp),%xmm0
  4d7087:	vinsertps $0x10,0x84(%rsp),%xmm0,%xmm0
  4d7092:	mov    0x0(%rbp,%rcx,8),%rcx
  4d7097:	vmovlps %xmm0,(%rcx,%r12,1)
  4d709d:	cmp    $0x6,%ebx
  4d70a0:	jle    4d70e6 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x266>
  4d70a2:	lea    0x6(%r13),%ecx
  4d70a6:	vmovss 0x68(%rsp),%xmm0
  4d70ac:	vinsertps $0x10,0x88(%rsp),%xmm0,%xmm0
  4d70b7:	mov    0x0(%rbp,%rcx,8),%rcx
  4d70bc:	vmovlps %xmm0,(%rcx,%r12,1)
  4d70c2:	cmp    $0x7,%ebx
  4d70c5:	jle    4d70e6 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x266>
  4d70c7:	add    $0x7,%edx
  4d70ca:	vmovss 0x6c(%rsp),%xmm0
  4d70d0:	vinsertps $0x10,0x8c(%rsp),%xmm0,%xmm0
  4d70db:	mov    0x0(%rbp,%rdx,8),%rdx
  4d70e0:	vmovlps %xmm0,(%rdx,%r12,1)
  4d70e6:	mov    0x18(%rsp),%rax
  4d70eb:	add    $0x8,%r13
  4d70ef:	sub    $0x8,%ebx
  4d70f2:	add    $0x40,%r15
  4d70f6:	cmp    %rax,%r13
  4d70f9:	jne    4d6f90 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x110>
  4d70ff:	mov    0x28(%rsp),%rsi
  4d7104:	mov    0x30(%rsp),%rax
  4d7109:	add    $0x8,%r12
  4d710d:	mov    %rbp,%r13
  4d7110:	lea    (%rsi,%r14,1),%rdi
  4d7114:	cmp    %rax,%r12
  4d7117:	je     4d7125 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x2a5>
  4d7119:	mov    %rdi,%rax
  4d711c:	lea    (%rsi,%rdi,1),%r14
  4d7120:	jmp    4d6f68 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0xe8>
  4d7125:	mov    0x44(%rsp),%edx
  4d7129:	mov    0x48(%rsp),%r10
  4d712e:	mov    0x40(%rsp),%r9d
  4d7133:	mov    0xe8(%rsp),%ebx
  4d713a:	mov    0xf0(%rsp),%ecx
  4d7141:	lea    0x2(%rbx,%rdx,2),%ebx
  4d7145:	cmp    %ebx,%ecx
  4d7147:	jle    4d717c <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x2fc>
  4d7149:	movslq %ebx,%rdi
  4d714c:	mov    %rsi,%rdx
  4d714f:	sub    %ebx,%ecx
  4d7151:	imul   %rdi,%rdx
  4d7155:	add    %rdi,%rcx
  4d7158:	lea    0x0(,%rdi,4),%rax
  4d7160:	lea    0x0(,%rcx,4),%rdi
  4d7168:	add    %r10,%rdx
  4d716b:	test   %r9d,%r9d
  4d716e:	jg     4d71bd <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x33d>
  4d7170:	add    $0x4,%rax
  4d7174:	add    %rsi,%rdx
  4d7177:	cmp    %rdi,%rax
  4d717a:	jne    4d716b <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x2eb>
  4d717c:	mov    0x98(%rsp),%rax
  4d7184:	sub    %fs:0x28,%rax
  4d718d:	jne    4d735d <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x4dd>
  4d7193:	add    $0xa8,%rsp
  4d719a:	pop    %rbx
  4d719b:	pop    %rbp
  4d719c:	pop    %r12
  4d719e:	pop    %r13
  4d71a0:	pop    %r14
  4d71a2:	pop    %r15
  4d71a4:	ret
  4d71a5:	mov    0x30(%rsp),%rax
  4d71aa:	add    $0x8,%rbp
  4d71ae:	lea    (%rsi,%r14,1),%rdi
  4d71b2:	cmp    %rax,%rbp
  4d71b5:	jne    4d6f23 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0xa3>
  4d71bb:	jmp    4d7141 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x2c1>
  4d71bd:	lea    -0x1(%r9),%ebx
  4d71c1:	lea    0x70(%rsp),%rcx
  4d71c6:	mov    %rax,%r14
  4d71c9:	mov    %r13,%r15
  4d71cc:	shr    $0x3,%ebx
  4d71cf:	mov    %rcx,(%rsp)
  4d71d3:	add    $0x1,%ebx
  4d71d6:	shl    $0x3,%rbx
  4d71da:	mov    %rbx,%rax
  4d71dd:	mov    %rdx,%rbx
  4d71e0:	mov    0x38(%rsp),%r13
  4d71e5:	mov    %rdi,0x18(%rsp)
  4d71ea:	xor    %r12d,%r12d
  4d71ed:	mov    %r9d,%ebp
  4d71f0:	mov    %rsi,0x20(%rsp)
  4d71f5:	mov    %r9d,0x28(%rsp)
  4d71fa:	mov    %r13,%rdx
  4d71fd:	mov    %r14,%r13
  4d7200:	mov    %rbx,0x10(%rsp)
  4d7205:	mov    %rax,%rbx
  4d7208:	mov    %r15,%rax
  4d720b:	mov    %rdx,%r14
  4d720e:	mov    %r12,%r15
  4d7211:	mov    %rax,%r12
  4d7214:	nopl   0x0(%rax)
  4d7218:	mov    $0x8,%r8d
  4d721e:	mov    0xc(%rsp),%edx
  4d7222:	mov    (%rsp),%r9
  4d7226:	mov    %r14,%rcx
  4d7229:	cmp    %r8d,%ebp
  4d722c:	mov    0x10(%rsp),%rsi
  4d7231:	mov    0x8(%rsp),%edi
  4d7235:	cmovle %ebp,%r8d
  4d7239:	call   4ccc40 <_ZN5guild7kernels3cpu12_GLOBAL__N_18dot_rowsEiPKhiPKPKviPf>
  4d723e:	mov    (%r12,%r15,8),%rdx
  4d7242:	vmovss 0x70(%rsp),%xmm0
  4d7248:	vmovss %xmm0,(%rdx,%r13,1)
  4d724e:	cmp    $0x1,%ebp
  4d7251:	jle    4d7317 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x497>
  4d7257:	lea    0x1(%r15),%esi
  4d725b:	vmovss 0x74(%rsp),%xmm0
  4d7261:	mov    %r15d,%edx
  4d7264:	mov    (%r12,%rsi,8),%rsi
  4d7268:	vmovss %xmm0,(%rsi,%r13,1)
  4d726e:	cmp    $0x2,%ebp
  4d7271:	jle    4d7317 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x497>
  4d7277:	lea    0x2(%r15),%esi
  4d727b:	vmovss 0x78(%rsp),%xmm0
  4d7281:	mov    (%r12,%rsi,8),%rsi
  4d7285:	vmovss %xmm0,(%rsi,%r13,1)
  4d728b:	cmp    $0x3,%ebp
  4d728e:	jle    4d7317 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x497>
  4d7294:	lea    0x3(%r15),%esi
  4d7298:	vmovss 0x7c(%rsp),%xmm0
  4d729e:	mov    (%r12,%rsi,8),%rsi
  4d72a2:	vmovss %xmm0,(%rsi,%r13,1)
  4d72a8:	cmp    $0x4,%ebp
  4d72ab:	jle    4d7317 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x497>
  4d72ad:	vmovss 0x80(%rsp),%xmm0
  4d72b6:	lea    0x4(%r15),%esi
  4d72ba:	mov    (%r12,%rsi,8),%rsi
  4d72be:	vmovss %xmm0,(%rsi,%r13,1)
  4d72c4:	cmp    $0x5,%ebp
  4d72c7:	jle    4d7317 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x497>
  4d72c9:	vmovss 0x84(%rsp),%xmm0
  4d72d2:	lea    0x5(%r15),%esi
  4d72d6:	mov    (%r12,%rsi,8),%rsi
  4d72da:	vmovss %xmm0,(%rsi,%r13,1)
  4d72e0:	cmp    $0x6,%ebp
  4d72e3:	jle    4d7317 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x497>
  4d72e5:	vmovss 0x88(%rsp),%xmm0
  4d72ee:	lea    0x6(%r15),%esi
  4d72f2:	mov    (%r12,%rsi,8),%rsi
  4d72f6:	vmovss %xmm0,(%rsi,%r13,1)
  4d72fc:	cmp    $0x7,%ebp
  4d72ff:	jle    4d7317 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x497>
  4d7301:	vmovss 0x8c(%rsp),%xmm0
  4d730a:	add    $0x7,%edx
  4d730d:	mov    (%r12,%rdx,8),%rdx
  4d7311:	vmovss %xmm0,(%rdx,%r13,1)
  4d7317:	add    $0x8,%r15
  4d731b:	sub    $0x8,%ebp
  4d731e:	add    $0x40,%r14
  4d7322:	cmp    %r15,%rbx
  4d7325:	jne    4d7218 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x398>
  4d732b:	mov    %r13,%r14
  4d732e:	mov    0x18(%rsp),%rdi
  4d7333:	mov    %rbx,%rax
  4d7336:	mov    0x20(%rsp),%rsi
  4d733b:	add    $0x4,%r14
  4d733f:	mov    0x10(%rsp),%rbx
  4d7344:	mov    0x28(%rsp),%r9d
  4d7349:	mov    %r12,%r15
  4d734c:	cmp    %r14,%rdi
  4d734f:	je     4d717c <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x2fc>
  4d7355:	add    %rsi,%rbx
  4d7358:	jmp    4d71e0 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii+0x360>
  4d735d:	call   13390 <__stack_chk_fail@plt>

Disassembly of section .fini:
