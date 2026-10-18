
/home/ubuntu/Guild/build-cuda12/ornith_prefill_bench:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

000000000048de90 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii>:
  48de90:	endbr64
  48de94:	push   %r15
  48de96:	push   %r14
  48de98:	push   %r13
  48de9a:	push   %r12
  48de9c:	mov    %r9d,%r12d
  48de9f:	push   %rbp
  48dea0:	movslq %ecx,%rbp
  48dea3:	push   %rbx
  48dea4:	sub    $0x78,%rsp
  48dea8:	mov    %rdi,0x38(%rsp)
  48dead:	mov    0xb0(%rsp),%r13d
  48deb5:	mov    %rsi,0x48(%rsp)
  48deba:	mov    %rdx,0x40(%rsp)
  48debf:	mov    %r8,0x50(%rsp)
  48dec4:	mov    %fs:0x28,%rax
  48decd:	mov    %rax,0x68(%rsp)
  48ded2:	xor    %eax,%eax
  48ded4:	movzbl 0x1e227ad(%rip),%eax        # 22b0688 <_ZGVZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6avx512>
  48dedb:	test   %al,%al
  48dedd:	jne    48def6 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x66>
  48dedf:	lea    0x1e227a2(%rip),%rbx        # 22b0688 <_ZGVZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6avx512>
  48dee6:	mov    %rbx,%rdi
  48dee9:	call   14e00 <__cxa_guard_acquire@plt>
  48deee:	test   %eax,%eax
  48def0:	jne    48e23a <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x3aa>
  48def6:	movzbl 0x1e2277b(%rip),%eax        # 22b0678 <_ZGVZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE4avx2>
  48defd:	test   %al,%al
  48deff:	jne    48df18 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x88>
  48df01:	lea    0x1e22770(%rip),%rbx        # 22b0678 <_ZGVZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE4avx2>
  48df08:	mov    %rbx,%rdi
  48df0b:	call   14e00 <__cxa_guard_acquire@plt>
  48df10:	test   %eax,%eax
  48df12:	jne    48e21a <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x38a>
  48df18:	movzbl 0x1e22749(%rip),%eax        # 22b0668 <_ZGVZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6mt_min>
  48df1f:	test   %al,%al
  48df21:	je     48e19c <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x30c>
  48df27:	movzbl 0x1e2272a(%rip),%eax        # 22b0658 <_ZGVZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE2kq>
  48df2e:	test   %al,%al
  48df30:	je     48e141 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x2b1>
  48df36:	cmpb   $0x0,0x1e22723(%rip)        # 22b0660 <_ZZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE2kq>
  48df3d:	je     48df59 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0xc9>
  48df3f:	mov    0x38(%rsp),%rax
  48df44:	mov    (%rax),%edi
  48df46:	mov    %edi,%eax
  48df48:	and    $0xfffffffd,%eax
  48df4b:	cmp    $0xc,%eax
  48df4e:	jne    48df59 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0xc9>
  48df50:	cmp    $0x1,%ebp
  48df53:	jg     48e26b <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x3db>
  48df59:	movzbl 0x1e226e8(%rip),%eax        # 22b0648 <_ZGVZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6cpu512>
  48df60:	test   %al,%al
  48df62:	je     48e1eb <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x35b>
  48df68:	mov    0x38(%rsp),%rax
  48df6d:	cmp    %ebp,0x1e226fd(%rip)        # 22b0670 <_ZZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6mt_min>
  48df73:	mov    (%rax),%edi
  48df75:	jle    48e0e1 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x251>
  48df7b:	call   4f2cb0 <ggml_get_type_traits_cpu>
  48df80:	mov    0x38(%rsp),%rdi
  48df85:	mov    0x10(%rdi),%r14d
  48df89:	cmp    %r13d,%r12d
  48df8c:	jge    48e0be <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x22e>
  48df92:	test   %ebp,%ebp
  48df94:	jle    48e0be <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x22e>
  48df9a:	movslq %r12d,%rsi
  48df9d:	sub    %r12d,%r13d
  48dfa0:	mov    0x8(%rax),%rbx
  48dfa4:	lea    0x0(%r13,%rsi,1),%rax
  48dfa9:	mov    %rax,0x58(%rsp)
  48dfae:	mov    0x40(%rsp),%rax
  48dfb3:	lea    (%rax,%rbp,8),%rax
  48dfb7:	mov    %rax,0x18(%rsp)
  48dfbc:	lea    0x60(%rsp),%rax
  48dfc1:	mov    %rax,0x20(%rsp)
  48dfc6:	lea    0x64(%rsp),%rax
  48dfcb:	mov    %rax,0x28(%rsp)
  48dfd0:	mov    0x38(%rsp),%rax
  48dfd5:	mov    0x48(%rsp),%rcx
  48dfda:	mov    %rsi,0x30(%rsp)
  48dfdf:	lea    0x0(,%rsi,4),%r12
  48dfe7:	mov    0x40(%rsp),%r15
  48dfec:	mov    0x20(%rax),%rbp
  48dff0:	imul   %rsi,%rbp
  48dff4:	lea    (%rcx,%rbp,1),%r13
  48dff8:	add    0x30(%rax),%rbp
  48dffc:	lea    (%rcx,%rbp,1),%rax
  48e000:	mov    0x50(%rsp),%rbp
  48e005:	mov    %rax,0x10(%rsp)
  48e00a:	nopw   0x0(%rax,%rax,1)
  48e010:	movl   $0x0,0x60(%rsp)
  48e018:	xor    %r8d,%r8d
  48e01b:	mov    %r13,%rcx
  48e01e:	xor    %edx,%edx
  48e020:	movl   $0x0,0x64(%rsp)
  48e028:	mov    %r14d,%edi
  48e02b:	add    $0x8,%r15
  48e02f:	add    $0x8,%rbp
  48e033:	push   $0x1
  48e035:	push   $0x0
  48e037:	mov    0x30(%rsp),%rsi
  48e03c:	mov    -0x8(%r15),%r9
  48e040:	call   *%rbx
  48e042:	pop    %rax
  48e043:	xor    %r8d,%r8d
  48e046:	pop    %rdx
  48e047:	mov    %r14d,%edi
  48e04a:	push   $0x1
  48e04c:	xor    %edx,%edx
  48e04e:	push   $0x0
  48e050:	mov    0x20(%rsp),%rcx
  48e055:	mov    0x38(%rsp),%rsi
  48e05a:	mov    -0x8(%r15),%r9
  48e05e:	call   *%rbx
  48e060:	movss  0x70(%rsp),%xmm1
  48e066:	movaps %xmm1,%xmm0
  48e069:	xorps  0x20f580(%rip),%xmm0        # 69d5f0 <_ZTSZ4mainEUlRKN5guild7runtime10TokenEventEE_+0x2a90>
  48e070:	movss  %xmm1,0x1c(%rsp)
  48e076:	call   14bd0 <expf@plt>
  48e07b:	movss  0x1c(%rsp),%xmm1
  48e081:	mov    -0x8(%rbp),%rdx
  48e085:	addss  0x1c3f77(%rip),%xmm0        # 652004 <_IO_stdin_used+0x4>
  48e08d:	divss  %xmm0,%xmm1
  48e091:	mulss  0x74(%rsp),%xmm1
  48e097:	movss  %xmm1,(%rdx,%r12,1)
  48e09d:	pop    %rcx
  48e09e:	pop    %rsi
  48e09f:	cmp    %r15,0x18(%rsp)
  48e0a4:	jne    48e010 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x180>
  48e0aa:	mov    0x30(%rsp),%rsi
  48e0af:	add    $0x1,%rsi
  48e0b3:	cmp    %rsi,0x58(%rsp)
  48e0b8:	jne    48dfd0 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x140>
  48e0be:	mov    0x68(%rsp),%rax
  48e0c3:	sub    %fs:0x28,%rax
  48e0cc:	jne    48e3ae <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x51e>
  48e0d2:	add    $0x78,%rsp
  48e0d6:	pop    %rbx
  48e0d7:	pop    %rbp
  48e0d8:	pop    %r12
  48e0da:	pop    %r13
  48e0dc:	pop    %r14
  48e0de:	pop    %r15
  48e0e0:	ret
  48e0e1:	call   4aaa10 <_ZN5guild7kernels3cpu15iq512_supportedEi>
  48e0e6:	test   %al,%al
  48e0e8:	jne    48e107 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x277>
  48e0ea:	mov    0x38(%rsp),%rax
  48e0ef:	cmpb   $0x0,0x1e2255a(%rip)        # 22b0650 <_ZZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6cpu512>
  48e0f6:	mov    (%rax),%edi
  48e0f8:	jne    48df7b <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0xeb>
  48e0fe:	call   4d4c00 <_ZN5guild7kernels3cpu15iq256_supportedEi>
  48e103:	test   %al,%al
  48e105:	je     48e135 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x2a5>
  48e107:	mov    0x38(%rsp),%rax
  48e10c:	cmpb   $0x0,0x1e2257d(%rip)        # 22b0690 <_ZZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6avx512>
  48e113:	mov    (%rax),%edi
  48e115:	jne    48e256 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x3c6>
  48e11b:	cmpb   $0x0,0x1e2255e(%rip)        # 22b0680 <_ZZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE4avx2>
  48e122:	je     48df7b <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0xeb>
  48e128:	call   4d4c00 <_ZN5guild7kernels3cpu15iq256_supportedEi>
  48e12d:	test   %al,%al
  48e12f:	jne    48e33b <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x4ab>
  48e135:	mov    0x38(%rsp),%rax
  48e13a:	mov    (%rax),%edi
  48e13c:	jmp    48df7b <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0xeb>
  48e141:	lea    0x1e22510(%rip),%rbx        # 22b0658 <_ZGVZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE2kq>
  48e148:	mov    %rbx,%rdi
  48e14b:	call   14e00 <__cxa_guard_acquire@plt>
  48e150:	test   %eax,%eax
  48e152:	je     48df36 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0xa6>
  48e158:	lea    0x1c9afc(%rip),%rdi        # 657c5b <_IO_stdin_used+0x5c5b>
  48e15f:	call   14540 <getenv@plt>
  48e164:	mov    %rax,%r14
  48e167:	call   4715c0 <_ZN5guild7kernels3cpu11cpu_avx2_okEv>
  48e16c:	test   %al,%al
  48e16e:	je     48e189 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x2f9>
  48e170:	test   %r14,%r14
  48e173:	je     48e189 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x2f9>
  48e175:	xor    %esi,%esi
  48e177:	mov    $0xa,%edx
  48e17c:	mov    %r14,%rdi
  48e17f:	call   15000 <__isoc23_strtol@plt>
  48e184:	test   %eax,%eax
  48e186:	setne  %al
  48e189:	mov    %rbx,%rdi
  48e18c:	mov    %al,0x1e224ce(%rip)        # 22b0660 <_ZZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE2kq>
  48e192:	call   13eb0 <__cxa_guard_release@plt>
  48e197:	jmp    48df36 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0xa6>
  48e19c:	lea    0x1e224c5(%rip),%rbx        # 22b0668 <_ZGVZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6mt_min>
  48e1a3:	mov    %rbx,%rdi
  48e1a6:	call   14e00 <__cxa_guard_acquire@plt>
  48e1ab:	test   %eax,%eax
  48e1ad:	je     48df27 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x97>
  48e1b3:	lea    0x1c9a91(%rip),%rdi        # 657c4b <_IO_stdin_used+0x5c4b>
  48e1ba:	call   14540 <getenv@plt>
  48e1bf:	mov    %rax,%rdi
  48e1c2:	mov    $0x2,%eax
  48e1c7:	test   %rdi,%rdi
  48e1ca:	je     48e1d8 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x348>
  48e1cc:	mov    $0xa,%edx
  48e1d1:	xor    %esi,%esi
  48e1d3:	call   15000 <__isoc23_strtol@plt>
  48e1d8:	mov    %rbx,%rdi
  48e1db:	mov    %eax,0x1e2248f(%rip)        # 22b0670 <_ZZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6mt_min>
  48e1e1:	call   13eb0 <__cxa_guard_release@plt>
  48e1e6:	jmp    48df27 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x97>
  48e1eb:	lea    0x1e22456(%rip),%rbx        # 22b0648 <_ZGVZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6cpu512>
  48e1f2:	mov    %rbx,%rdi
  48e1f5:	call   14e00 <__cxa_guard_acquire@plt>
  48e1fa:	test   %eax,%eax
  48e1fc:	je     48df68 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0xd8>
  48e202:	call   471780 <_ZN5guild7kernels3cpu13cpu_avx512_okEv>
  48e207:	mov    %rbx,%rdi
  48e20a:	mov    %al,0x1e22440(%rip)        # 22b0650 <_ZZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6cpu512>
  48e210:	call   13eb0 <__cxa_guard_release@plt>
  48e215:	jmp    48df68 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0xd8>
  48e21a:	call   4715c0 <_ZN5guild7kernels3cpu11cpu_avx2_okEv>
  48e21f:	test   %al,%al
  48e221:	jne    48e2b1 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x421>
  48e227:	mov    %rbx,%rdi
  48e22a:	mov    %al,0x1e22450(%rip)        # 22b0680 <_ZZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE4avx2>
  48e230:	call   13eb0 <__cxa_guard_release@plt>
  48e235:	jmp    48df18 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x88>
  48e23a:	call   471780 <_ZN5guild7kernels3cpu13cpu_avx512_okEv>
  48e23f:	test   %al,%al
  48e241:	jne    48e29d <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x40d>
  48e243:	mov    %rbx,%rdi
  48e246:	mov    %al,0x1e22444(%rip)        # 22b0690 <_ZZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6avx512>
  48e24c:	call   13eb0 <__cxa_guard_release@plt>
  48e251:	jmp    48def6 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x66>
  48e256:	call   4aaa10 <_ZN5guild7kernels3cpu15iq512_supportedEi>
  48e25b:	test   %al,%al
  48e25d:	jne    48e2c8 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x438>
  48e25f:	mov    0x38(%rsp),%rax
  48e264:	mov    (%rax),%edi
  48e266:	jmp    48e11b <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x28b>
  48e26b:	mov    0x38(%rsp),%rax
  48e270:	mov    0x30(%rax),%rcx
  48e274:	mov    0x20(%rax),%rdx
  48e278:	push   %r13
  48e27a:	push   %r12
  48e27c:	push   0x60(%rsp)
  48e280:	push   %rbp
  48e281:	mov    0x60(%rsp),%r9
  48e286:	mov    0x68(%rsp),%rsi
  48e28b:	mov    0x10(%rax),%r8d
  48e28f:	call   4f07a0 <_ZN5guild7kernels3cpu13kq256_gu_rowsEiPKhmmiPKPKviPKPfii>
  48e294:	add    $0x20,%rsp
  48e298:	jmp    48e0be <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x22e>
  48e29d:	lea    0x1c9989(%rip),%rdi        # 657c2d <_IO_stdin_used+0x5c2d>
  48e2a4:	call   14540 <getenv@plt>
  48e2a9:	test   %rax,%rax
  48e2ac:	sete   %al
  48e2af:	jmp    48e243 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x3b3>
  48e2b1:	lea    0x1c9984(%rip),%rdi        # 657c3c <_IO_stdin_used+0x5c3c>
  48e2b8:	call   14540 <getenv@plt>
  48e2bd:	test   %rax,%rax
  48e2c0:	sete   %al
  48e2c3:	jmp    48e227 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x397>
  48e2c8:	test   %ebp,%ebp
  48e2ca:	jle    48e0be <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x22e>
  48e2d0:	lea    -0x1(%rbp),%eax
  48e2d3:	mov    0x40(%rsp),%r15
  48e2d8:	mov    0x50(%rsp),%r14
  48e2dd:	lea    -0x8(%rbp),%ebx
  48e2e0:	and    $0xfffffff8,%eax
  48e2e3:	sub    %eax,%ebx
  48e2e5:	mov    %ebx,0xc(%rsp)
  48e2e9:	mov    %r15,%rbx
  48e2ec:	mov    %r14,%r15
  48e2ef:	mov    0x38(%rsp),%r14
  48e2f4:	mov    $0x8,%eax
  48e2f9:	mov    0x30(%r14),%rcx
  48e2fd:	mov    0x20(%r14),%rdx
  48e301:	mov    %rbx,%r9
  48e304:	cmp    %eax,%ebp
  48e306:	mov    (%r14),%edi
  48e309:	push   %r13
  48e30b:	cmovle %ebp,%eax
  48e30e:	push   %r12
  48e310:	sub    $0x8,%ebp
  48e313:	add    $0x40,%rbx
  48e317:	push   %r15
  48e319:	add    $0x40,%r15
  48e31d:	push   %rax
  48e31e:	mov    0x68(%rsp),%rsi
  48e323:	mov    0x10(%r14),%r8d
  48e327:	call   4aaa30 <_ZN5guild7kernels3cpu13iq512_gu_rowsEiPKhmmiPKPKviPKPfii>
  48e32c:	add    $0x20,%rsp
  48e330:	cmp    %ebp,0xc(%rsp)
  48e334:	jne    48e2f4 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x464>
  48e336:	jmp    48e0be <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x22e>
  48e33b:	test   %ebp,%ebp
  48e33d:	jle    48e0be <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x22e>
  48e343:	lea    -0x1(%rbp),%eax
  48e346:	mov    0x40(%rsp),%r15
  48e34b:	mov    0x50(%rsp),%r14
  48e350:	lea    -0x8(%rbp),%ebx
  48e353:	and    $0xfffffff8,%eax
  48e356:	sub    %eax,%ebx
  48e358:	mov    %ebx,0xc(%rsp)
  48e35c:	mov    %r15,%rbx
  48e35f:	mov    %r14,%r15
  48e362:	mov    0x38(%rsp),%r14
  48e367:	mov    $0x8,%eax
  48e36c:	mov    0x30(%r14),%rcx
  48e370:	mov    0x20(%r14),%rdx
  48e374:	mov    %rbx,%r9
  48e377:	cmp    %eax,%ebp
  48e379:	mov    (%r14),%edi
  48e37c:	push   %r13
  48e37e:	cmovle %ebp,%eax
  48e381:	push   %r12
  48e383:	sub    $0x8,%ebp
  48e386:	add    $0x40,%rbx
  48e38a:	push   %r15
  48e38c:	add    $0x40,%r15
  48e390:	push   %rax
  48e391:	mov    0x68(%rsp),%rsi
  48e396:	mov    0x10(%r14),%r8d
  48e39a:	call   4d4c20 <_ZN5guild7kernels3cpu13iq256_gu_rowsEiPKhmmiPKPKviPKPfii>
  48e39f:	add    $0x20,%rsp
  48e3a3:	cmp    %ebp,0xc(%rsp)
  48e3a7:	jne    48e367 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x4d7>
  48e3a9:	jmp    48e0be <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x22e>
  48e3ae:	call   143e0 <__stack_chk_fail@plt>
  48e3b3:	endbr64
  48e3b7:	mov    %rax,%rbp
  48e3ba:	jmp    396dc <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii.cold>
  48e3bf:	endbr64
  48e3c3:	mov    %rax,%rbp
  48e3c6:	jmp    396fc <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii.cold+0x20>
  48e3cb:	endbr64
  48e3cf:	mov    %rax,%rbp
  48e3d2:	jmp    39721 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii.cold+0x45>
  48e3d7:	endbr64
  48e3db:	mov    %rax,%rbp
  48e3de:	jmp    39741 <_ZN5guild7kernels3cpu14native_gu_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii.cold+0x65>

Disassembly of section .fini:
