
build-cuda12/ornith_prefill_bench:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

0000000000475e40 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii>:
  475e40:	endbr64
  475e44:	push   %r15
  475e46:	push   %r14
  475e48:	push   %r13
  475e4a:	push   %r12
  475e4c:	push   %rbp
  475e4d:	mov    %r9d,%ebp
  475e50:	push   %rbx
  475e51:	movslq %ecx,%rbx
  475e54:	sub    $0x58,%rsp
  475e58:	mov    %rdi,0x18(%rsp)
  475e5d:	mov    0x90(%rsp),%r12d
  475e65:	mov    %rsi,0x28(%rsp)
  475e6a:	mov    %rdx,0x20(%rsp)
  475e6f:	mov    %r8,0x30(%rsp)
  475e74:	mov    %fs:0x28,%rax
  475e7d:	mov    %rax,0x48(%rsp)
  475e82:	xor    %eax,%eax
  475e84:	movzbl 0x1df3415(%rip),%eax        # 22692a0 <_ZGVZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE8iq4nl_mt>
  475e8b:	test   %al,%al
  475e8d:	je     476174 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x334>
  475e93:	movzbl 0x1df33f6(%rip),%eax        # 2269290 <_ZGVZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6mt_min>
  475e9a:	test   %al,%al
  475e9c:	je     476125 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x2e5>
  475ea2:	movzbl 0x1df33d7(%rip),%eax        # 2269280 <_ZGVZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE2kq>
  475ea9:	test   %al,%al
  475eab:	je     4760ca <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x28a>
  475eb1:	cmpb   $0x0,0x1df33d0(%rip)        # 2269288 <_ZZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE2kq>
  475eb8:	je     475ec3 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x83>
  475eba:	cmp    $0x1,%ebx
  475ebd:	jg     47607c <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x23c>
  475ec3:	call   46f3d0 <_ZN5guild7kernels3cpu11cpu_avx2_okEv>
  475ec8:	test   %al,%al
  475eca:	mov    0x18(%rsp),%rax
  475ecf:	mov    0x4(%rax),%edi
  475ed2:	je     475ee0 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0xa0>
  475ed4:	cmp    %ebx,0x1df33be(%rip)        # 2269298 <_ZZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6mt_min>
  475eda:	jle    475fea <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x1aa>
  475ee0:	call   4d9ed0 <ggml_get_type_traits_cpu>
  475ee5:	mov    0x18(%rsp),%rcx
  475eea:	mov    0x18(%rcx),%r15d
  475eee:	cmp    %r12d,%ebp
  475ef1:	jge    475fc7 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x187>
  475ef7:	test   %ebx,%ebx
  475ef9:	jle    475fc7 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x187>
  475eff:	movslq %ebp,%rsi
  475f02:	sub    %ebp,%r12d
  475f05:	mov    0x8(%rax),%r14
  475f09:	lea    (%r12,%rsi,1),%rax
  475f0d:	mov    %rax,0x38(%rsp)
  475f12:	mov    0x20(%rsp),%rax
  475f17:	lea    (%rax,%rbx,8),%rax
  475f1b:	mov    %rax,(%rsp)
  475f1f:	lea    0x44(%rsp),%rax
  475f24:	mov    %rax,0x8(%rsp)
  475f29:	nopl   0x0(%rax)
  475f30:	mov    0x18(%rsp),%rax
  475f35:	mov    0x20(%rsp),%r12
  475f3a:	mov    %rsi,0x10(%rsp)
  475f3f:	lea    0x0(,%rsi,4),%rbp
  475f47:	mov    0x30(%rsp),%r13
  475f4c:	mov    0x28(%rax),%rbx
  475f50:	imul   %rsi,%rbx
  475f54:	add    0x38(%rax),%rbx
  475f58:	mov    0x28(%rsp),%rax
  475f5d:	add    %rax,%rbx
  475f60:	mov    %rbp,%rax
  475f63:	mov    %rbx,%rbp
  475f66:	mov    %r12,%rbx
  475f69:	mov    %rax,%r12
  475f6c:	nopl   0x0(%rax)
  475f70:	movl   $0x0,0x44(%rsp)
  475f78:	xor    %edx,%edx
  475f7a:	xor    %r8d,%r8d
  475f7d:	mov    %rbp,%rcx
  475f80:	push   $0x1
  475f82:	mov    %r15d,%edi
  475f85:	add    $0x8,%rbx
  475f89:	add    $0x8,%r13
  475f8d:	push   $0x0
  475f8f:	mov    0x18(%rsp),%rsi
  475f94:	mov    -0x8(%rbx),%r9
  475f98:	call   *%r14
  475f9b:	movss  0x54(%rsp),%xmm0
  475fa1:	mov    -0x8(%r13),%rdx
  475fa5:	movss  %xmm0,(%rdx,%r12,1)
  475fab:	pop    %rax
  475fac:	pop    %rdx
  475fad:	cmp    %rbx,(%rsp)
  475fb1:	jne    475f70 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x130>
  475fb3:	mov    0x10(%rsp),%rsi
  475fb8:	add    $0x1,%rsi
  475fbc:	cmp    %rsi,0x38(%rsp)
  475fc1:	jne    475f30 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0xf0>
  475fc7:	mov    0x48(%rsp),%rax
  475fcc:	sub    %fs:0x28,%rax
  475fd5:	jne    4761ae <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x36e>
  475fdb:	add    $0x58,%rsp
  475fdf:	pop    %rbx
  475fe0:	pop    %rbp
  475fe1:	pop    %r12
  475fe3:	pop    %r13
  475fe5:	pop    %r14
  475fe7:	pop    %r15
  475fe9:	ret
  475fea:	cmp    $0x14,%edi
  475fed:	jne    475ee0 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0xa0>
  475ff3:	cmpb   $0x0,0x1df32ae(%rip)        # 22692a8 <_ZZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE8iq4nl_mt>
  475ffa:	je     475ee0 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0xa0>
  476000:	test   %ebx,%ebx
  476002:	jle    475fc7 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x187>
  476004:	mov    0x20(%rsp),%r15
  476009:	lea    -0x1(%rbx),%edx
  47600c:	lea    -0x8(%rbx),%r13d
  476010:	mov    %r12d,0x90(%rsp)
  476018:	and    $0xfffffff8,%edx
  47601b:	mov    %ebp,%r12d
  47601e:	mov    0x30(%rsp),%r14
  476023:	sub    %edx,%r13d
  476026:	mov    %r15,%rbp
  476029:	mov    0x90(%rsp),%r15d
  476031:	mov    %r13d,(%rsp)
  476035:	mov    %rax,%r13
  476038:	mov    $0x8,%r8d
  47603e:	mov    0x28(%r13),%rsi
  476042:	mov    %rbp,%rcx
  476045:	mov    %r14,%r9
  476048:	mov    0x28(%rsp),%rdi
  47604d:	add    0x38(%r13),%rdi
  476051:	push   %r15
  476053:	cmp    %r8d,%ebx
  476056:	push   %r12
  476058:	cmovle %ebx,%r8d
  47605c:	mov    0x18(%r13),%edx
  476060:	sub    $0x8,%ebx
  476063:	add    $0x40,%rbp
  476067:	add    $0x40,%r14
  47606b:	call   4bdee0 <_ZN5guild7kernels3cpu18iq4nl256_down_rowsEPKhmiPKPKviPKPfii>
  476070:	pop    %rcx
  476071:	pop    %rsi
  476072:	cmp    %ebx,(%rsp)
  476075:	jne    476038 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x1f8>
  476077:	jmp    475fc7 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x187>
  47607c:	mov    0x18(%rsp),%rax
  476081:	mov    0x4(%rax),%edi
  476084:	call   4d79e0 <_ZN5guild7kernels3cpu15kq256_supportedEi>
  476089:	test   %al,%al
  47608b:	je     475ec3 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x83>
  476091:	mov    0x18(%rsp),%rax
  476096:	mov    0x28(%rsp),%rsi
  47609b:	sub    $0x8,%rsp
  47609f:	mov    %ebx,%r9d
  4760a2:	mov    0x28(%rax),%rdx
  4760a6:	mov    0x4(%rax),%edi
  4760a9:	add    0x38(%rax),%rsi
  4760ad:	push   %r12
  4760af:	push   %rbp
  4760b0:	push   0x48(%rsp)
  4760b4:	mov    0x40(%rsp),%r8
  4760b9:	mov    0x18(%rax),%ecx
  4760bc:	call   4d83e0 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii>
  4760c1:	add    $0x20,%rsp
  4760c5:	jmp    475fc7 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x187>
  4760ca:	lea    0x1df31af(%rip),%r13        # 2269280 <_ZGVZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE2kq>
  4760d1:	mov    %r13,%rdi
  4760d4:	call   13d80 <__cxa_guard_acquire@plt>
  4760d9:	test   %eax,%eax
  4760db:	je     475eb1 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x71>
  4760e1:	lea    0x1c8833(%rip),%rdi        # 63e91b <_IO_stdin_used+0x591b>
  4760e8:	call   134d0 <getenv@plt>
  4760ed:	mov    %rax,%r14
  4760f0:	call   46f3d0 <_ZN5guild7kernels3cpu11cpu_avx2_okEv>
  4760f5:	test   %al,%al
  4760f7:	je     476112 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x2d2>
  4760f9:	test   %r14,%r14
  4760fc:	je     476112 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x2d2>
  4760fe:	xor    %esi,%esi
  476100:	mov    $0xa,%edx
  476105:	mov    %r14,%rdi
  476108:	call   13f80 <__isoc23_strtol@plt>
  47610d:	test   %eax,%eax
  47610f:	setne  %al
  476112:	mov    %r13,%rdi
  476115:	mov    %al,0x1df316d(%rip)        # 2269288 <_ZZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE2kq>
  47611b:	call   12e70 <__cxa_guard_release@plt>
  476120:	jmp    475eb1 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x71>
  476125:	lea    0x1df3164(%rip),%r13        # 2269290 <_ZGVZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6mt_min>
  47612c:	mov    %r13,%rdi
  47612f:	call   13d80 <__cxa_guard_acquire@plt>
  476134:	test   %eax,%eax
  476136:	je     475ea2 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x62>
  47613c:	lea    0x1c87c8(%rip),%rdi        # 63e90b <_IO_stdin_used+0x590b>
  476143:	call   134d0 <getenv@plt>
  476148:	mov    %rax,%rdi
  47614b:	mov    $0x2,%eax
  476150:	test   %rdi,%rdi
  476153:	je     476161 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x321>
  476155:	mov    $0xa,%edx
  47615a:	xor    %esi,%esi
  47615c:	call   13f80 <__isoc23_strtol@plt>
  476161:	mov    %r13,%rdi
  476164:	mov    %eax,0x1df312e(%rip)        # 2269298 <_ZZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6mt_min>
  47616a:	call   12e70 <__cxa_guard_release@plt>
  47616f:	jmp    475ea2 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x62>
  476174:	lea    0x1df3125(%rip),%r13        # 22692a0 <_ZGVZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE8iq4nl_mt>
  47617b:	mov    %r13,%rdi
  47617e:	call   13d80 <__cxa_guard_acquire@plt>
  476183:	test   %eax,%eax
  476185:	je     475e93 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x53>
  47618b:	lea    0x1c8795(%rip),%rdi        # 63e927 <_IO_stdin_used+0x5927>
  476192:	call   134d0 <getenv@plt>
  476197:	mov    %r13,%rdi
  47619a:	test   %rax,%rax
  47619d:	sete   0x1df3104(%rip)        # 22692a8 <_ZZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE8iq4nl_mt>
  4761a4:	call   12e70 <__cxa_guard_release@plt>
  4761a9:	jmp    475e93 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x53>
  4761ae:	call   13390 <__stack_chk_fail@plt>
  4761b3:	endbr64
  4761b7:	mov    %rax,%rbx
  4761ba:	jmp    2e801 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii.cold>

Disassembly of section .fini:
