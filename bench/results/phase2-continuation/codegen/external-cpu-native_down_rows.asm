
/home/ubuntu/guild-kernels-cpu/build/libguild_cpu_plugin.so:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

000000000001ec40 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii>:
   1ec40:	endbr64
   1ec44:	push   %r15
   1ec46:	push   %r14
   1ec48:	push   %r13
   1ec4a:	push   %r12
   1ec4c:	push   %rbp
   1ec4d:	mov    %r9d,%ebp
   1ec50:	push   %rbx
   1ec51:	movslq %ecx,%rbx
   1ec54:	sub    $0x58,%rsp
   1ec58:	mov    %rdi,0x18(%rsp)
   1ec5d:	mov    0x90(%rsp),%r12d
   1ec65:	mov    %rsi,0x28(%rsp)
   1ec6a:	mov    %rdx,0x20(%rsp)
   1ec6f:	mov    %r8,0x30(%rsp)
   1ec74:	mov    %fs:0x28,%rax
   1ec7d:	mov    %rax,0x48(%rsp)
   1ec82:	xor    %eax,%eax
   1ec84:	movzbl 0x21ca05(%rip),%eax        # 23b690 <_ZGVZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE8iq4nl_mt>
   1ec8b:	test   %al,%al
   1ec8d:	je     1ef74 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x334>
   1ec93:	movzbl 0x21c9e6(%rip),%eax        # 23b680 <_ZGVZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6mt_min>
   1ec9a:	test   %al,%al
   1ec9c:	je     1ef25 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x2e5>
   1eca2:	movzbl 0x21c9c7(%rip),%eax        # 23b670 <_ZGVZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE2kq>
   1eca9:	test   %al,%al
   1ecab:	je     1eeca <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x28a>
   1ecb1:	cmpb   $0x0,0x21c9c0(%rip)        # 23b678 <_ZZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE2kq>
   1ecb8:	je     1ecc3 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x83>
   1ecba:	cmp    $0x1,%ebx
   1ecbd:	jg     1ee7c <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x23c>
   1ecc3:	call   81b10 <_ZN5guild7kernels3cpu11cpu_avx2_okEv>
   1ecc8:	test   %al,%al
   1ecca:	mov    0x18(%rsp),%rax
   1eccf:	mov    0x4(%rax),%edi
   1ecd2:	je     1ece0 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0xa0>
   1ecd4:	cmp    %ebx,0x21c9ae(%rip)        # 23b688 <_ZZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6mt_min>
   1ecda:	jle    1edea <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x1aa>
   1ece0:	call   9e6d0 <ggml_get_type_traits_cpu>
   1ece5:	mov    0x18(%rsp),%rcx
   1ecea:	mov    0x18(%rcx),%r15d
   1ecee:	cmp    %r12d,%ebp
   1ecf1:	jge    1edc7 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x187>
   1ecf7:	test   %ebx,%ebx
   1ecf9:	jle    1edc7 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x187>
   1ecff:	movslq %ebp,%rsi
   1ed02:	sub    %ebp,%r12d
   1ed05:	mov    0x8(%rax),%r14
   1ed09:	lea    (%r12,%rsi,1),%rax
   1ed0d:	mov    %rax,0x38(%rsp)
   1ed12:	mov    0x20(%rsp),%rax
   1ed17:	lea    (%rax,%rbx,8),%rax
   1ed1b:	mov    %rax,(%rsp)
   1ed1f:	lea    0x44(%rsp),%rax
   1ed24:	mov    %rax,0x8(%rsp)
   1ed29:	nopl   0x0(%rax)
   1ed30:	mov    0x18(%rsp),%rax
   1ed35:	mov    0x20(%rsp),%r12
   1ed3a:	mov    %rsi,0x10(%rsp)
   1ed3f:	lea    0x0(,%rsi,4),%rbp
   1ed47:	mov    0x30(%rsp),%r13
   1ed4c:	mov    0x28(%rax),%rbx
   1ed50:	imul   %rsi,%rbx
   1ed54:	add    0x38(%rax),%rbx
   1ed58:	mov    0x28(%rsp),%rax
   1ed5d:	add    %rax,%rbx
   1ed60:	mov    %rbp,%rax
   1ed63:	mov    %rbx,%rbp
   1ed66:	mov    %r12,%rbx
   1ed69:	mov    %rax,%r12
   1ed6c:	nopl   0x0(%rax)
   1ed70:	movl   $0x0,0x44(%rsp)
   1ed78:	xor    %edx,%edx
   1ed7a:	xor    %r8d,%r8d
   1ed7d:	mov    %rbp,%rcx
   1ed80:	push   $0x1
   1ed82:	mov    %r15d,%edi
   1ed85:	add    $0x8,%rbx
   1ed89:	add    $0x8,%r13
   1ed8d:	push   $0x0
   1ed8f:	mov    0x18(%rsp),%rsi
   1ed94:	mov    -0x8(%rbx),%r9
   1ed98:	call   *%r14
   1ed9b:	movss  0x54(%rsp),%xmm0
   1eda1:	mov    -0x8(%r13),%rdx
   1eda5:	movss  %xmm0,(%rdx,%r12,1)
   1edab:	pop    %rax
   1edac:	pop    %rdx
   1edad:	cmp    %rbx,(%rsp)
   1edb1:	jne    1ed70 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x130>
   1edb3:	mov    0x10(%rsp),%rsi
   1edb8:	add    $0x1,%rsi
   1edbc:	cmp    %rsi,0x38(%rsp)
   1edc1:	jne    1ed30 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0xf0>
   1edc7:	mov    0x48(%rsp),%rax
   1edcc:	sub    %fs:0x28,%rax
   1edd5:	jne    1efae <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x36e>
   1eddb:	add    $0x58,%rsp
   1eddf:	pop    %rbx
   1ede0:	pop    %rbp
   1ede1:	pop    %r12
   1ede3:	pop    %r13
   1ede5:	pop    %r14
   1ede7:	pop    %r15
   1ede9:	ret
   1edea:	cmp    $0x14,%edi
   1eded:	jne    1ece0 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0xa0>
   1edf3:	cmpb   $0x0,0x21c89e(%rip)        # 23b698 <_ZZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE8iq4nl_mt>
   1edfa:	je     1ece0 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0xa0>
   1ee00:	test   %ebx,%ebx
   1ee02:	jle    1edc7 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x187>
   1ee04:	mov    0x20(%rsp),%r15
   1ee09:	lea    -0x1(%rbx),%edx
   1ee0c:	lea    -0x8(%rbx),%r13d
   1ee10:	mov    %r12d,0x90(%rsp)
   1ee18:	and    $0xfffffff8,%edx
   1ee1b:	mov    %ebp,%r12d
   1ee1e:	mov    0x30(%rsp),%r14
   1ee23:	sub    %edx,%r13d
   1ee26:	mov    %r15,%rbp
   1ee29:	mov    0x90(%rsp),%r15d
   1ee31:	mov    %r13d,(%rsp)
   1ee35:	mov    %rax,%r13
   1ee38:	mov    $0x8,%r8d
   1ee3e:	mov    0x28(%r13),%rsi
   1ee42:	mov    %rbp,%rcx
   1ee45:	mov    %r14,%r9
   1ee48:	mov    0x28(%rsp),%rdi
   1ee4d:	add    0x38(%r13),%rdi
   1ee51:	push   %r15
   1ee53:	cmp    %r8d,%ebx
   1ee56:	push   %r12
   1ee58:	cmovle %ebx,%r8d
   1ee5c:	mov    0x18(%r13),%edx
   1ee60:	sub    $0x8,%ebx
   1ee63:	add    $0x40,%rbp
   1ee67:	add    $0x40,%r14
   1ee6b:	call   66ce0 <_ZN5guild7kernels3cpu18iq4nl256_down_rowsEPKhmiPKPKviPKPfii>
   1ee70:	pop    %rcx
   1ee71:	pop    %rsi
   1ee72:	cmp    %ebx,(%rsp)
   1ee75:	jne    1ee38 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x1f8>
   1ee77:	jmp    1edc7 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x187>
   1ee7c:	mov    0x18(%rsp),%rax
   1ee81:	mov    0x4(%rax),%edi
   1ee84:	call   807e0 <_ZN5guild7kernels3cpu15kq256_supportedEi>
   1ee89:	test   %al,%al
   1ee8b:	je     1ecc3 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x83>
   1ee91:	mov    0x18(%rsp),%rax
   1ee96:	mov    0x28(%rsp),%rsi
   1ee9b:	sub    $0x8,%rsp
   1ee9f:	mov    %ebx,%r9d
   1eea2:	mov    0x28(%rax),%rdx
   1eea6:	mov    0x4(%rax),%edi
   1eea9:	add    0x38(%rax),%rsi
   1eead:	push   %r12
   1eeaf:	push   %rbp
   1eeb0:	push   0x48(%rsp)
   1eeb4:	mov    0x40(%rsp),%r8
   1eeb9:	mov    0x18(%rax),%ecx
   1eebc:	call   811e0 <_ZN5guild7kernels3cpu10kq256_rowsEiPKhmiPKPKviPKPfii>
   1eec1:	add    $0x20,%rsp
   1eec5:	jmp    1edc7 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x187>
   1eeca:	lea    0x21c79f(%rip),%r13        # 23b670 <_ZGVZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE2kq>
   1eed1:	mov    %r13,%rdi
   1eed4:	call   1ce40 <__cxa_guard_acquire@plt>
   1eed9:	test   %eax,%eax
   1eedb:	je     1ecb1 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x71>
   1eee1:	lea    0x1ca1a8(%rip),%rdi        # 1e9090 <_fini+0x854>
   1eee8:	call   1cb90 <getenv@plt>
   1eeed:	mov    %rax,%r14
   1eef0:	call   81b10 <_ZN5guild7kernels3cpu11cpu_avx2_okEv>
   1eef5:	test   %al,%al
   1eef7:	je     1ef12 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x2d2>
   1eef9:	test   %r14,%r14
   1eefc:	je     1ef12 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x2d2>
   1eefe:	xor    %esi,%esi
   1ef00:	mov    $0xa,%edx
   1ef05:	mov    %r14,%rdi
   1ef08:	call   1ced0 <__isoc23_strtol@plt>
   1ef0d:	test   %eax,%eax
   1ef0f:	setne  %al
   1ef12:	mov    %r13,%rdi
   1ef15:	mov    %al,0x21c75d(%rip)        # 23b678 <_ZZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE2kq>
   1ef1b:	call   1c900 <__cxa_guard_release@plt>
   1ef20:	jmp    1ecb1 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x71>
   1ef25:	lea    0x21c754(%rip),%r13        # 23b680 <_ZGVZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6mt_min>
   1ef2c:	mov    %r13,%rdi
   1ef2f:	call   1ce40 <__cxa_guard_acquire@plt>
   1ef34:	test   %eax,%eax
   1ef36:	je     1eca2 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x62>
   1ef3c:	lea    0x1ca13d(%rip),%rdi        # 1e9080 <_fini+0x844>
   1ef43:	call   1cb90 <getenv@plt>
   1ef48:	mov    %rax,%rdi
   1ef4b:	mov    $0x2,%eax
   1ef50:	test   %rdi,%rdi
   1ef53:	je     1ef61 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x321>
   1ef55:	mov    $0xa,%edx
   1ef5a:	xor    %esi,%esi
   1ef5c:	call   1ced0 <__isoc23_strtol@plt>
   1ef61:	mov    %r13,%rdi
   1ef64:	mov    %eax,0x21c71e(%rip)        # 23b688 <_ZZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE6mt_min>
   1ef6a:	call   1c900 <__cxa_guard_release@plt>
   1ef6f:	jmp    1eca2 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x62>
   1ef74:	lea    0x21c715(%rip),%r13        # 23b690 <_ZGVZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE8iq4nl_mt>
   1ef7b:	mov    %r13,%rdi
   1ef7e:	call   1ce40 <__cxa_guard_acquire@plt>
   1ef83:	test   %eax,%eax
   1ef85:	je     1ec93 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x53>
   1ef8b:	lea    0x1ca10a(%rip),%rdi        # 1e909c <_fini+0x860>
   1ef92:	call   1cb90 <getenv@plt>
   1ef97:	mov    %r13,%rdi
   1ef9a:	test   %rax,%rax
   1ef9d:	sete   0x21c6f4(%rip)        # 23b698 <_ZZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfiiE8iq4nl_mt>
   1efa4:	call   1c900 <__cxa_guard_release@plt>
   1efa9:	jmp    1ec93 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii+0x53>
   1efae:	call   1cae0 <__stack_chk_fail@plt>
   1efb3:	endbr64
   1efb7:	mov    %rax,%rbx
   1efba:	jmp    1cfd9 <_ZN5guild7kernels3cpu16native_down_rowsERK28guild_cpu_expert_format_v1_sPKhPKPKviPKPfii.cold>

Disassembly of section .fini:
