// Read bandwidth: 32 threads (one per physical core), each sums its slice of a buffer.
// mode 0: one buffer, pages interleaved across nodes (like the expert arena).
// mode 1: per-node buffers, each thread reads only memory on its own node.
#define _GNU_SOURCE
#include <immintrin.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define NT 32
static size_t per; static char* bufs[2]; static int mode; static pthread_barrier_t bar; static double secs[NT];
static void* alloc(size_t n,int policy,unsigned long mask){void*p=mmap(0,n,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
  if(syscall(SYS_mbind,p,n,policy,&mask,8,0)!=0){perror("mbind");exit(1);}return p;}
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void* run(void* a){ long i=(long)a; int cpu=i; int node=cpu<16?0:1; // cpus 0-15 node0, 16-31 node1
  cpu_set_t s;CPU_ZERO(&s);CPU_SET(cpu,&s);pthread_setaffinity_np(pthread_self(),sizeof s,&s);
  const char* p = mode==0 ? bufs[0]+i*per : bufs[node]+(i%16)*per;
  __m512i acc=_mm512_setzero_si512();
  for(int rep=0;rep<6;rep++){ pthread_barrier_wait(&bar); double t0=now();
    for(size_t o=0;o<per;o+=256){acc=_mm512_add_epi64(acc,_mm512_load_si512(p+o));acc=_mm512_add_epi64(acc,_mm512_load_si512(p+o+64));
      acc=_mm512_add_epi64(acc,_mm512_load_si512(p+o+128));acc=_mm512_add_epi64(acc,_mm512_load_si512(p+o+192));}
    pthread_barrier_wait(&bar); if(rep) secs[i]+=now()-t0; }
  volatile long x=_mm512_reduce_add_epi64(acc);(void)x;return 0;}
int main(int c,char**v){ mode=atoi(v[1]); per=(size_t)256<<20; size_t tot=per*NT;
  if(mode==0){bufs[0]=alloc(tot,3,3UL);memset(bufs[0],1,tot);}
  else{for(int n=0;n<2;n++){bufs[n]=alloc(tot/2,2,1UL<<n);memset(bufs[n],1,tot/2);}}
  pthread_barrier_init(&bar,0,NT);pthread_t th[NT];
  for(long i=0;i<NT;i++)pthread_create(&th[i],0,run,(void*)i);for(int i=0;i<NT;i++)pthread_join(th[i],0);
  double m=0;for(int i=0;i<NT;i++)if(secs[i]>m)m=secs[i];
  printf("%s: %.1f GB/s\n",mode?"node-local":"interleaved",5.0*tot/m/1e9);}
