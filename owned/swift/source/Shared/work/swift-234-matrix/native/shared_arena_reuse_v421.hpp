#pragma once
// Owned tmpfs arena admission. No CUDA calls, no payload copy, no stale-bank reload.
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/vfs.h>
#include <sys/statvfs.h>
#include <fcntl.h>
#include <unistd.h>
#include <array>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
namespace swift_shared_v421 {
struct Seal {
 char magic[24]{};uint64_t version=1,pack_hash=0,payload=0,allocation=0,device=0,inode=0,size=0;
 int64_t mtime_sec=0,mtime_nsec=0;uint64_t uid=0;char boot[48]{};
};
class Admission {
 int lock_=-1;std::string path_,seal_,boot_;uint64_t hash_=0,payload_=0,allocation_=0;
 bool active_=false,reused_=false;uint64_t admitted_device_=0,admitted_inode_=0;
 static bool regular(const struct stat&s){return S_ISREG(s.st_mode)&&s.st_uid==geteuid();}
 bool failure(std::string&e,const std::string&why){e="V421 shared arena refused: "+why;return false;}
 bool payload_stat(struct stat&s,std::string&e){
  int fd=::open(path_.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW);if(fd<0)return failure(e,"payload open");struct statfs fs{};bool ok=fstat(fd,&s)==0&&fstatfs(fd,&fs)==0&&regular(s)&&fs.f_type==0x01021994;::close(fd);
  return ok?true:failure(e,"payload must be owned regular tmpfs file");
 }
 static uint64_t available_kib(){std::ifstream f("/proc/meminfo");std::string k,u;uint64_t n;while(f>>k>>n>>u)if(k=="MemAvailable:")return n;return 0;}
public:
 Admission()=default;Admission(const Admission&)=delete;
 ~Admission(){if(lock_>=0){flock(lock_,LOCK_UN);::close(lock_);}}
 bool active()const{return active_;}bool reused()const{return reused_;}
 bool begin(const std::string&path,uint64_t hash,uint64_t payload,uint64_t allocation,std::string&e){
  const char*flag=std::getenv("STRATA_SHARED_ARENA_REUSE_V421");bool requested=flag&&std::string(flag)=="1";
  if(path.empty())return requested?failure(e,"mandatory shared path absent"):true;
  if(!requested)return failure(e,"shared path requires explicit sealed reuse; legacy overwrite prohibited");
  if(!hash||!payload||allocation<payload||allocation>(96ull<<30))return failure(e,"size/hash scope");
  active_=true;path_=path;seal_=path+".ready-v421";hash_=hash;payload_=payload;allocation_=allocation;
  std::ifstream boot("/proc/sys/kernel/random/boot_id");std::getline(boot,boot_);if(boot_.size()!=36)return failure(e,"boot identity unavailable");
  lock_=::open((path+".lock-v421").c_str(),O_CREAT|O_RDWR|O_CLOEXEC|O_NOFOLLOW,0600);if(lock_<0)return failure(e,"lock open");struct stat ls{};if(fstat(lock_,&ls)||!regular(ls)||flock(lock_,LOCK_EX|LOCK_NB))return failure(e,"another startup writer/attacher owns bank");
  struct stat ps{},ss{};bool payload_exists=lstat(path.c_str(),&ps)==0;bool seal_exists=lstat(seal_.c_str(),&ss)==0;
  if(!payload_exists&&!seal_exists){
   auto slash=path.find_last_of('/');std::string parent=slash==std::string::npos?".":path.substr(0,slash);struct statfs fs{};struct statvfs cap{};
   if(statfs(parent.c_str(),&fs)||fs.f_type!=0x01021994||statvfs(parent.c_str(),&cap))return failure(e,"fresh bank parent must be tmpfs");
   uint64_t free=uint64_t(cap.f_bavail)*uint64_t(cap.f_frsize);if(free<allocation+(4ull<<30)||available_kib()*1024<allocation+(32ull<<30))return failure(e,"fresh bank lacks independent RAM/tmpfs reserve");
   return true;
  }
  if(!payload_exists||!seal_exists)return failure(e,"incomplete existing bank; no overwrite or fallback load");
  if(!payload_stat(ps,e))return false;
  int fd=::open(seal_.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW);if(fd<0)return failure(e,"seal open");Seal s{};bool ok=fstat(fd,&ss)==0&&regular(ss)&&ss.st_size==sizeof(s)&&pread(fd,&s,sizeof(s),0)==sizeof(s);::close(fd);
  const char magic[]="SWIFT_SHARED_ARENA_V421";
  if(!ok||memcmp(s.magic,magic,sizeof(magic))||s.version!=1||s.pack_hash!=hash||s.payload!=payload||s.allocation!=allocation||s.device!=uint64_t(ps.st_dev)||s.inode!=uint64_t(ps.st_ino)||s.size!=uint64_t(ps.st_size)||s.size!=allocation+4096||s.uid!=geteuid()||s.mtime_sec!=ps.st_mtim.tv_sec||s.mtime_nsec!=ps.st_mtim.tv_nsec||std::string(s.boot,strnlen(s.boot,sizeof(s.boot)))!=boot_)return failure(e,"seal/source/boot/file identity changed");
  admitted_device_=ps.st_dev;admitted_inode_=ps.st_ino;reused_=true;return true;
 }
 bool commit(std::string&e){
  if(!active_)return true;
  struct stat ps{};if(!payload_stat(ps,e))return false;if(uint64_t(ps.st_size)!=allocation_+4096)return failure(e,"completed size mismatch");
  if(reused_&&(uint64_t(ps.st_dev)!=admitted_device_||uint64_t(ps.st_ino)!=admitted_inode_))return failure(e,"bank replaced during attach");
  // A supported RW CUDA registration can mark an unchanged shmem inode dirty.
  // Caller has positively completed registration, protected CPU payload read-only,
  // and still holds our exclusive startup lock; refresh only this admitted inode.
  Seal s{};const char magic[]="SWIFT_SHARED_ARENA_V421";memcpy(s.magic,magic,sizeof(magic));s.pack_hash=hash_;s.payload=payload_;s.allocation=allocation_;s.device=ps.st_dev;s.inode=ps.st_ino;s.size=ps.st_size;s.uid=geteuid();s.mtime_sec=ps.st_mtim.tv_sec;s.mtime_nsec=ps.st_mtim.tv_nsec;memcpy(s.boot,boot_.data(),boot_.size());
  std::string tmp=seal_+".tmp-"+std::to_string(getpid());int fd=::open(tmp.c_str(),O_CREAT|O_EXCL|O_WRONLY|O_CLOEXEC|O_NOFOLLOW,0600);if(fd<0)return failure(e,"seal creation");bool ok=write(fd,&s,sizeof(s))==sizeof(s)&&fsync(fd)==0;::close(fd);if(!ok||rename(tmp.c_str(),seal_.c_str())){unlink(tmp.c_str());return failure(e,"seal commit");}return true;
 }
};
}
