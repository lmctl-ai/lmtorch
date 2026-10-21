#include "lmtorch/safetensors.hpp"
#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#ifdef _WIN32
#include <process.h>
#define getpid _getpid
#else
#include <sys/stat.h>
#include <unistd.h>
#endif
using namespace lmtorch;
// A fresh, not yet existing path in the temporary directory (the working directory during the test, because artifact
// paths must stay inside it); portable replacement for mkstemp + unlink.
static std::string fresh(const std::string&name){static std::mt19937_64 rng(static_cast<uint64_t>(getpid())*1000003u+static_cast<uint64_t>(std::random_device{}()));for(;;){auto p=(std::filesystem::temp_directory_path()/(name+"-"+std::to_string(rng()%1000000000))).string();if(!std::filesystem::exists(p))return p;}}
static std::string artifact(const std::string&name,const std::string&header,const std::string&data={}){std::string p=fresh(name);std::ofstream f(p,std::ios::binary|std::ios::trunc);uint64_t n=header.size();for(int i=0;i<8;++i)f.put(static_cast<char>((n>>(i*8))&255));f<<header<<data;return p;}
template<class F> static bool throws_tool(F&&f,const std::string&code){try{f();}catch(const lmcpp::Error&e){return code.empty()?e.code==lmcpp::ErrorCode::Conflict:e.data.value("tool_code",std::string())==code;}return false;}
int main(){try { std::filesystem::current_path(std::filesystem::temp_directory_path());std::string tmp=fresh("lmtorch-roundtrip");TensorData t{"F32",{2,2},{}};float values[]={1,2,3,4};t.bytes.resize(sizeof(values));std::memcpy(t.bytes.data(),values,sizeof(values));write_tensor(tmp,t);
#ifndef _WIN32
  struct stat st{};assert(stat(tmp.c_str(),&st)==0&&(st.st_mode&0777)==0644);
#endif
  auto r=read_tensor({tmp,"tensor"});assert(r.dtype=="F32"&&r.shape==std::vector<int64_t>({2,2})&&r.bytes==t.bytes);assert(throws_tool([&]{write_tensor(tmp,t);},""));auto d=descriptor({tmp,"tensor"},r);assert(d["nbytes"]==16);std::filesystem::remove(tmp);
#ifndef _WIN32
  std::string private_mode=fresh("lmtorch-mode");mode_t previous=umask(0077);write_tensor(private_mode,t);umask(previous);assert(stat(private_mode.c_str(),&st)==0&&(st.st_mode&0777)==0600);std::filesystem::remove(private_mode);
#endif
  TensorData second{"I64",{2},{}};int64_t numbers[]={7,9};second.bytes.resize(sizeof(numbers));std::memcpy(second.bytes.data(),numbers,sizeof(numbers));TensorBundle bundle;bundle.tensors={{"alpha",t},{"beta",second}};bundle.metadata={{"torchcli_format","1"},{"arch","mlp_relu_v1"},{"step","3"}};std::string multi=fresh("lmtorch-bundle");write_bundle(multi,bundle);auto br=read_bundle(multi);assert(br.tensors.size()==2&&br.tensors.at("alpha").bytes==t.bytes&&br.tensors.at("beta").bytes==second.bytes);assert(br.metadata==bundle.metadata);std::filesystem::remove(multi);
  double dv[]={1.25,-2.5};TensorData f64{"F64",{2},{}};f64.bytes.resize(sizeof(dv));std::memcpy(f64.bytes.data(),dv,sizeof(dv));auto p64=fresh("lmtorch-f64");write_tensor(p64,f64);assert(read_tensor({p64,"tensor"}).bytes==f64.bytes);std::filesystem::remove(p64);
  int64_t iv[]={-4,922337};TensorData i64{"I64",{2},{}};i64.bytes.resize(sizeof(iv));std::memcpy(i64.bytes.data(),iv,sizeof(iv));auto p64i=fresh("lmtorch-i64");write_tensor(p64i,i64);assert(read_tensor({p64i,"tensor"}).bytes==i64.bytes);std::filesystem::remove(p64i);
  auto trunc=artifact("lmtorch-truncated",R"({"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})","12");assert(throws_tool([&]{read_tensor({trunc,"tensor"});},"INVALID_OFFSETS"));std::filesystem::remove(trunc);
  auto bad=artifact("lmtorch-offset",R"({"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,8]}})","12345678");assert(throws_tool([&]{read_tensor({bad,"tensor"});},"INVALID_OFFSETS"));std::filesystem::remove(bad);
  std::string duplicate=R"({"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})";auto dup=artifact("lmtorch-duplicate",duplicate,"1234");assert(throws_tool([&]{read_tensor({dup,"tensor"});},"DUPLICATE_KEY"));std::filesystem::remove(dup);
  std::string huge((1024*1024)+1,' ');auto over=artifact("lmtorch-header-overflow",huge);assert(throws_tool([&]{read_tensor({over,"tensor"});},"INVALID_HEADER"));std::filesystem::remove(over);
  auto malformed=artifact("lmtorch-short-offset",R"({"tensor":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}})","1234");assert(throws_tool([&]{read_tensor({malformed,"tensor"});},"INVALID_OFFSETS"));std::filesystem::remove(malformed);
  auto overlap=artifact("lmtorch-overlap",R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}})","123456789012");assert(throws_tool([&]{read_tensor({overlap,"a"});},"INVALID_OFFSETS"));std::filesystem::remove(overlap);
  auto floatdim=artifact("lmtorch-floatdim",R"({"tensor":{"dtype":"F32","shape":[1.0],"data_offsets":[0,4]}})","1234");assert(throws_tool([&]{read_tensor({floatdim,"tensor"});},"INVALID_HEADER"));std::filesystem::remove(floatdim);
  auto floatoff=artifact("lmtorch-floatoff",R"({"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0.0,4]}})","1234");assert(throws_tool([&]{read_tensor({floatoff,"tensor"});},"INVALID_HEADER"));std::filesystem::remove(floatoff);
  auto bigshape=artifact("lmtorch-bigshape",R"({"tensor":{"dtype":"F32","shape":[9223372036854775808],"data_offsets":[0,4]}})","1234");assert(throws_tool([&]{read_tensor({bigshape,"tensor"});},"INVALID_HEADER"));std::filesystem::remove(bigshape);
  auto trailing=artifact("lmtorch-trailing",R"({"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})","12345678");assert(throws_tool([&]{read_tensor({trailing,"tensor"});},"INVALID_OFFSETS"));std::filesystem::remove(trailing);
  std::string boolbytes("\x00\x01\x02",3);auto badbool=artifact("lmtorch-badbool",R"({"tensor":{"dtype":"BOOL","shape":[3],"data_offsets":[0,3]}})",boolbytes);assert(throws_tool([&]{read_tensor({badbool,"tensor"});},"INVALID_ARTIFACT"));std::filesystem::remove(badbool);
  auto goodbool=artifact("lmtorch-goodbool",R"({"tensor":{"dtype":"BOOL","shape":[2],"data_offsets":[0,2]}})",std::string("\x00\x01",2));assert(read_tensor({goodbool,"tensor"}).bytes==std::vector<uint8_t>({0,1}));std::filesystem::remove(goodbool);
  } catch (const lmcpp::Error& e) { fprintf(stderr, "lmcpp error: %s data=%s\n", e.message.c_str(), e.data.dump().c_str()); return 1; }
}
