#include "torchcli/safetensors.hpp"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <map>
#include <set>
#include <sys/stat.h>
#include <unistd.h>

namespace torchcli {
using lmcpp::Error;
using lmcpp::ErrorCode;
using Json = lmcpp::Json;
namespace {
constexpr uint64_t kMaxHeader = 1024 * 1024;
constexpr uint64_t kMaxArtifact = 256 * 1024 * 1024;
[[noreturn]] void bad(std::string message, std::string code="INVALID_ARTIFACT") {
  throw Error{ErrorCode::CommandFailed, std::move(message), {{"tool_code", code}}};
}
uint64_t element_size(const std::string& dtype) {
  if (dtype=="F32") return 4;
  if (dtype=="F64" || dtype=="I64") return 8;
  bad("unsupported safetensors dtype: "+dtype, "UNSUPPORTED_DTYPE");
}
uint64_t count_elements(const std::vector<int64_t>& shape) {
  if(shape.size()>32) bad("tensor rank exceeds 32", "INVALID_SHAPE");
  uint64_t n=1;
  for(int64_t d:shape){
    if(d<0)bad("negative tensor dimension", "INVALID_SHAPE");
    if(d && n>kMaxArtifact/static_cast<uint64_t>(d))bad("tensor size exceeds artifact limit", "OUTPUT_TOO_LARGE");
    n*=static_cast<uint64_t>(d);
  }
  return n;
}
uint64_t little_u64(const uint8_t*p){uint64_t n=0;for(int i=7;i>=0;--i)n=(n<<8)|p[i];return n;}
void put_u64(std::string&out,uint64_t n){for(int i=0;i<8;++i)out.push_back(static_cast<char>((n>>(i*8))&0xff));}
void write_all(int fd,const char*p,size_t n){while(n){ssize_t w=::write(fd,p,n);if(w<0){if(errno==EINTR)continue;throw Error{ErrorCode::CommandFailed,"failed writing safetensors file",{{"tool_code","OUTPUT_WRITE_FAILED"}}};}p+=w;n-=static_cast<size_t>(w);}}
}
TensorRef parse_ref(const std::string&ref){auto h=ref.find('#');if(h==std::string::npos||h==0||h+1==ref.size()||ref.find('#',h+1)!=std::string::npos)throw Error{ErrorCode::ValidationError,"tensor reference must be path#key",{{"tool_code","INVALID_TENSOR_REF"}}};return{ref.substr(0,h),ref.substr(h+1)};}
TensorBundle read_bundle(const std::string&path){
  std::ifstream f(path,std::ios::binary);if(!f)throw Error{ErrorCode::NotFound,"cannot open tensor artifact: "+path,{{"tool_code","ARTIFACT_NOT_FOUND"}}};
  f.seekg(0,std::ios::end);auto end=f.tellg();if(end<8||static_cast<uint64_t>(end)>kMaxArtifact)bad("invalid or oversized artifact", "INVALID_ARTIFACT");f.seekg(0);
  std::vector<uint8_t> file(static_cast<size_t>(end));f.read(reinterpret_cast<char*>(file.data()),file.size());if(!f)bad("truncated artifact", "INVALID_ARTIFACT");
  uint64_t hlen=little_u64(file.data());if(hlen==0||hlen>kMaxHeader||hlen>file.size()-8)bad("invalid safetensors header length", "INVALID_HEADER");
  std::string header(reinterpret_cast<char*>(file.data()+8),static_cast<size_t>(hlen));bool duplicate=false;std::vector<std::set<std::string>> keys;
  auto callback=[&](int,Json::parse_event_t event,Json&parsed){if(event==Json::parse_event_t::object_start)keys.emplace_back();else if(event==Json::parse_event_t::key){std::string k=parsed.get<std::string>();if(keys.empty()||!keys.back().insert(k).second)duplicate=true;}else if(event==Json::parse_event_t::object_end&&!keys.empty())keys.pop_back();return true;};
  Json doc;try{doc=Json::parse(header,callback);}catch(...){bad("malformed safetensors JSON header", "INVALID_HEADER");}if(duplicate)bad("duplicate key in safetensors header", "DUPLICATE_KEY");if(!doc.is_object())bad("safetensors header must be an object", "INVALID_HEADER");
  struct Entry{std::string name,dtype;std::vector<int64_t>shape;uint64_t begin,end;};std::vector<Entry> entries;uint64_t data_size=file.size()-8-hlen;TensorBundle result;
  for(auto it=doc.begin();it!=doc.end();++it){
    if(it.key()=="__metadata__"){
      if(!it.value().is_object())bad("invalid safetensors metadata", "INVALID_HEADER");
      for(auto m=it.value().begin();m!=it.value().end();++m){if(!m.value().is_string())bad("safetensors metadata values must be strings", "INVALID_HEADER");result.metadata[m.key()]=m.value().get<std::string>();}
      continue;
    }
    const auto&v=it.value();if(!v.is_object()||!v.contains("dtype")||!v["dtype"].is_string()||!v.contains("shape")||!v["shape"].is_array()||!v.contains("data_offsets")||!v["data_offsets"].is_array()||v["data_offsets"].size()!=2)bad("invalid tensor header entry", "INVALID_HEADER");
    Entry e;e.name=it.key();e.dtype=v["dtype"].get<std::string>();(void)element_size(e.dtype);
    try{for(auto&d:v["shape"])e.shape.push_back(d.get<int64_t>());e.begin=v["data_offsets"][0].get<uint64_t>();e.end=v["data_offsets"][1].get<uint64_t>();}catch(...){bad("invalid tensor shape or offsets", "INVALID_HEADER");}
    uint64_t count=count_elements(e.shape), unit=element_size(e.dtype);if(count>kMaxArtifact/unit)bad("tensor exceeds artifact size limit", "OUTPUT_TOO_LARGE");uint64_t bytes=count*unit;
    if(e.begin>e.end||e.end>data_size||e.end-e.begin!=bytes) bad("tensor offsets do not match shape and dtype", "INVALID_OFFSETS");
    entries.push_back(std::move(e));
  }
  if(entries.empty()) bad("safetensors artifact has no tensors", "INVALID_HEADER");
  std::sort(entries.begin(),entries.end(),[](auto&a,auto&b){return a.begin<b.begin;});uint64_t cursor=0;
  for(auto&e:entries){if(e.begin!=cursor)bad("safetensors offsets overlap or leave gaps", "INVALID_OFFSETS");cursor=e.end;TensorData tensor;tensor.dtype=e.dtype;tensor.shape=e.shape;auto begin=file.begin()+static_cast<std::ptrdiff_t>(8+hlen+e.begin);tensor.bytes.assign(begin,begin+static_cast<std::ptrdiff_t>(e.end-e.begin));result.tensors.emplace(e.name,std::move(tensor));}
  if(cursor!=data_size) bad("safetensors data section is not fully covered", "INVALID_OFFSETS");
  return result;
}
TensorData read_tensor(const TensorRef&ref){auto bundle=read_bundle(ref.path);auto it=bundle.tensors.find(ref.key);if(it==bundle.tensors.end())throw Error{ErrorCode::NotFound,"tensor key not found: "+ref.key,{{"tool_code","TENSOR_NOT_FOUND"}}};return it->second;}
std::string write_bundle(const std::string&path,const TensorBundle&bundle){
  if(bundle.tensors.empty())bad("cannot write an empty safetensors artifact", "INVALID_TENSOR");
  Json hdr=Json::object();uint64_t offset=0;std::vector<uint8_t> payload;
  for(const auto&entry:bundle.tensors){const auto&t=entry.second;if(entry.first.empty()||entry.first=="__metadata__")bad("invalid safetensors tensor key", "INVALID_TENSOR");uint64_t count=count_elements(t.shape),unit=element_size(t.dtype);if(count>kMaxArtifact/unit)bad("tensor exceeds artifact size limit", "OUTPUT_TOO_LARGE");uint64_t size=count*unit;if(size!=t.bytes.size()||offset>kMaxArtifact-size)bad("tensor bytes do not match dtype and shape", "INVALID_TENSOR");hdr[entry.first]={{"dtype",t.dtype},{"shape",t.shape},{"data_offsets",{offset,offset+size}}};payload.insert(payload.end(),t.bytes.begin(),t.bytes.end());offset+=size;}
  if(!bundle.metadata.empty()){hdr["__metadata__"]=Json::object();for(const auto&item:bundle.metadata)hdr["__metadata__"][item.first]=item.second;}
  std::string h=hdr.dump();size_t padded=(h.size()+7)&~size_t(7);h.resize(padded,' ');if(h.size()>kMaxHeader||h.size()+8+payload.size()>kMaxArtifact)bad("output artifact exceeds size limit", "OUTPUT_TOO_LARGE");
  std::string tmp=path+".tmpXXXXXX";std::vector<char>name(tmp.begin(),tmp.end());name.push_back('\0');int fd=::mkstemp(name.data());if(fd<0)throw Error{ErrorCode::CommandFailed,"cannot create output temporary file",{{"tool_code","OUTPUT_WRITE_FAILED"}}};std::string tempfile(name.data());
  try{std::string prefix;put_u64(prefix,h.size());write_all(fd,prefix.data(),prefix.size());write_all(fd,h.data(),h.size());if(!payload.empty())write_all(fd,reinterpret_cast<const char*>(payload.data()),payload.size());if(::fsync(fd)!=0)throw std::runtime_error("fsync failed");if(::close(fd)!=0){fd=-1;throw std::runtime_error("close failed");}fd=-1;if(::link(tempfile.c_str(),path.c_str())!=0){if(errno==EEXIST)throw Error{ErrorCode::Conflict,"output already exists: "+path,{{"path",path}}};throw Error{ErrorCode::CommandFailed,"cannot install output artifact: "+path,{{"tool_code","OUTPUT_WRITE_FAILED"}}};}::unlink(tempfile.c_str());}catch(...){if(fd>=0)::close(fd);::unlink(tempfile.c_str());throw;}return path;
}
std::string write_tensor(const std::string&path,const TensorData&t){TensorBundle b;b.tensors.emplace("tensor",t);return write_bundle(path,b);}
lmcpp::Json descriptor(const TensorRef&ref,const TensorData&t){return{{"ref",ref.path+"#"+ref.key},{"path",ref.path},{"key",ref.key},{"dtype",t.dtype},{"shape",t.shape},{"nbytes",t.bytes.size()}};}
}
