#include "torchcli/ops.hpp"
#include "torchcli/safetensors.hpp"
#include <torch/torch.h>
#include <cmath>
#include <cstring>
#include <limits>

namespace {
using namespace lmcpp;
torch::ScalarType scalar_type(const std::string&d){if(d=="F32")return torch::kFloat32;if(d=="F64")return torch::kFloat64;if(d=="I64")return torch::kInt64;throw Error{ErrorCode::ValidationError,"dtype must be F32, F64, or I64",{{"tool_code","INVALID_DTYPE"}}};}
std::string dtype_name(torch::ScalarType t){if(t==torch::kFloat32)return "F32";if(t==torch::kFloat64)return "F64";if(t==torch::kInt64)return "I64";throw Error{ErrorCode::ValidationError,"unsupported tensor dtype",{{"tool_code","UNSUPPORTED_DTYPE"}}};}
torch::Tensor tensor_from(const torchcli::TensorData&d){auto t=torch::empty(d.shape,torch::TensorOptions().dtype(scalar_type(d.dtype)).device(torch::kCPU));if(t.nbytes()!=d.bytes.size())throw Error{ErrorCode::CommandFailed,"tensor payload has invalid size",{{"tool_code","INVALID_ARTIFACT"}}};if(!d.bytes.empty())std::memcpy(t.data_ptr(),d.bytes.data(),d.bytes.size());return t;}
torchcli::TensorData tensor_to(const torch::Tensor&input){auto t=input.contiguous().cpu();torchcli::TensorData d;d.dtype=dtype_name(t.scalar_type());d.shape.assign(t.sizes().begin(),t.sizes().end());d.bytes.resize(t.nbytes());if(!d.bytes.empty())std::memcpy(d.bytes.data(),t.data_ptr(),d.bytes.size());return d;}
torchcli::TensorRef ref(const Json&j,const char*k){return torchcli::parse_ref(j.at(k).get<std::string>());}
torchcli::TensorRef out_ref(const Json&j){auto path=j.at("out").get<std::string>();if(path.find('#')!=std::string::npos)throw Error{ErrorCode::ValidationError,"output path cannot contain #",{{"tool_code","INVALID_OUTPUT_PATH"}}};return{path,"tensor"};}
Param param(std::string n,Type t,bool req=true,std::string occurs="1",Json def=nullptr,std::string desc={}){Param p;p.name=std::move(n);p.type=t;p.required=req;p.occurs=std::move(occurs);p.default_value=std::move(def);p.description=std::move(desc);return p;}
Json descriptor_plus(const torchcli::TensorRef&r,const torchcli::TensorData&d){return torchcli::descriptor(r,d);}
}
lmcpp::CliSpec make_torchcli_spec(){
  using namespace lmcpp;CliSpec spec{"torchcli","0.1.0",{}};
  Operation create;create.name="tensor_create";create.access=Access::Write;create.description="Create a dense CPU tensor artifact";create.returns="object";
  auto shape=param("shape",Type::Number,true,"+");shape.description="Repeated positive integer dimensions";
  auto dtype=param("dtype",Type::String,false,"?","F32");dtype.enum_values={"F32","F64","I64"};
  auto fill=param("fill",Type::Number,false,"?",0);
  create.params={shape,dtype,fill,param("out",Type::String)};
  create.handler=[](const Params&p){std::string dtype=p.value("dtype","F32");auto st=scalar_type(dtype);uint64_t elem=st==torch::kFloat32?4:8;std::vector<int64_t>dims;uint64_t count=1;for(const auto&v:p.at("shape")){double x=v.get<double>();if(!std::isfinite(x)||x<1||std::floor(x)!=x||x>static_cast<double>(std::numeric_limits<int64_t>::max()))throw Error{ErrorCode::ValidationError,"shape dimensions must be positive integers",{{"tool_code","INVALID_SHAPE"}}};auto d=static_cast<uint64_t>(x);if(count>256ull*1024*1024/elem/d)throw Error{ErrorCode::ValidationError,"tensor exceeds artifact size limit",{{"tool_code","OUTPUT_TOO_LARGE"}}};count*=d;dims.push_back(static_cast<int64_t>(d));}if(dims.empty())throw Error{ErrorCode::ValidationError,"shape must not be empty",{{"tool_code","INVALID_SHAPE"}}};double value=p.value("fill",0.0);if(!std::isfinite(value))throw Error{ErrorCode::ValidationError,"fill must be finite",{{"tool_code","INVALID_VALUE"}}};if(st==torch::kInt64&&(std::floor(value)!=value||value<static_cast<double>(std::numeric_limits<int64_t>::min())||value>=static_cast<double>(std::numeric_limits<int64_t>::max())))throw Error{ErrorCode::ValidationError,"I64 fill must be an in-range integer",{{"tool_code","INVALID_VALUE"}}};if(st==torch::kFloat32&&std::abs(value)>std::numeric_limits<float>::max())throw Error{ErrorCode::ValidationError,"F32 fill is out of range",{{"tool_code","INVALID_VALUE"}}};auto t=torch::full(dims,value,torch::TensorOptions().dtype(st).device(torch::kCPU));auto data=tensor_to(t);auto r=out_ref(p);torchcli::write_tensor(r.path,data);return descriptor_plus(r,data);};
  Operation matmul;matmul.name="tensor_matmul";matmul.access=Access::Write;matmul.description="Multiply two rank-2 dense CPU tensors";matmul.returns="object";matmul.params={param("a",Type::String),param("b",Type::String),param("out",Type::String)};
  matmul.handler=[](const Params&p){auto ar=ref(p,"a"),br=ref(p,"b"),or_=out_ref(p);auto ad=torchcli::read_tensor(ar),bd=torchcli::read_tensor(br);if(ad.dtype!=bd.dtype)throw Error{ErrorCode::ValidationError,"matmul inputs must have matching dtypes",{{"tool_code","DTYPE_MISMATCH"}}};if(ad.shape.size()!=2||bd.shape.size()!=2||ad.shape[1]!=bd.shape[0])throw Error{ErrorCode::ValidationError,"matmul requires compatible rank-2 shapes",{{"tool_code","SHAPE_MISMATCH"}}};auto result=tensor_from(ad).matmul(tensor_from(bd));auto data=tensor_to(result);torchcli::write_tensor(or_.path,data);return descriptor_plus(or_,data);};
  Operation inspect;inspect.name="tensor_inspect";inspect.access=Access::Read;inspect.description="Inspect tensor metadata and a bounded value preview";inspect.returns="object";inspect.params={param("input",Type::String),param("limit",Type::Number,false,"?",32)};
  inspect.handler=[](const Params&p){auto r=ref(p,"input");auto d=torchcli::read_tensor(r);auto t=tensor_from(d);double lim=p.value("limit",32.0);if(!std::isfinite(lim)||lim<0||std::floor(lim)!=lim||lim>32)throw Error{ErrorCode::ValidationError,"limit must be an integer from 0 through 32",{{"tool_code","INVALID_LIMIT"}}};Json values=Json::array();uint64_t nonfinite=0;uint64_t n=static_cast<uint64_t>(t.numel()),take=std::min<uint64_t>(n,static_cast<uint64_t>(lim));for(uint64_t i=0;i<take;++i){if(t.scalar_type()==torch::kInt64)values.push_back(t.data_ptr<int64_t>()[i]);else{double x=t.scalar_type()==torch::kFloat32?static_cast<double>(t.data_ptr<float>()[i]):t.data_ptr<double>()[i];if(std::isfinite(x))values.push_back(x);else{values.push_back(nullptr);++nonfinite;}}}auto result=descriptor_plus(r,d);result["values"]=values;result["nonfinite_count"]=nonfinite;return result;};
  spec.operations={create,matmul,inspect};return spec;
}
