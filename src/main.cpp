#include "lmtorch/ops.hpp"
#include "lmtorch/dsl_host.hpp"
#include <torch/torch.h>
int main(int argc,char**argv){
  if(lmtorch::dsl::selects_dsl(argc,argv))return lmtorch::dsl::run(argc,argv);
  bool meta=argc>1&&(std::string(argv[1])=="--schema"||std::string(argv[1])=="--list"||std::string(argv[1])=="--help"||std::string(argv[1])=="--explain");
  if(!meta){at::set_num_threads(2);at::set_num_interop_threads(1);}
  auto spec=make_lmtorch_spec();
  return lmcpp::run_cli(spec,argc,argv);
}
