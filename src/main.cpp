#include "torchcli/ops.hpp"
#include "torchcli/dsl_host.hpp"
#include <torch/torch.h>
int main(int argc,char**argv){
  if(torchcli::dsl::selects_dsl(argc,argv))return torchcli::dsl::run(argc,argv);
  bool meta=argc>1&&(std::string(argv[1])=="--schema"||std::string(argv[1])=="--list"||std::string(argv[1])=="--help"||std::string(argv[1])=="--explain");
  if(!meta){at::set_num_threads(2);at::set_num_interop_threads(1);}
  auto spec=make_torchcli_spec();
  return lmcpp::run_cli(spec,argc,argv);
}
