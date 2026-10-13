#include "torchcli/ops.hpp"
#include <torch/torch.h>
int main(int argc,char**argv){
  bool meta=argc>1&&(std::string(argv[1])=="--schema"||std::string(argv[1])=="--list"||std::string(argv[1])=="--help"||std::string(argv[1])=="--explain");
  if(!meta){at::set_num_threads(2);at::set_num_interop_threads(1);}
  auto spec=make_torchcli_spec();
  return lmcpp::run_cli(spec,argc,argv);
}
