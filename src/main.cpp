#include "torchcli/ops.hpp"
#include <torch/torch.h>
int main(int argc,char**argv){
  at::set_num_threads(2);
  at::set_num_interop_threads(1);
  auto spec=make_torchcli_spec();
  return lmcpp::run_cli(spec,argc,argv);
}
