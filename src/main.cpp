#include "lmtorch/ops.hpp"
#include "lmtorch/dsl_host.hpp"
#include <torch/torch.h>
#include <string>
#include <vector>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif
// Host flag standard (R42 in lmscript's dsl-bytecode requirements): --cli-schema prints the lmapi cli.schema.v1 document
// (also reachable as the lmapi protocol's --schema), --dsl-schema prints the _dsl_ schema.
int main(int argc,char**argv){
#ifdef _WIN32
  // The CRT's text mode would turn \n into \r\n: --dsl-schema --json and --schema are byte-exact, results are JSON lines.
  _setmode(_fileno(stdout),_O_BINARY);_setmode(_fileno(stderr),_O_BINARY);_setmode(_fileno(stdin),_O_BINARY);
#endif
  if(lmtorch::dsl::selects_dsl(argc,argv))return lmtorch::dsl::run(argc,argv);
  std::vector<char*> args(argv,argv+argc);
  static char schema_flag[]="--schema";
  if(argc>1&&std::string(argv[1])=="--cli-schema"){
    args[1]=schema_flag;
    if(argc==3&&std::string(argv[2])=="--json")args.resize(2);  // the document is JSON either way (symmetry with --dsl-schema --json)
  }
  bool meta=argc>1&&(std::string(args[1])=="--schema"||std::string(args[1])=="--list"||std::string(args[1])=="--help"||std::string(args[1])=="--explain");
  if(!meta){at::set_num_threads(2);at::set_num_interop_threads(1);}
  auto spec=make_lmtorch_spec();
  return lmcpp::run_cli(spec,static_cast<int>(args.size()),args.data());
}
