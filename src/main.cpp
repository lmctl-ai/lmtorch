#include "lmtorch/dsl_host.hpp"
#include <iostream>
#include <string>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

int main(int argc, char** argv) {
#ifdef _WIN32
  // Keep schema bytes and invocation frames LF-delimited on Windows.
  _setmode(_fileno(stdout), _O_BINARY);
  _setmode(_fileno(stderr), _O_BINARY);
  _setmode(_fileno(stdin), _O_BINARY);
#endif
  if (argc == 2 && std::string(argv[1]) == "--version") {
    std::cout << "lmtorch " LMTORCH_VERSION "\n";
    return 0;
  }
  return lmtorch::dsl::run(argc, argv);
}
