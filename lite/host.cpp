#include <iostream>

#include "bindings.hpp"
#include "schema_embedded.hpp"
#include "stdlib/invocation.hpp"
using namespace lmtorch_lite;
static std::string schema() { return std::string(reinterpret_cast<const char*>(lite_schema), sizeof(lite_schema)); }
static int run(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--version") {
    std::cout << "lmtorch lite 0.1.0\n";
    return 0;
  }
  if (argc >= 2 && (std::string(argv[1]) == "--lmcode-schema" || std::string(argv[1]) == "--cli-schema")) {
    std::cout << schema();
    return 0;
  }
  Host host;
  dslvm::Registry registry;
  register_bindings(registry, host, schema());
  if (argc == 2 && std::string(argv[1]) == "--lmcode-signatures") {
    for (auto& s : registry.signatures()) std::cout << s << '\n';
    return 0;
  }
  std::vector<std::string> files;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--lmcode") continue;
    files.push_back(argv[i]);
  }
  require(!files.empty(), "usage: lmtorch --lmcode FILE...");
  auto results = dslvm::InvocationResults::from_command(files);
  auto contract = dslvm::ResultContract::from_schema(schema());
  std::vector<dslvm::Program> programs;
  for (size_t i = 0; i < files.size(); ++i) {
    try {
      programs.push_back(dslvm::Program::load(dslvm::read_file(files[i]), schema(), registry));
    } catch (const dslvm::Error& e) {
      results.fail(results.registrations()[i].index, "load", e);
      return results.write_frame(std::cout);
    }
  }
  dslvm::Context context;
  context.set("torch", &host);
  context.set("out", &host);
  context.set_dry(results.dry());
  for (size_t i = 0; i < programs.size(); ++i) {
    auto index = results.registrations()[i].index;
    auto& result = results.select(index, registry, contract);
    host.output = Json::object();
    grad_enabled = true;
    try {
      programs[i].run(context, registry);
      std::string encoded = host.output.dump();
      dslvm::stdlib::json_detail::Parser parser(encoded);
      auto node = parser.val();
      result.set(Value::object("Json", &node));
      results.complete(index);
    } catch (const dslvm::Error& e) {
      results.fail(index, e.code == "APP_FAIL" ? "app" : "run", e);
      break;
    } catch (const std::exception& e) {
      results.fail(index, "run", dslvm::Error("RUN_HOST", e.what()));
      break;
    }
  }
  return results.write_frame(std::cout);
}
int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const dslvm::Error& e) {
    return dslvm::failure_frame("cli", e);
  } catch (const std::exception& e) {
    return dslvm::failure_frame("cli", dslvm::Error("RUN_HOST", e.what()));
  }
}
