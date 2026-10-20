#pragma once
namespace lmtorch::dsl {
// True when argv[1] selects the DSL host (--dsl, --dsl-schema, --dsl-signatures) instead of the lmapi operations.
bool selects_dsl(int argc, char** argv);
// The `_dsl_` bytecode host: prints exactly one JSON result line; exit 0 ok, 2 usage, 3 load/run error, 4 application failure.
int run(int argc, char** argv);
}
