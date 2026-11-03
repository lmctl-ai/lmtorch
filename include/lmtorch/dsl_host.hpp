#pragma once
namespace lmtorch::dsl {
// The lmcode bytecode host: emits the SDK invocation return frame and never dispatches a legacy CLI surface.
int run(int argc, char** argv);
}
