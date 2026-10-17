#pragma once
#include <string>
namespace guild::runtime {
// Registry scope is explicit: compatible default candidates, not model bindings.
bool version_report(bool json, std::string& output, std::string& error);
}
