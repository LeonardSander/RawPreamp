#pragma once
#include <string>
namespace rawpreamp {
// Explicit diagnostic entry point; never runs during normal application use.
int runSpatialGainValidation(const std::string& reportPath);
int runRawDecodeValidation(const std::string& reportPath);
}
