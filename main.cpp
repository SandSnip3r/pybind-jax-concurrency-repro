#include "common.hpp"
#include "jaxInterface.hpp"

#include <pybind11/embed.h>

#include <iostream>
#include <csignal>

namespace py = pybind11;
using namespace std;

void trainingLoop(JaxInterface &jaxInterface) {
  common::ModelInput input = {1.1, 2.2, 3.3, 4.4};
  common::ModelOutput output = {123.456};
  while (1) {
    jaxInterface.train(input, output);
  }
}

int main() {
  std::cout << "Initializing Python interpreter" << std::endl;
  py::scoped_interpreter interpreter;
  // Take back default signal handler from Python so we can kill our program with ctrl+C.
  std::signal(SIGINT, SIG_DFL);

  // Add the current source directory to the path so that we can later find our python module.
  py::module sys = py::module::import("sys");
  const std::string sourceDir = std::string(SOURCE_DIR);
  sys.attr("path").cast<py::list>().append(sourceDir);

  // Release the GIL so that other threads may grab it.
  pybind11::gil_scoped_release release;

  JaxInterface jaxInterface;
  jaxInterface.initialize();
  cout << "JaxInterface is initialized" << std::endl;

  // Run a tight training loop on another thread.
  thread thr(trainingLoop, std::ref(jaxInterface));
  
  // Run a tight inference loop on this thread.
  common::ModelInput input = {1.1, 2.2, 3.3, 4.4};
  while(1) {
    common::ModelOutput res = jaxInterface.inference(input);
  }

  thr.join();
  return 0;
}
