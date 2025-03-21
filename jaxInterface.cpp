#include "jaxInterface.hpp"

#include <iostream>

namespace py = pybind11;

JaxInterface::~JaxInterface() {
  if (jaxModule_) {
    jaxModule_.reset();
  }
  if (randomModule_) {
    randomModule_.reset();
  }
  if (rngKey_) {
    rngKey_.reset();
  }
  if (model_) {
    model_.reset();
  }
  if (optimizerState_) {
    optimizerState_.reset();
  }
}

void JaxInterface::initialize() {
  constexpr int kSeed{0};
  constexpr float kLearningRate{1e-5};

  std::cout << "Constructing JaxInterface" << std::endl;
  
  // Grab the GIL since we'll be doing lots of things in Python.
  py::gil_scoped_acquire acquire;

  // Load modules.
  jaxModule_ = py::module::import("my_module");
  randomModule_ = py::module::import("jax.random");

  // Initialize random numbers/stream.
  rngKey_ = randomModule_->attr("key")(kSeed);
  py::module nnxModule = py::module::import("flax.nnx");
  py::object nnxRngs = nnxModule.attr("Rngs")(getNextRngKey());

  // Construct model.
  py::object ModelType = jaxModule_->attr("Model");
  model_ = ModelType(std::tuple_size_v<common::ModelInput>, std::tuple_size_v<common::ModelOutput>, nnxRngs);

  // Construct optimizer.
  py::module optaxModule = py::module::import("optax");
  py::object adam = optaxModule.attr("adam")(kLearningRate);
  optimizerState_ = nnxModule.attr("Optimizer")(*model_, adam);
}

common::ModelOutput JaxInterface::inference(const common::ModelInput &input) {
  py::gil_scoped_acquire acquire;
  py::array_t<float> numpyResult = jaxModule_->attr("inference")(*model_, toNumpy(input));
  common::ModelOutput result;
  std::memcpy(result.data(), numpyResult.data(), result.size());
  return result;
}

void JaxInterface::train(const common::ModelInput &input, const common::ModelOutput &target) {
  py::gil_scoped_acquire acquire;
  jaxModule_->attr("train")(*model_, toNumpy(input), toNumpy(target), *optimizerState_);
}

py::object JaxInterface::getNextRngKey() {
  py::tuple keys = randomModule_->attr("split")(*rngKey_);
  if (keys.size() != 2) {
    throw std::runtime_error("Splitting key gave a weird tuple size");
  }
  rngKey_ = keys[0];
  return keys[1];
}