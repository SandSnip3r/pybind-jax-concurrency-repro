#ifndef JAX_INTERFACE_HPP_
#define JAX_INTERFACE_HPP_

#include "common.hpp"

#include <pybind11/embed.h>
#include <pybind11/numpy.h>

#include <optional>

// JaxInterface is a class which abstracts away everything we need JAX for. The model weights & optimizer state are held in this class. Calls to inference and training use the stored model.
class JaxInterface {
public:
  ~JaxInterface();

  // Load Python modules, create model, create optimizer.
  void initialize();

  // Pass the input through a deep neural network and get the result.
  common::ModelOutput inference(const common::ModelInput &input);

  // Train the model to move towards the target.
  void train(const common::ModelInput &input, const common::ModelOutput &target);
private:
  std::optional<pybind11::module> jaxModule_;
  std::optional<pybind11::module> randomModule_;
  std::optional<pybind11::object> rngKey_;
  std::optional<pybind11::object> model_;
  std::optional<pybind11::object> optimizerState_;

  pybind11::object getNextRngKey();

  // Converts from either the model input or model output to a numpy array.
  // Note: Expects caller to be holding the GIL.
  template<typename CppType>
  pybind11::array_t<float> toNumpy(const CppType &cppData) {
    pybind11::array_t<float> numpyArray(std::tuple_size_v<CppType>);
    float *mutableArray = numpyArray.mutable_unchecked<1>().mutable_data(0);
    std::memcpy(mutableArray, cppData.data(), cppData.size());
    return numpyArray;
  }
};

#endif // JAX_INTERFACE_HPP_