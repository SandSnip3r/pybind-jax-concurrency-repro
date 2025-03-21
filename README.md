# Multithreaded Pybind11 JAX Reproducer

This project stands as a small reproducer of what seems to be a bug.

## System Explanation

This project is a C++ project which uses Pybind11 to call into Python/JAX. It does so with 2 threads. The two threads share python objects which should be protected by the GIL. One thread does inference on a deep neural network. The other thread trains the same deep neural network.

## Repro Steps

Create a directory where you'd like to test this. After you're done testing, you will be able to delete the directory and your machine will be exactly as it was before.

```
python -m venv repro
source repro/bin/activate
git clone https://github.com/SandSnip3r/pybind-jax-concurrency-repro.git
cd pybind-jax-concurrency-repro
mkdir build
cmake -B build
cmake --build build/
pip install -r requirements.txt
./build/pybind_jax_repro
```

## Error

```
Initializing Python interpreter
Constructing JaxInterface
JaxInterface is initialized
Enter training loop
terminate called after throwing an instance of 'pybind11::error_already_set'
  what():  AttributeError[WITH __notes__]: 'Model' object has no attribute 'linear1'
__notes__ (len=1):
--------------------
For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set JAX_TRACEBACK_FILTERING=off to include these.

At:
  /path/to/project/pybind-jax-concurrency-repro/my_module.py(13): __call__
  /path/to/project/pybind-jax-concurrency-repro/my_module.py(20): inference
  /path/to/venv/repro/lib/python3.12/site-packages/flax/nnx/transforms/compilation.py(129): __call__
  /path/to/venv/repro/lib/python3.12/site-packages/jax/_src/linear_util.py(388): _get_result_paths_thunk
  /path/to/venv/repro/lib/python3.12/site-packages/jax/_src/api_util.py(73): flatten_fun
  /path/to/venv/repro/lib/python3.12/site-packages/jax/_src/api_util.py(284): _argnums_partial
  /path/to/venv/repro/lib/python3.12/site-packages/jax/_src/linear_util.py(210): call_wrapped
  /path/to/venv/repro/lib/python3.12/site-packages/jax/_src/interpreters/partial_eval.py(2181): trace_to_jaxpr_dynamic
  /path/to/venv/repro/lib/python3.12/site-packages/jax/_src/profiler.py(334): wrapper
  /path/to/venv/repro/lib/python3.12/site-packages/jax/_src/pjit.py(1289): _create_pjit_jaxpr
  /path/to/venv/repro/lib/python3.12/site-packages/jax/_src/linear_util.py(460): memoized_fun
  /path/to/venv/repro/lib/python3.12/site-packages/jax/_src/pjit.py(618): _infer_params_impl
  /path/to/venv/repro/lib/python3.12/site-packages/jax/_src/pjit.py(718): _infer_params_internal
  /path/to/venv/repro/lib/python3.12/site-packages/jax/_src/pjit.py(695): _infer_params
  /path/to/venv/repro/lib/python3.12/site-packages/jax/_src/pjit.py(179): _python_pjit_helper
  /path/to/venv/repro/lib/python3.12/site-packages/jax/_src/pjit.py(339): cache_miss
  /path/to/venv/repro/lib/python3.12/site-packages/jax/_src/traceback_util.py(228): reraise_with_filtered_traceback
  /path/to/venv/repro/lib/python3.12/site-packages/flax/nnx/transforms/compilation.py(350): jit_wrapper

Aborted (core dumped)
```

## Tested Environments

- Ubuntu