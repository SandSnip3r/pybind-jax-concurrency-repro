from flax import nnx
import jax
import jax.numpy as jnp

class Model(nnx.Module):
  def __init__(self, inSize: int, outSize: int, rngs: nnx.Rngs):
    intermediateSize = 64
    key = rngs.params()
    self.linear1 = nnx.Linear(inSize, intermediateSize, rngs=rngs)
    self.linear2 = nnx.Linear(intermediateSize, outSize, rngs=rngs)

  def __call__(self, x):
    x = self.linear1(x)
    x = jax.nn.relu(x)
    x = self.linear2(x)
    return x

@nnx.jit
def inference(model, input):
  return model(input)

@nnx.jit
def train(model, input, target, optimizerState):
  def lossFunction(model, input, target):
    result = model(input)
    return jnp.mean(jnp.square(result-target))

  gradients = nnx.grad(lossFunction)(model, input, target)
  optimizerState.update(gradients)