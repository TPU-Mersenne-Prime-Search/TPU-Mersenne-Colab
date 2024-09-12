import os
os.environ["XLA_FLAGS"] = '--xla_gpu_cuda_data_dir=/home/I/.guix-profile/bin/ --xla_force_host_platform_device_count=8'
import jax
jax.config.update("jax_enable_x64", True)
from jax import lax
import einops
import jax.numpy as jnp
import functools
from jax.experimental.shard_map import shard_map
import collections


class DFT:
  def __init__(self, n, inverse=False):
    self.n = n
    self.n_devices = len(jax.devices())
    self.partition = [self.n_devices, self.n_devices]
    dft_matrix     = self.mk_dft_matrix(n, inverse=inverse)
    self.shards = einops.rearrange(
      dft_matrix,
      '(m1 m) (n1 n) -> m1 n1 m n',
      m1=self.n_devices,
      n1=self.n_devices)
    self.perm = [(i, (i + 1) % self.n_devices) for i in range(self.n_devices)]
    
  def mk_dft_matrix(self, n, inverse):
    inv_factor = 1
    if inverse:
      inv_factor = -1

    j = jnp.arange(n)  # Shape (n,)
    k = jnp.arange(n)[:, None]  # Shape (n, 1)
  
    theta = -2 * inv_factor * jnp.pi * j * k / n 
    dft_matrix = self.exp_itheta(theta)  

    if inverse:
      return 1/n * dft_matrix
    return dft_matrix
  
  def exp_itheta(self, theta: jnp.ndarray) -> jnp.ndarray:
    return jnp.cos(theta) + jnp.complex64(1.0j) * jnp.sin(theta)

  def run(self, vec):
    vec_shards = einops.rearrange(
      vec,
      '(n1 n) -> n1 n',
      n1 = self.n_devices
    )
    @functools.partial(
      jax.pmap,
      axis_name='axis')
    def compute_dft(vec):
      return self.shard_mul(vec)

    dft_vec_shards = compute_dft(vec_shards)
    dft_vec = einops.rearrange(
      dft_vec_shards,
      'n1 n -> (n1 n)',
      n1 = self.n_devices)
    
    return dft_vec

  def shard_mul(
    self,
    vec:jnp.ndarray
  ):
    LoopVars = collections.namedtuple('LoopVars', [
      'i', 'x_index', 'y_index', 'vec', 'accum_vec'])
    def cond(v):
      return v.i < self.n_devices
    def body(v):
      y_index = lax.ppermute(
        x = v.y_index,
        axis_name ='axis',
        perm = self.perm)
      vec = lax.ppermute(
        x = v.vec,
        axis_name = 'axis',
        perm = self.perm)
      delta_vec = jnp.einsum(
        'ij,j->i',
        self.shards[v.x_index][y_index],
        vec,
        precision=lax.Precision.HIGHEST)
      accum_vec = v.accum_vec + delta_vec
      i = v.i+1
      return LoopVars(i, x_index, y_index, vec, accum_vec)
    
    i=1
    x_index = lax.axis_index("axis")
    y_index = lax.axis_index("axis")
    accum_vec = jnp.einsum(
        'ij,j->i',
        self.shards[x_index][y_index],
        vec,
        precision=lax.Precision.HIGHEST)
    
    _, __, ___, ____, res_vec = lax.while_loop(
      cond,
      body,
      init_val=LoopVars(i, x_index, y_index, vec, accum_vec))
    return res_vec 


