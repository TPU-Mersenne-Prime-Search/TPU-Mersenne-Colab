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
    self.partition = [n_devices, n_devices]
    self.inverse = inverse
    dft_real, dft_imag = self.mk_dft_matrix(n)
    self.real_shards = einops.rearrange(
      dft_real,
      '(m1 m) (n1 n) -> m1 n1 m n',
      m1=self.n_devices,
      n1=self.n_devices)
    self.imag_shards = einops.rearrange(
      dft_imag,
      '(m1 m) (n1 n) -> m1 n1 m n',
      m1=self.n_devices,
      n1=self.n_devices)
    self.perm = [(i, (i + 1) % self.n_devices) for i in range(self.n_devices)]
    
    
  def mk_dft_matrix(self, n):
    inv_factor = 1
    if self.inverse:
      inv_factor = -1

    j = jnp.arange(n)  # Shape (n,)
    k = jnp.arange(n)[:, None]  # Shape (n, 1)
    
    theta = -2 * inv_factor * jnp.pi * j * k / n 
    dft_cos = jnp.cos(theta)
    dft_sin = jnp.sin(theta)
    
    return dft_cos, dft_sin

  
  def run(self, vec):
    @functools.partial(
      jax.pmap,
      axis_name='axis')
    def compute_real(vec):
      return self.real_shard_mul(vec)
    
    @functools.partial(
      jax.pmap,
      axis_name='axis')
    def compute_imag(vec):
      return self.imag_shard_mul(vec)
    
    vec_real, vec_imag = vec
    
    vec_real_shards = einops.rearrange(
      vec_real,
      '(n1 n) -> n1 n',
      n1 = self.n_devices
    )
    Av_shards = compute_real(vec_real_shards)
    Bv_shards = compute_imag(vec_real_shards)
    Av = einops.rearrange(
      Av_shards,
      'n1 n -> (n1 n)',
      n1 = self.n_devices)
    Bv = einops.rearrange(
      Bv_shards,
      'n1 n -> (n1 n)',
      n1 = self.n_devices)
    real_vec = Av
    imag_vec = Bv
    if len(vec_imag) != 0:
      vec_imag_shards = einops.rearrange(
        vec_imag,
        '(n1 n) -> n1 n',
        n1 = self.n_devices
      )
      
      Aw_shards = compute_real(vec_imag_shards)
      Bw_shards = compute_imag(vec_imag_shards)
      
      Aw = einops.rearrange(
        Aw_shards,
        'n1 n -> (n1 n)',
        n1 = self.n_devices)
      
      Bw = einops.rearrange(
        Bw_shards,
        'n1 n -> (n1 n)',
        n1 = self.n_devices)
      
      #(A+iB)(v+iw) = (Av - Bw) + i(Bv + Aw)
      real_vec -= Bw
      imag_vec += Aw
      
    if self.inverse:
      return real_vec / len(real_vec), imag_vec / len(imag_vec)
    return real_vec, imag_vec
    
      

  def real_shard_mul(
    self,
    vec:jnp.ndarray,
      
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
        self.real_shards[v.x_index][y_index],
        vec)
      accum_vec = v.accum_vec + delta_vec
      i = v.i+1
      return LoopVars(i, x_index, y_index, vec, accum_vec)
    
    i=1
    x_index = lax.axis_index("axis")
    y_index = lax.axis_index("axis")
    accum_vec = jnp.einsum(
        'ij,j->i',
        self.real_shards[x_index][y_index],
        vec)

    
    _, __, ___, ____, res_vec = lax.while_loop(
      cond,
      body,
      init_val=LoopVars(i, x_index, y_index, vec, accum_vec))
    return res_vec
  
  def imag_shard_mul(
    self,
    vec:jnp.ndarray,
      
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
        self.imag_shards[v.x_index][y_index],
        vec)
      accum_vec = v.accum_vec + delta_vec
      i = v.i+1
      return LoopVars(i, x_index, y_index, vec, accum_vec)
    
    i=1


    x_index = lax.axis_index("axis")
    y_index = lax.axis_index("axis")
    accum_vec = jnp.einsum(
        'ij,j->i',
        self.imag_shards[x_index][y_index],
        vec,
        precision=lax.Precision.HIGHEST)
    
    _, __, ___, ____, res_vec = lax.while_loop(
      cond,
      body,
      init_val=LoopVars(i, x_index, y_index, vec, accum_vec))
    return res_vec 


################################################
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

n_devices = len(jax.devices())

x = jnp.arange(128)

dft = DFT(128, inverse=False)
idft = DFT(128, inverse=True)
res1 = dft.run([x,[]])
res2 = idft.run(res1)

  




