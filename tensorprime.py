import jax
#jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax, jit
from functools import partial
import time
from datetime import timedelta
import logging
import saveload
import math
from config import config
from log_helper import init_logger

jnp_precision = jnp.float32

def is_known_mersenne_prime(p):
  """Returns True if the given Mersenne prime is known, and False otherwise."""
  primes = frozenset([2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203, 2281, 3217, 4253, 4423, 9689, 9941, 11213, 19937, 21701, 23209, 44497, 86243, 110503, 132049, 216091,
                      756839, 859433, 1257787, 1398269, 2976221, 3021377, 6972593, 13466917, 20996011, 24036583, 25964951, 30402457, 32582657, 37156667, 42643801, 43112609, 57885161, 74207281, 77232917, 82589933])
  return p in primes


def is_prime(n):
  """Return True if n is a prime number, else False."""
  return n >= 2 and not any(
    not n % p for p in range(2, math.isqrt(n) + 1))
def rollback():
  gerbicz_error_general = "Gerbicz error checking found an error but had nothing to rollback to. Exiting"
  if jnp.shape(gec_s_saved) is None:
    raise Exception(gerbicz_error_general)
  if jnp.shape(gec_d_saved) is None:
    raise Exception(gerbicz_error_general)
  if gec_i_saved is None:
    raise Exception(gerbicz_error_general)
  return gec_i_saved, gec_s_saved, gec_d_saved


def update_gec_save(i, s, d):
  global gec_i_saved, gec_s_saved, gec_d_saved
  gec_i_saved = i
  gec_s_saved = s.copy()
  gec_d_saved = d.copy()  
  
def mk_bit_array(exponent, signal_length):
  indices = jnp.arange(1, signal_length+1, dtype=jnp_precision)
  return jnp.ceil((exponent * indices) / signal_length) - jnp.ceil(exponent * (indices - 1) / signal_length)

def mk_power_bit_array(bit_array):
  return jnp.power(2, bit_array)

def mk_weight_array(exponent, signal_length):
  indices = jnp.arange(0, signal_length, dtype=jnp_precision)
  return jnp.power(2, (jnp.ceil(exponent * indices /signal_length) - (exponent * indices / signal_length)))

def initialize_constants(exponent, signal_length):
  bit_array = mk_bit_array(exponent, signal_length)
  power_bit_array = mk_power_bit_array(bit_array)
  weight_array = mk_weight_array(exponent, signal_length)
  return bit_array, power_bit_array, weight_array

def firstcarry(signal, power_bit_array):
  def body(carry_val, x):
    signal_val, power_bit = x
    val = signal_val + carry_val
    carry_val = jnp.floor_divide(val, power_bit)
    return carry_val, jnp.mod(val, power_bit)

  carry, vals = lax.scan(body, 0, (signal, power_bit_array))
  
  return carry, vals

def secondcarry(carryval, signal, power_bit_array):
  def body(carryval, x):
    signal_val, power_bit = x
    val = carryval + signal_val
    carryval = jnp.floor_divide(val, power_bit)
    return carryval, jnp.mod(val, power_bit)
  carryval, vals = lax.scan(body, carryval, (signal, power_bit_array))
  return vals

def partial_carry(signal, power_bit_array):
  def forloop_body(i, vals):
    signal, carry_values = vals
    signal = jnp.add(signal, carry_values)
    carry_values = jnp.floor_divide(signal, power_bit_array)
    carry_values = jnp.roll(carry_values, 1)
    signal = jnp.mod(signal, power_bit_array)
    return signal, carry_values
  carry_values = jnp.empty_like(signal)
  (signal, carry_values) = lax.fori_loop(
    0, 3, forloop_body, (signal, carry_values))

  return jnp.add(signal, carry_values)

def weighted_transform(signal_to_transform, weight_array):
  weighted_signal = jnp.multiply(signal_to_transform, weight_array)
  return jnp.fft.rfft(weighted_signal)

def inverse_weighted_transform(transformed_weighted_signal, weight_array):
  weighted_signal = jnp.fft.irfft(transformed_weighted_signal)
  return jnp.divide(weighted_signal, weight_array)

def balance(signal, power_bit_array):
  def subtract_and_carry(x):
    signal_val, power_bit = x
    return signal_val - power_bit, 1
  def set_carry_to_zero(x):
    signal_val, _ = x
    return signal_val, 0
  def body(carry_val, x):
    signal_val, power_bit = x
    signal_val = signal_val + carry_val
    signal_val, carry_val = lax.cond(
      signal_val >= power_bit/2,
      subtract_and_carry,
      set_carry_to_zero,
      (signal_val, power_bit)
    )
    return carry_val, signal_val
  carry_val, signal = lax.scan(body, 0, (signal, power_bit_array))
  return signal.at[0].set(signal[0] + carry_val)

@jit
def squaremod_with_ibdwt(
    signal,
    prime_exponent,
    signal_length,
    power_bit_array,
    weight_array):
  """
  Squares a number (mod 2^prime_exponent - 1) as
  described in "Discrete Weighted Transforms".
  This is functionally identital to a call to
  `multmod_with_ibdwt` where signal1 and signal2
  are identical, with the added benefit of 1 fewer
  FFT runs.
  """
  balanced_signal = balance(signal, power_bit_array)
  transformed_signal = weighted_transform(balanced_signal, weight_array)
  squared_transformed_signal = jnp.multiply(
    transformed_signal, transformed_signal)
  squared_signal = inverse_weighted_transform(
    squared_transformed_signal, weight_array)
  rounded_signal = jnp.round(squared_signal)
  roundoff = jnp.max(jnp.abs(jnp.subtract(squared_signal, rounded_signal)))
  parially_carried_signal = partial_carry(rounded_signal, power_bit_array)
  return parially_carried_signal, roundoff

@jit
def multmod_with_ibdwt(signal1, signal2, prime_exponent,
                       signal_length, power_bit_array, weight_array):
  """
  Multiplies two numbers
  (mod 2^prime_exponent - 1) as described in
  "Discrete Weighted Transforms".
  """
  balanced_signal1 = balance(signal1, power_bit_array)
  balanced_signal2 = balance(signal2, power_bit_array)
  transformed_signal1 = weighted_transform(balanced_signal1, weight_array)
  transformed_signal2 = weighted_transform(balanced_signal2, weight_array)
  multiplied_transformed_signal = jnp.multiply(
    transformed_signal1, transformed_signal2)
  multiplied_signal = inverse_weighted_transform(
    multiplied_transformed_signal, weight_array)
  rounded_signal = jnp.round(multiplied_signal)
  roundoff = jnp.max(jnp.abs(jnp.subtract(
    multiplied_signal, rounded_signal)))
  carryval, firstcarried_signal = firstcarry(rounded_signal, power_bit_array)
  fullycarried_signal = secondcarry(
    carryval, firstcarried_signal, power_bit_array)
  return fullycarried_signal, roundoff

def prptest(exponent, siglen, bit_array, power_bit_array,
            weight_array, start_pos=0, s=None, d=None, prev_d=None):
  # Load settings values for this function
  GEC_enabled = config.getboolean("TensorPrime", "GECEnabled")
  GEC_iterations = config.getint("TensorPrime", "GECIter")

  # Uses counters to avoid modulo check
  save_i_count = save_iter = config.getint("TensorPrime", "SaveIter")
  print_i_count = print_iter = config.getint("TensorPrime", "PrintIter")
  if s is None:
    s = jnp.zeros(siglen).at[0].set(3)
  i = start_pos
  current_time = start = time.perf_counter_ns()
  while i < exponent:
      # Create a save checkpoint every save_i_count
      # iterations.
    if not save_i_count:
      logging.info(
        f"Saving progress (performed every {save_iter} iterations)...")
      saveload.save(exponent, siglen, s, i)
      save_i_count = save_iter
    save_i_count -= 1

        # Print a progress update every print_i_count
        # iterations
    if not print_i_count:
      temp = time.perf_counter_ns()
      delta_time = temp - current_time
      current_time = temp
      logging.info(
        f"Time elapsed at iteration {i}: {timedelta(microseconds=(current_time - start) // 1000)}, {(delta_time / 1000) / print_iter:.2f} µs/iter")
      print_i_count = print_iter
    print_i_count -= 1
      
    # Gerbicz error checking
    if GEC_enabled:
      L = math.isqrt(GEC_iterations)
      L_2 = L * L
      three_signal = jnp.zeros(siglen).at[0].set(3)
      if d is None:
        prev_d = d = three_signal
        update_gec_save(i, s, d)

      # Every L iterations, update d and prev_d
      if i and not i % L:
        prev_d = d
        d, roundoff = multmod_with_ibdwt(
          d, s, exponent, siglen, power_bit_array, weight_array)
      # Every L^2 iterations, check the current d value with and independently calculated d
      if i and (not i % L_2 or (
          not i % L and i + L > exponent)):
        prev_d_pow_signal = prev_d
        for _j in range(L):
          prev_d_pow_signal, roundoff = squaremod_with_ibdwt(prev_d_pow_signal, exponent, siglen,
                                                           power_bit_array, weight_array)
        check_value, roundoff = multmod_with_ibdwt(three_signal, prev_d_pow_signal, exponent, siglen,
                                                   power_bit_array, weight_array)

        if not jnp.array_equal(d, check_value):
          logging.error("error occurred. rolling back to last save.")
          i, s, d = rollback()

        else:
          logging.info("updating gec_save")
          update_gec_save(i, s, d)

      # Running squaremod
    s, roundoff = squaremod_with_ibdwt(
      s, exponent, siglen, power_bit_array, weight_array)

    # Quick check to avoid roundoff errors. If a
    # roundoff error is encountered we have no
    # current method for dealing with it, so throw
    # an exception and terminate the program.
    if roundoff > 0.40625:
      logging.warning(f"Roundoff (iteration {i}): {roundoff}")
      if roundoff > 0.4375:
        msg = f"Roundoff error exceeded threshold (iteration {i}): {roundoff} vs 0.4375"
        raise Exception(msg)
          
    i += 1

  # The "partial carry" may leave some values in
  # an incorrect state. Running a final carry
  # will clean this up to produce the residue we
  # want to check.
  carry_val, s = firstcarry(s, power_bit_array)
  return secondcarry(carry_val, s, power_bit_array)


# Sum up the values in the signal until the total
# is 9. If there are any values left in the signal
# we know the total value cannot be 9.
def result_is_nine(signal, power_bit_array, n):
  res = base = 0
  i = 0
  nine = 9 % n
  while res < nine and i < signal.shape[0]:
    res += int(signal[i]) * (1 << base)
    base += int(power_bit_array[i])
    i += 1
  return res == nine and not signal[i:].any()

primes = [2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203, 2281, 3217, 4253, 4423, 9689, 9941, 11213, 19937, 21701, 23209, 44497, 86243, 110503, 132049, 216091,
                       756839, 859433, 1257787, 1398269, 2976221, 3021377, 6972593, 13466917, 20996011, 24036583, 25964951, 30402457, 32582657, 37156667, 42643801, 43112609, 57885161, 74207281, 77232917, 82589933]

primes = [2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203, 2281, 3217, 4253, 4423, 9689, 9941, 11213, 19937, 21701, 23209, 44497, 86243, 110503, 132049, 216091,
                       756839, 859433, 1257787, 1398269, 2976221, 3021377, 6972593, 13466917, 20996011, 24036583, 25964951, 30402457, 32582657, 37156667, 42643801, 43112609, 57885161, 74207281, 77232917, 82589933]

def find_closest(x, xs):
  min_dist = 100000000000
  for i,x_ in enumerate(xs):
    dist = abs(x_ - x)
    if dist <= min_dist:
      min_dist = dist
    else:
      return xs[i-1]
exponent = 4423
siglen= max(1,int(jnp.log2(exponent/2.5)))
check = True
while check:
  try:
    bit_array, power_bit_array, weight_array = initialize_constants(
      exponent, siglen)
    t1 = time.time()
    s = prptest(exponent, siglen, bit_array, power_bit_array, weight_array)
    t2 = time.time()
    n = (1<<exponent) - 1
    print(result_is_nine(s, power_bit_array, n))
    print("Success!:", s)
    check = False
  except Exception as e:
    print(e)
    print("Tried ", siglen, " but ran into round off errors. Incrementing...")
    siglen *= 2
