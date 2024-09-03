from tensorprime import * 

def main():
    # Read settings and program arguments
    parser = argparse.ArgumentParser(description="TensorPrime")
    # config.getSettings()

    # In order to add more arguments to the parser,
    # attempt a similar declaration to below.
    # Anthing without a dash is becomes ordinal
    # and required
    parser.add_argument("--version", action="version",
                        version="TensorPrime 1.0")
    parser.add_argument("-p", "--prime", "--prp", type=int,
                        help="Run PRP primality test of exponent and exit")
    parser.add_argument(
        "--ll", type=int, help="Run LL primality test of exponent and exit")
    # parser.add_argument("--bench", action="store_true", help="perform testing etc")
    parser.add_argument("--iters", type=int,
                        help="Run test for this many iterations and exit")
    parser.add_argument("-f", "--fft", "--fftlen", "--sig",
                        "--siglen", type=int, help="FFT/Signal length")
    parser.add_argument("--shift", type=int,
                        help="Number of bits to shift the initial seed")
    parser.add_argument("--prp_base", type=int, default=3,
                        help="PRP base, Default: %(default)s")
    parser.add_argument("--prp_residue", type=int, choices=range(1, 6),
                        default=1, help="PRP residue type, Default: %(default)s")
    parser.add_argument("--proof_power", type=int, choices=range(1, 13), default=8,
                        help="Maximum proof power, every lower power halves storage requirements, but doubles the certification cost, Default: %(default)s")
    parser.add_argument("--proof_power_mult", type=int, choices=range(1, 5),
                        help="Proof power multiplier, to simulate a higher proof power by creating a larger proof file")
    parser.add_argument("-w", "--dir", "--workdir", default=os.curdir,
                        help="Working directory with the work, results and local files, Default: %(default)s (current directory)")
    parser.add_argument("-i", "--workfile", default="worktodo.txt",
                        help="Work File filename, Default: '%(default)s'")
    parser.add_argument("-r", "--resultsfile", default="results.txt",
                        help="Results File filename, Default: '%(default)s'")
    parser.add_argument("-l", "--localfile", default="local.ini",
                        help="Local configuration file filename, Default: '%(default)s'")
    parser.add_argument("--resume", type=int, default=-1,
                        help="Savefile/Checkpoint to resume from. Most recent is 0, Default %(default)s")
    parser.add_argument("-x", "--output_iter", type=int, default=10000,
                        help="Output/report every this many iterations, Default %(default)s iterations")
    parser.add_argument("--save_count", type=int, default=2,
                        help="Number of savefiles/checkpoints to keep (-1 to keep all), Default %(default)s")
    parser.add_argument("-c", "--save_iters", type=int, default=100000,
                        help="Write savefile/checkpoint every this many iterations, Default %(default)s iterations")
    parser.add_argument("--error", action="store_false", default=True,
                        help="Do Round off error (ROE) checking, Default %(default)s")
    # Prime95/MPrime: if near FFT length limit 1/else 128 iterations (ErrorCheck), if near FFT length limit 0.421875/else 0.40625 (MaxRoundoffError)
    # Mlucas: 1 iterations, 0.40625 warn/0.4375 error,
    # CUDALucas: 100 iterations (ErrorIterations, must have form k*10^m for k = 1, 2, or 5),
        # 40 (-e, ErrorLimit, must be 1-47), thus (ErrorLimit - ErrorLimit / 8.0 * log(ErrorIterations) / log(100.0)) / 100.0 = 0.35 (up to 0.41125)
    # [print("ErrorLimit:", i, "ErrorIterations:", j, "ROE:", (i - i / 8.0 * math.log(j) / math.log(100.0)) / 100.0) for i in range(1, 48) for j in (k*(10**m) for m in range(3) for k in (1, 2, 5))]
    parser.add_argument("--error_iter", type=int, default=100,
                        help="Run ROE checking every this many iterations, Default %(default)s iterations")
    parser.add_argument("-e", "--error_limit", type=float, default=0.4375,
                        help="Round off error (ROE) limit (0 - 0.47), Default %(default)s")
    parser.add_argument("--jacobi", action="store_false", default=True,
                        help="Do Jacobi Error Check (LL only), Default %(default)s")
    # Prime95/MPrime: 12 hours (JacobiErrorCheckingInterval)
    # GpuOwl v6.11: 500,000 iterations (must divide 10,000, usage says 1,000,000 iterations)
    parser.add_argument("--jacobi_iter", type=int, default=100000,
                        help="Run Jacobi Error Check every this many iterations (LL only), Default %(default)s iterations")
    parser.add_argument("--gerbicz", action="store_false", default=True,
                        help="Do Gerbicz Error Check (GEC) (PRP only), Default %(default)s")
    # Prime95/MPrime: 1,000, 1,000,000 iterations (PRPGerbiczCompareInterval),
    # Mlucas: 1,000, 1,000,000 iterations,
    # GpuOwl: 400 (-block <value>, must divide 10,000), 200,000 iterations (-log <step> value, must divide 10,000)
    parser.add_argument("--gerbicz_iter", type=int, default=100000,
                        help="Run GEC every this many iterations (PRP only), Default %(default)s iterations")
    parser.add_argument("--proof_files_dir", default=os.curdir,
                        help="Directory to hold large temporary proof files/residues, Default: %(default)s (current directory)")
    parser.add_argument(
        "--archive_proofs", help="Directory to archive PRP proof files after upload, Default: %(default)s")
    parser.add_argument("-u", "--username", default="ANONYMOUS",
                        help="GIMPS/PrimeNet User ID. Create a GIMPS/PrimeNet account: https://www.mersenne.org/update/. If you do not want a PrimeNet account, you can use ANONYMOUS.")
    parser.add_argument("-T", "--worktype", type=int, choices=[4, 100, 101, 102, 104, 150, 151, 152, 153, 154, 155, 160, 161], default=150, help="""Type of work, Default: %(default)s,
4 (P-1 factoring),
100 (smallest available first-time LL),
101 (double-check LL),
102 (world-record-sized first-time LL),
104 (100M digit number LL),
150 (smallest available first-time PRP),
151 (double-check PRP),
152 (world-record-sized first-time PRP),
153 (100M digit number PRP),
154 (smallest available first-time PRP that needs P-1 factoring),
155 (double-check using PRP with proof),
160 (first time Mersenne cofactors PRP),
161 (double-check Mersenne cofactors PRP)
"""
                        )
    parser.add_argument("--cert_work", action="store_false", default=True,
                        help="Get PRP proof certification work, Default: %default")
    parser.add_argument("--min_exp", type=int,
                        help="Minimum exponent to get from PrimeNet (2 - 999,999,999)")
    parser.add_argument("--max_exp", type=int,
                        help="Maximum exponent to get from PrimeNet (2 - 999,999,999)")
    parser.add_argument("-W", "--days_work", type=float, default=3.0,
                        help="Days of work to queue (1-90 days), Default: %(default)s days. Adds one to num_cache when the time left for the current assignment is less then this number of days.")
    parser.add_argument("-t", "--hours", type=float, default=6.0,
                        help="Hours between checkins and sending estimated completion dates, Default: %(default)s hours")
    parser.add_argument("-s", "--status", action="store_true", default=False,
                        help="Output a status report and any expected completion dates for all assignments and exit.")
    parser.add_argument("--upload_proofs", action="store_true", default=False,
                        help="Report assignment results, upload all PRP proofs and exit. Requires PrimeNet User ID.")
    parser.add_argument("--unreserve", type=int,
                        help="Unreserve assignment and exit. Use this only if you are sure you will not be finishing this exponent. Requires that the instance is registered with PrimeNet.")
    parser.add_argument("--unreserve_all", action="store_true", default=False,
                        help="Unreserve all assignments and exit. Quit GIMPS immediately. Requires that the instance is registered with PrimeNet.")
    parser.add_argument("--no_more_work", action="store_true", default=False,
                        help="Prevent script from getting new assignments and exit. Quit GIMPS after current work completes.")
    parser.add_argument("-H", "--hostname", default=platform.node(),
                        help="Optional computer name, Default: %(default)s")
    parser.add_argument("--hours_day", type=int, default=24,
                        help="Hours per day you expect to run TensorPrime (1 - 24), Default: %(default)s hours. Used to give better estimated completion dates.")
    parser.add_argument("--64-bit", action="store_true", default=False,
                        help="Enable 64 bit on Jax")

    # args is a dictionary in python types, in a
    # per-flag key-value mapping, which can be
    # accessed via, eg, flags["prime"], which will
    # return the integer passed in.
    args = parser.parse_args()

    # enable 64 bit support
    if getattr(args, "64_bit"):
        from jax.config import config as jax_config
        jax_config.update("jax_enable_x64", True)
        global jnp_precision
        jnp_precision = jnp.float64

    # Initialize logger specific to our runtime
    init_logger("tensorprime.log")

    p = args.prime
    if not p or not is_prime(p):
        parser.error("runtime requires a prime number for testing!")
    logging.info(f"Testing p={p}")

    # Used for save checkpointing. If a user's
    # program crashes they can resume from an
    # existing save file.
    resume = True
    preval = saveload.load(args.resume, p)
    if preval is None:
        resume = False

    # We choose the signal length by rounding down
    # the exponent to the nearest power of 2 and
    # then dividing by two twice. Experimentally
    # this will work for almost all known Mersenne
    # primes on the TPU out of the box. If a known-
    # working Mersenne prime throws a precision
    # error exception, double the FFT length and try
    # again.
    siglen = args.fft or 1 << max(
        1, int(math.log2(p / (10 if getattr(args, "64_bit") else 2.5))))
    logging.info(f"Using FFT length {siglen}")

    logging.info("Starting TensorPrime")
    logging.info("Starting Probable Prime Test.")
    logging.debug("Initializing arrays")
    bit_array, power_bit_array, weight_array = initialize_constants(
        p, siglen)
    logging.debug(f"bit_array: {bit_array}")
    logging.debug(f"power_bit_array: {power_bit_array}")
    logging.debug(f"weight_array: {weight_array}")
    logging.debug("Array initialization complete")
    start_time = time.perf_counter()

    if resume:
        logging.info(f"Resuming at iteration {preval['iteration']}")
        s = prptest(p, siglen, bit_array, power_bit_array, weight_array,
                    start_pos=preval["iteration"], s=preval["signal"],
                    d=preval["d"], prev_d=preval["d_prev"])
    else:
        s = prptest(p, siglen, bit_array, power_bit_array, weight_array)

    end_time = time.perf_counter()
    logging.debug(s)
    n = (1 << p) - 1
    is_probable_prime = result_is_nine(s, power_bit_array, n)
    logging.info(
        f"{p} tested in {timedelta(seconds=end_time - start_time)}: {'probably prime!' if is_probable_prime else 'composite'}")

    # Clean save checkpoint files now that the
    # program has finished.
    if not is_probable_prime or not is_known_mersenne_prime(p):
        saveload.clean(p)

if __name__ == '__main__':
  main()
