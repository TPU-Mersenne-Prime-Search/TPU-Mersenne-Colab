# TensorPrime

This program was developed as part of a two-term senior capstone project at Portland State University. Our goal was to develop a program which uses the Tensor Processing Unit to accelerate the Probable Prime Test, an algorithm for testing the (probable) primality of large [Mersenne numbers](https://mathworld.wolfram.com/MersenneNumber.html).

Currently due to precision issues with the TPU the known primes we are able to check range from `2^3 - 1` to `2^4423 - 1` inclusive.

See our project [Wiki](https://github.com/TPU-Mersenne-Prime-Search/TensorPrime/wiki) for more details about the project.

# Quickstart
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TPU-Mersenne-Prime-Search/TensorPrime/blob/master/GoogleColabTPU.ipynb)

Currently due to precision limitations of the `bfloat16` data type used by the TPU for matrix multiplication operations, the largest known Mersenne prime this program is able to check is `2^4423 - 1`. Because of this the PrimeNet assignments this program supports will not work on the TPU architecture. To run TensorPrime on the TPU, the following steps are recommended.

1. Run the cell. Follow the prompts to connect to your Google Drive account. As Colab VMs do not provide persistent storage Google Drive is necessary to store persistent information about the program such as user settings.

## Example

To check `2^4423 - 1` for primality:

```
!python3 main.py -p 4423
```