# Bitstream provenance

`finn-accel.bit` and `finn-accel.hwh` in this folder are the files that were
programmed onto the PYNQ-Z2 board and that produced every measurement reported
in the project.

    finn-accel.bit   md5  604477bf9bea686e3065ece0fbb1d250
    finn-accel.hwh   md5  e1e77a790708ef07e007e405e97c4895

Verified on 2026-09-29 by comparing against the copy on the board at
`/home/xilinx/jupyter_notebooks/cifar10_hardware/deploy/.../bitfile/`.
The md5 sums are identical.

Source: the FINN build directory
`build_vgg_lite_int8_bitfile_final9283rce6/bitfile/`, built 2026-08-22.

To confirm that a bitstream is the one that produced the reported results,
compare its md5 checksum with the values above; the file name alone is not
enough.
