# Source attribution

The exponential coefficients/table and algorithm derive from Arm
optimized-routines commit f5205d212ac67727689176b7fcfaef7864d898a5:
`math/expf.c` (Copyright2017–2025 Arm Limited) and `math/exp2f_data.c`
(Copyright2017–2018 Arm Limited), each marked
`SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception`.
V86 chooses the MIT option below. Source and SHA pins are retained by V84 and
bound in this packet's SOURCE_PINS. Modifications are explicit rounded CUDA
operations matching the separately inspected installed libm FMA sequence.
No glibc LGPL source body was copied into the implementation; its machine-code
operation sequence and constants were used as the independent target evidence.

MIT License

Copyright (c)2017–2025 Arm Limited

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

The packed NV4 operation mapping and Q8/software-half sequence correspond to
the pinned Strata/ggml sources. Their original MIT notices are retained in
`STRATA_LICENSE` and `GGML_LICENSE`; the original CPU-v224 archive is linked
unchanged as reference. No model-weight data is included.
