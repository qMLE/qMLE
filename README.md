qMLE
====

This collection of MATLAB files is aimed at making computing the maximum-likelihood estimator (MLE) for quantum tomography straightforward, simple and fast.  The most important files are:

 * `qse_apg.m` quantum state estimation using accelerated projected gradients (APG).
 * `qmt.m` Born-rule computation, with speedups for product structure.
 * `tutorial.m` a tutorial for using this MATLAB package.
 * `html` directory of generated documentation from `tutorial.m`
  * `tutorial.html` HTML version of the tutorial.

References
----

 * J. Shang, Z. Zhang, and H. K. Ng, "[Superfast maximum likelihood reconstruction for quantum tomography](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.95.062336)," *Phys. Rev. A* **95**, 062336 (2017).


License
----

Copyright 2017 J. Shang, Z. Zhang and H. K. Ng

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

