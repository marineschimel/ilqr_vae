# iLQR-VAE

## Installation steps

- install opam : https://opam.ocaml.org/doc/Install.html
- choose opam switch 4.12.0 
- install dune (run `opam install dune`) 
- install owl (run `opam install owl`); you will need to have OpenBLAS installed (see below in case issues with OpenBLAS arise). 
- install mpi, accessor, owl_ode, (`opam install mpi accessor ppx_accessor owl-ode ppx_deriving_yojson`)
- clone https://github.com/hennequin-lab/adam, https://github.com/hennequin-lab/owl_parameters, https://github.com/hennequin-lab/comm, https://github.com/hennequin-lab/cmdargs, https://github.com/tachukao/dilqr, https://github.com/tachukao/owl_bmo and for each of these repos, do `dune build @install` followed by `dune install` (after `cd`-ing into the corresponding directory)

## To run examples

- compile the example by running e.g. `dune build src/lorenz.exe`. If linking issues arise, please get in touch.
- to execute it on multiple cores, run `mpirun -n [number of cores] _build/default/src/lorenz.exe -d [results_directory]` (where `[results_directory]` is where you want your results to be saved). Depending on the example file you are trying to execute, there might be additional command line arguments.
 
## OpenBLAS installation

- on certain operating systems linking errors to OpenBLAS can arise when installing owl. One solution to circumvent them is to install OpenBLAS from source (https://github.com/xianyi/OpenBLAS.git), and to then manually include the path to the OpenBLAS installation in LD_LIBRARY_PATH and PKG_CONFIG_PATH.


