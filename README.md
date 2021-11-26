# iLQR-VAE

## Installation steps

- install opam : https://opam.ocaml.org/doc/Install.html
- choose opam switch 4.12.0 
- install dune (run `opam install dune`) 
- install owl (run `opam install owl`); you will need to have OpenBLAS installed.
- install mpi, accessor, owl_ode, (`opam install mpi accessor ppx_accessor owl_ode`)
- clone https://github.com/ghennequin/adam, https://github.com/hennequin-lab/owl_parameters, https://github.com/hennequin-lab/comm, https://github.com/hennequin-lab/cmdargs and for each of these repos, do `dune build @install` followed by `dune install` (after `cd`-ing into the corresponding directory)
- install dilqr (open zip file and run `dune build @install` and `dune install`) 

## To run examples

- compile the example by running e.g. `dune build src/lorenz.exe`. If linking issues arise, please get in touch.
- to execute it on multiple cores, run `mpirun -n [number of cores] _build/default/src/lorenz.exe -d [results_directory]` (where `[results_directory]` is where you want your results to be saved). Depending on the example file you are trying to execute, there might be additional command line arguments.
 