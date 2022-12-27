(* open Base
open Owl
open Ilqr_vae
open Variational
open Owl_parameters
open Accessor.O

let data i =
  Mat.load_txt
    (Printf.sprintf
       "/home/mmcs3/rds/hpc-work/_results/why_prep/baseline_for_double/reach_%i_500"
       i)

let data = C.broadcast (Array.init ~) *)
