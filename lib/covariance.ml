open Base
open Owl
include Covariance_typ
module P = Owl_parameters.Make (P)
open Covariance_typ.P

let init
    ?(no_triangle = false)
    ?(pin_diag = false)
    ?(sigma2 = 1.)
    (set : Owl_parameters.setter)
    n
  =
  let d = Mat.create 1 n Maths.(sqrt sigma2) in
  let t = AD.Mat.zeros n n in
  { d = (if pin_diag then Owl_parameters.pinned else set) ~above:0.0001 (AD.pack_arr d)
  ; t = (if no_triangle then Owl_parameters.pinned else set) t
  }


let to_chol_factor c =
  let t = Owl_parameters.extract c.t in
  let d = Owl_parameters.extract c.d in
  AD.Maths.(triu ~k:1 t + diagm d)


let invert c =
  let ell = c |> to_chol_factor in
  let ell_inv = AD.Linalg.linsolve ~typ:`u ell AD.Mat.(eye (row_num ell)) in
  let d = AD.Maths.diag ell in
  let t = ell_inv in
  let open Owl_parameters in
  let d =
    match c.d with
    | Pinned _ -> Pinned d
    | Learned_bounded (_, lb, None) -> Learned_bounded (d, lb, None)
    | _ -> failwith "oooops"
  in
  let t =
    match c.d with
    | Pinned _ -> Pinned t
    | _ -> Learned t
  in
  { d; t }

