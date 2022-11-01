open Base
open Owl_parameters

module type Dims_T = sig
  val n : int
  val m : int
end

module type T = sig
  module P : Owl_parameters.T
  open P
  include Dims_T

  val n_beg : int
  val requires_linesearch : bool
  val dyn : theta:p -> ext_u:AD.t option -> k:int -> x:AD.t -> u:AD.t -> AD.t
  val dyn_x : (theta:p -> ext_u:AD.t option -> k:int -> x:AD.t -> u:AD.t -> AD.t) option
  val dyn_u : (theta:p -> ext_u:AD.t option -> k:int -> x:AD.t -> u:AD.t -> AD.t) option
end

module Nonlinear_P = struct
  type 'a prm =
    { a : 'a
    ; bias : 'a
    ; b_ext : 'a
    ; b : 'a option
    }
  [@@deriving accessors ~submodule:A]

  let map ~f x = { a = f x.a; bias = f x.bias; b_ext = f x.b_ext; b = Option.map ~f x.b }

  let fold ?prefix ~init ~f x =
    let init = f init (x.a, with_prefix ?prefix "a") in
    let init = f init (x.bias, with_prefix ?prefix "bias") in
    match x.b with
    | None -> init
    | Some b -> f init (b, with_prefix ?prefix "b")
end
