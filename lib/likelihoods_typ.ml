open Base
open Owl_parameters

module type Likelihood_T = sig
  module P : Owl_parameters.T
  open P

  type datum
  type data

  val requires_linesearch : bool
  val label : string
  val save_data : ?prefix:string -> data -> unit
  val data_slice : k:int -> data -> datum
  val to_mat_list : data -> (string * AD.t) list
  val size : prms:p -> int
  val pre_sample : prms:p -> z:AD.t -> data
  val sample : prms:p -> z:AD.t -> data
  val neg_logp_t : prms:p -> data_t:datum -> k:int -> z_t:AD.t -> AD.t
  val neg_jac_t : (prms:p -> data_t:datum -> k:int -> z_t:AD.t -> AD.t) option
  val neg_hess_t : (prms:p -> data_t:datum -> k:int -> z_t:AD.t -> AD.t) option
  val logp : prms:p -> data:data -> z:AD.t -> AD.t
end

module Gaussian_P = struct
  type 'a prm =
    { c : 'a
    ; c_mask : AD.t option
    ; bias : 'a
    ; variances : 'a (* 1 x space *)
    }
  [@@deriving accessors ~submodule:A]

  let map ~f x =
    { c = f x.c; c_mask = x.c_mask; bias = f x.bias; variances = f x.variances }


  let fold ?prefix ~init ~f x =
    let init = f init (x.c, with_prefix ?prefix "c") in
    let init = f init (x.bias, with_prefix ?prefix "bias") in
    f init (x.variances, with_prefix ?prefix "variances")
end

module Poisson_P = struct
  type 'a prm =
    { c : 'a
    ; c_mask : AD.t option
    ; bias : 'a
    ; gain : 'a
    }
  [@@deriving accessors ~submodule:A]

  let map ~f x = { c = f x.c; c_mask = x.c_mask; bias = f x.bias; gain = f x.gain }

  let fold ?prefix ~init ~f x =
    let init = f init (x.c, with_prefix ?prefix "c") in
    let init = f init (x.bias, with_prefix ?prefix "bias") in
    f init (x.gain, with_prefix ?prefix "gain")
end

include Pair_typ
