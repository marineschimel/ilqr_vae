open Base
open Owl_parameters
open Covariance

module type Prior_T = sig
  module P : Owl_parameters.T
  open P

  val requires_linesearch : bool
  val spatial_stds : prms:p -> AD.t

  val kl_to_gaussian
    : [ `direct of
        prms:p -> mu:AD.t -> space:Covariance.P.p -> time:Covariance.P.p -> AD.t
      | `sampling_based
      ]

  val sample : prms:p -> n_steps:int -> m:int -> AD.t
  val neg_logp_t : prms:p -> k:int -> x:AD.t -> u:AD.t -> AD.t
  val neg_jac_t : (prms:p -> k:int -> x:AD.t -> u:AD.t -> AD.t) option
  val neg_hess_t : (prms:p -> k:int -> x:AD.t -> u:AD.t -> AD.t) option
  val logp : prms:p -> n_steps:int -> AD.t -> AD.t
end

module Gaussian_P = struct
  type 'a prm =
    { spatial_stds : 'a
    ; first_bin : 'a
    }
  [@@deriving accessors ~submodule:A]

  (* first bin has the interpretation of a rescaling of the std, not the variance *)
  let map ~f x = { spatial_stds = f x.spatial_stds; first_bin = f x.first_bin }

  let fold ?prefix ~init ~f x =
    let init = f init (x.spatial_stds, with_prefix ?prefix "spatial_stds") in
    f init (x.first_bin, with_prefix ?prefix "first_bin")
end

module Student_P = struct
  type 'a prm =
    { spatial_stds : 'a
    ; nu : 'a
    ; first_step : 'a
    }
  [@@deriving accessors ~submodule:A]

  let map ~f x =
    { spatial_stds = f x.spatial_stds; nu = f x.nu; first_step = f x.first_step }


  let fold ?prefix ~init ~f x =
    let init = f init (x.spatial_stds, with_prefix ?prefix "spatial_stds") in
    let init = f init (x.first_step, with_prefix ?prefix "first_step") in
    f init (x.nu, with_prefix ?prefix "nu")
end
