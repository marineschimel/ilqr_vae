open Owl_parameters

module type Dims_T = sig
  val n : int
  val m : int
  val n_beg : int
end

module type T = sig
  include Dims_T
  module U : Prior.T
  module D : Dynamics.T
  module L : Likelihood.T
  module P : Owl_parameters.T

  val integrate : prms:P.p -> ext_u:AD.t option -> u:AD.t -> AD.t

  (** Samples the generative model; samples [u] from the prior unless [u] is supplied.
      Uses [L.sample] by default, unless [pre=true] in which case [L.pre_sample] is used instead *)
  val sample
    :  prms:P.p
    -> ext_u:AD.t option
    -> ?id:int
    -> ?pre:bool
    -> [ `prior of int | `some of AD.t ]
    -> ([ `uz | `o ], L.output) Data.t

  (** Log prior density p(u) for a K x T x M batch of inputs *)
  val log_prior : prms:P.p -> AD.t -> AD.t

  (** Log likelihood p(data.o | data.z) *)
  val log_likelihood : prms:P.p -> ([ `uz | `o ], L.output) Data.t -> AD.t
end

module P = struct
  type ('u, 'd, 'l) prm_ =
    { prior : 'u
    ; dynamics : 'd
    ; likelihood : 'l
    }
  [@@deriving accessors ~submodule:A]

  module Make (U : Owl_parameters.T) (D : Owl_parameters.T) (L : Owl_parameters.T) =
  struct
    type 'a prm = ('a U.prm, 'a D.prm, 'a L.prm) prm_

    let map ~f prms =
      { prior = U.map ~f prms.prior
      ; dynamics = D.map ~f prms.dynamics
      ; likelihood = L.map ~f prms.likelihood
      }


    let fold ?prefix ~init ~f prms =
      let w = with_prefix ?prefix in
      let init = U.fold ~prefix:(w "prior") ~init ~f prms.prior in
      let init = D.fold ~prefix:(w "dynamics") ~init ~f prms.dynamics in
      L.fold ~prefix:(w "likelihood") ~init ~f prms.likelihood
  end
end
