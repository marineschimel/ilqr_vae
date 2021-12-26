include module type of Likelihood_typ

module Gaussian (X : sig
  include Dims_T

  val label : string
  val normalize_c : bool
end) : sig
  include
    T
      with type 'a P.prm = 'a Gaussian_P.prm
       and type output_t = AD.t
       and type output = AD.t

  val init : ?sigma2:float -> ?bias:float -> Owl_parameters.setter -> P.p
end

module Poisson (X : sig
  include Dims_T

  val label : string
  val dt : AD.t
  val link_function : AD.t -> AD.t
  val d_link_function : AD.t -> AD.t
  val d2_link_function : AD.t -> AD.t
end) : sig
  include
    T
      with type 'a P.prm = 'a Poisson_P.prm
       and type output_t = AD.t
       and type output = AD.t

  val init : Owl_parameters.setter -> P.p
  val pre_sample_before_link_function : prms:P.p -> z:AD.t -> AD.t
end

module Pair (L1 : T) (L2 : T) :
  T
    with type 'a P.prm = ('a L1.P.prm, 'a L2.P.prm) Pair_P.prm_
     and type output_t = (L1.output_t, L2.output_t) Pair_P.prm_
     and type output = (L1.output, L2.output) Pair_P.prm_
