include module type of Likelihoods_typ

module Gaussian (X : sig
  val label : string
  val normalize_c : bool
end) : sig
  include
    Likelihood_T
      with type 'a P.prm = 'a Gaussian_P.prm
       and type datum = AD.t
       and type data = AD.t

  val init
    :  ?sigma2:float
    -> ?bias:float
    -> n:int
    -> n_output:int
    -> Owl_parameters.setter
    -> P.p
end

module Poisson (X : sig
  val label : string
  val dt : AD.t
  val link_function : AD.t -> AD.t
  val d_link_function : AD.t -> AD.t
  val d2_link_function : AD.t -> AD.t
end) : sig
  include
    Likelihood_T
      with type 'a P.prm = 'a Poisson_P.prm
       and type datum = AD.t
       and type data = AD.t

  val init : n:int -> n_output:int -> Owl_parameters.setter -> P.p
  val pre_sample_before_link_function : prms:P.p -> z:AD.t -> AD.t
end

module Pair (L1 : Likelihood_T) (L2 : Likelihood_T) :
  Likelihood_T
    with type 'a P.prm = ('a L1.P.prm, 'a L2.P.prm) Pair_P.prm_
     and type datum = (L1.datum, L2.datum) Pair_P.prm_
     and type data = (L1.data, L2.data) Pair_P.prm_
