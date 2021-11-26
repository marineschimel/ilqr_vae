include module type of Covariance_typ

module Covariance : sig
  module P : Owl_parameters.T with type 'a prm = 'a Covariance_P.prm

  val init
    :  ?no_triangle:bool
    -> ?pin_diag:bool
    -> ?sigma2:float
    -> Owl_parameters.setter
    -> int
    -> P.p

  val to_chol_factor : P.p -> AD.t
  val invert : P.p -> P.p
end
