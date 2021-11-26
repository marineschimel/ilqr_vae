include module type of Priors_typ

module Gaussian (X : sig
  val n_beg : int Option.t
end) : sig
  include Prior_T with type 'a P.prm = 'a Gaussian_P.prm

  val init
    :  ?spatial_std:float
    -> ?first_bin:float
    -> m:int
    -> Owl_parameters.setter
    -> P.p
end

module Student (X : sig
  val n_beg : int Option.t
end) : sig
  include Prior_T with type 'a P.prm = 'a Student_P.prm

  val init
    :  ?pin_std:bool
    -> ?spatial_std:float
    -> ?nu:float
    -> m:int
    -> Owl_parameters.setter
    -> P.p
end
