include module type of Prior_typ

module Gaussian (Dims : Dims_T) : sig
  include T with type 'a P.prm = 'a Gaussian_P.prm

  val init : ?spatial_std:float -> ?first_bin:float -> Owl_parameters.setter -> P.p
end

module Student (Dims : Dims_T) : sig
  include T with type 'a P.prm = 'a Student_P.prm

  val init
    :  ?pin_std:bool
    -> ?spatial_std:float
    -> ?nu:float
    -> Owl_parameters.setter
    -> P.p
end
