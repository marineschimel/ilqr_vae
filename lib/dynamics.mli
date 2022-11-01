include module type of Dynamics_typ

module Integrate (D : T) : sig
  val integrate : prms:D.P.p -> ext_u:AD.t option -> u:AD.t -> AD.t
end

module Nonlinear (X : sig
  include Dims_T

  val m_ext : int
  val phi : [ `linear | `nonlinear of (AD.t -> AD.t) * (AD.t -> AD.t) ]
end) : sig
  include T with type 'a P.prm = 'a Nonlinear_P.prm

  val init : ?radius:float -> Owl_parameters.setter -> Owl_parameters.t P.prm
end
