include module type of Dynamics_typ

module Integrate (D : T) : sig
  val integrate : prms:D.P.p -> ext_u:AD.t option -> u:AD.t -> AD.t
end

module Linear (Dims : Dims_T) : sig
  include T with type 'a P.prm = 'a Linear_P.prm

  val init
    :  dt_over_tau:float
    -> alpha:float
    -> beta:float
    -> Owl_parameters.setter
    -> P.p

  val unpack_a : prms:P.p -> AD.t
end

module Nonlinear (X : sig
  include Dims_T

  val phi : [ `linear | `nonlinear of (AD.t -> AD.t) * (AD.t -> AD.t) ]
end) : sig
  include T with type 'a P.prm = 'a Nonlinear_Init_P.prm

  val init : ?radius:float -> Owl_parameters.setter -> Owl_parameters.t P.prm
end

(** Mini-GRU (Heck, 2017) *)
module MGU (X : sig
  include Dims_T

  val phi : AD.t -> AD.t
  val d_phi : AD.t -> AD.t
  val sigma : AD.t -> AD.t
  val d_sigma : AD.t -> AD.t
end) : sig
  include T with type 'a P.prm = 'a MGU_P.prm

  val init : Owl_parameters.setter -> P.p
  val default_regularizer : ?lambda:float -> P.p -> AD.t
end

(** Mini-GRU, 2rd simplification (Heck, 2017) *)
module MGU2 (X : sig
  include Dims_T

  val phi : AD.t -> AD.t
  val d_phi : AD.t -> AD.t
  val sigma : AD.t -> AD.t
  val d_sigma : AD.t -> AD.t
end) : sig
  include T with type 'a P.prm = 'a MGU2_P.prm

  val init : Owl_parameters.setter -> P.p
  val default_regularizer : ?lambda:float -> P.p -> AD.t
end
