include module type of Dynamics_typ

module Integrate (D : Dynamics_T) : sig
  val integrate : prms:D.P.p -> n:int -> u:AD.t -> AD.t
end

module Linear (X : sig
  val n_beg : int Option.t
end) : sig
  include Dynamics_T with type 'a P.prm = 'a Linear_P.prm

  val init
    :  dt_over_tau:float
    -> alpha:float
    -> beta:float
    -> Owl_parameters.setter
    -> int
    -> int
    -> P.p

  val unpack_a : prms:P.p -> AD.t
end

module Nonlinear (X : sig
  val phi : [ `linear | `nonlinear of (AD.t -> AD.t) * (AD.t -> AD.t) ]
  val n_beg : int Option.t
end) : sig
  include Dynamics_T with type 'a P.prm = 'a Nonlinear_Init_P.prm

  val init
    :  ?radius:float
    -> n:int
    -> m:int
    -> Owl_parameters.setter
    -> Owl_parameters.t P.prm
end

(** Mini-GRU (Heck, 2017) *)
module MGU (X : sig
  val phi : AD.t -> AD.t
  val d_phi : AD.t -> AD.t
  val sigma : AD.t -> AD.t
  val d_sigma : AD.t -> AD.t
  val n_beg : int Option.t
end) : sig
  include Dynamics_T with type 'a P.prm = 'a MGU_P.prm

  val init : n:int -> m:int -> Owl_parameters.setter -> P.p
end

(** Mini-GRU, 2rd simplification (Heck, 2017) *)
module MGU2 (X : sig
  val phi : AD.t -> AD.t
  val d_phi : AD.t -> AD.t
  val sigma : AD.t -> AD.t
  val d_sigma : AD.t -> AD.t
  val n_beg : int Option.t
end) : sig
  include Dynamics_T with type 'a P.prm = 'a MGU2_P.prm

  val init : n:int -> m:int -> Owl_parameters.setter -> P.p
end
