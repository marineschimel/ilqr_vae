open Owl
include module type of Variational_typ
open Priors
open Dynamics
open Likelihoods

module ILQR (U : Prior_T) (D : Dynamics_T) (L : Likelihood_T) : sig
  module G : module type of Owl_parameters.Make (Generative_P.Make (U.P) (D.P) (L.P))

  val solve
    :  ?conv_threshold:float
    -> ?n_beg:int
    -> ?saving_iter:string
    -> u_init:Mat.mat option
    -> primal':(G.p -> G.p)
    -> n:int
    -> m:int
    -> n_steps:int
    -> prms:G.p
    -> L.data data
    -> AD.t
end

module VAE
    (U : Prior_T)
    (D : Dynamics_T)
    (L : Likelihood_T) (X : sig
      val n : int
      val m : int
      val n_steps : int
      val n_beg : int Option.t
      val diag_time_cov : bool
    end) : sig
  module G : module type of Owl_parameters.Make (Generative_P.Make (U.P) (D.P) (L.P))
  module R : module type of Owl_parameters.Make (Recognition_P.Make (U.P) (D.P) (L.P))
  module P : module type of Owl_parameters.Make (VAE_P.Make (U.P) (D.P) (L.P))
  open P

  val init : ?tie:bool -> ?sigma:float -> G.p -> Owl_parameters.setter -> P.p
  val sample_generative : prms:G.p -> L.data data
  val sample_generative_autonomous : sigma:float -> prms:G.p -> L.data data

  val posterior_mean
    :  ?saving_iter:string
    -> ?conv_threshold:float
    -> u_init:Mat.mat option
    -> prms:p
    -> L.data data
    -> AD.t

  val sample_recognition : prms:p -> mu_u:AD.t -> int -> AD.t

  val predictions
    :  ?pre:bool
    -> n_samples:int
    -> prms:p
    -> AD.t
    -> AD.t * AD.t * (string * AD.t) Array.t

  val elbo
    :  ?conv_threshold:float
    -> u_init:[ `known of AD.t | `guess of Mat.mat option ]
    -> n_samples:int
    -> ?beta:Float.t
    -> prms:p
    -> L.data data
    -> AD.t * Mat.mat

  type u_init =
    [ `known of AD.t option
    | `guess of Mat.mat option
    ]

  val train
    :  ?n_samples:(int -> int)
    -> ?mini_batch:int
    -> ?max_iter:int
    -> ?conv_threshold:float
    -> ?mu_u:u_init Array.t
    -> ?recycle_u:bool
    -> ?save_progress_to:int * int * string
    -> ?in_each_iteration:(u_init:Mat.mat option Array.t -> prms:p -> int -> unit)
    -> ?eta:[ `constant of float | `of_iter of int -> float ]
    -> ?reg:(prms:p -> AD.t)
    -> init_prms:p
    -> L.data data Array.t
    -> p

  val recalibrate_uncertainty
    :  ?n_samples:(int -> int)
    -> ?max_iter:int
    -> ?save_progress_to:int * int * string
    -> ?in_each_iteration:(u_init:Mat.mat option Array.t -> prms:p -> int -> unit)
    -> ?eta:[ `constant of float | `of_iter of int -> float ]
    -> prms:p
    -> L.data data Array.t
    -> p

  val check_grad
    :  prms:p
    -> L.data data Array.t
    -> [ `all | `random of int ]
    -> string
    -> unit

  val save_data : ?prefix:string -> L.data data -> unit
end
