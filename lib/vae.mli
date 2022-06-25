include module type of Vae_typ

module Make (G : Generative.T) (R : Recognition.T with module G = G) : sig
  module G :
    module type of G
      with module P = G.P
       and module U = G.U
       and module D = G.D
       and module L = G.L

  module R : module type of R with module P = R.P and module G = G
  module P : module type of Owl_parameters.Make (P.Make (G.P) (R.P))
  open P

  val init : G.P.p -> R.P.p -> P.p

  val posterior_predictive_sample
    :  ?id:int
    -> ?pre:bool
    -> prms:P.p
    -> ([ `o ], G.L.output) Data.t
    -> AD.t * (int -> ([ `o | `uz ], G.L.output) Data.t Array.t)

  val elbo : prms:P.p -> n_posterior_samples:int -> ([> `o ], G.L.output) Data.t -> AD.t

  val train
    :  ?n_posterior_samples:(int -> int)
    -> ?mini_batch:int
    -> ?max_iter:int
    -> ?save_progress_to:int * int * string
    -> ?zip:bool
    -> ?in_each_iteration:(prms:p -> int -> unit)
    -> ?learning_rate:[ `constant of float | `of_iter of int -> float ]
    -> ?regularizer:(prms:p -> AD.t)
    -> init_prms:p
    -> ([> `o ], G.L.output) Data.t Array.t
    -> p

  val save_results
    :  ?zip:bool
    -> prefix:String.t
    -> prms:P.p
    -> ([> `o ], G.L.output) Data.t Array.t
    -> unit
end
