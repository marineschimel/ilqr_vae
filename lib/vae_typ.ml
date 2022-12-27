open Base
open Owl_parameters

module P = struct
  type ('g, 'r) prm_ =
    { generative : 'g
    ; recognition : 'r
    }
  [@@deriving accessors ~submodule:A]

  module Make (G : Owl_parameters.T) (R : Owl_parameters.T) = struct
    type 'a prm = ('a G.prm, 'a R.prm) prm_

    let map ~f prms =
      { generative = G.map ~f prms.generative; recognition = R.map ~f prms.recognition }


    let fold ?prefix ~init ~f prms =
      let w = with_prefix ?prefix in
      G.fold ~prefix:(w "generative") ~init ~f prms.generative
      |> fun init -> R.fold ~prefix:(w "recognition") ~init ~f prms.recognition
  end
end

module type T = sig
  module G : Generative.T
  module R : Recognition.T
  module P : Owl_parameters.T

  (* 
  module U : Prior.T
  module D : Dynamics.T *)
  module U = G.U
  module D = G.D
  module L = G.L
  open P

  val init : G.P.p -> R.P.p -> P.p

  val posterior_predictive_sample
    :  ?id:int
    -> ?pre:bool
    -> prms:P.p
    -> ([ `o ], L.output) Data.t
    -> AD.t * (int -> ([ `o | `uz ], G.L.output) Data.t Array.t)

  val elbo : prms:P.p -> n_posterior_samples:int -> ([> `o ], L.output) Data.t -> AD.t

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
    -> ([> `o ], L.output) Data.t Array.t
    -> p

  val save_results
    :  ?zip:bool
    -> prefix:String.t
    -> prms:P.p
    -> n_to_save:int
    -> ([> `o ], L.output) Data.t Array.t
    -> unit
end
