include module type of Vae_typ

module Make (G : Generative.T) (R : Recognition.T with module G = G) :
  T
    with module G = G
     and module R = R
    (* and module U = U
     and module D = D
     and module L = L *)
     and module P = Owl_parameters.Make(P.Make(G.P)(R.P))
