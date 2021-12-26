include module type of Generative_typ

module Make (U : Prior.T) (D : Dynamics.T) (L : Likelihood.T) :
  T
    with module U = U
     and module D = D
     and module L = L
     and module P = Owl_parameters.Make(P.Make(U.P)(D.P)(L.P))

