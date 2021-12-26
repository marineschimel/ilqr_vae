include module type of Recognition_typ

module ILQR
    (U : Prior.T)
    (D : Dynamics.T)
    (L : Likelihood.T) (X : sig
      val conv_threshold : float
      val diag_time_cov : bool
      val n_steps : int
    end) : sig
  include
    T
      with module G = Generative.Make(U)(D)(L)
       and module P = Owl_parameters.Make(ILQR_P.Make(Generative.Make(U)(D)(L).P))

  val init : ?sigma:float -> Owl_parameters.setter -> P.p
end
