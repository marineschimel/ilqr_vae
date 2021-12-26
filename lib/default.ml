open Vae

module MGU_funs = struct
  let phi x = AD.Maths.(AD.requad x - F 1.)
  let d_phi = AD.d_requad
  let sigma x = AD.Maths.sigmoid x

  let d_sigma x =
    let tmp = AD.Maths.(exp (neg x)) in
    AD.Maths.(tmp / sqr (F 1. + tmp))
end

module Link_funs = struct
  let link_function = AD.requad
  let d_link_function = AD.d_requad
  let d2_link_function = AD.d2_requad
end

(* Default iLQR-VAE model *)
module Model
    (U : Prior.T)
    (D : Dynamics.T)
    (L : Likelihood.T) (S : sig
      val n_steps : int
    end) =
struct
  module G = Generative.Make (U) (D) (L)

  module R =
    Recognition.ILQR (U) (D) (L)
      (struct
        let conv_threshold = 1E-4
        let diag_time_cov = false
        let n_steps = S.n_steps
      end)

  module Model = Make (G) (R)
end

