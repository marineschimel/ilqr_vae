open Owl_parameters

module type T = sig
  module G : Generative.T
  module P : Owl_parameters.T

  val posterior_mean : ?gen_prms:G.P.p -> P.p -> ([> `o ], G.L.output) Data.t -> AD.t
  val posterior_cov_sample : ?gen_prms:G.P.p -> P.p -> n_samples:int -> AD.t
  val entropy : ?gen_prms:G.P.p -> P.p -> AD.t
end

module ILQR_P = struct
  type 'a prm_ =
    { space_cov : 'a Covariance.P.prm
    ; time_cov : 'a Covariance.P.prm
    }
  [@@deriving accessors ~submodule:A]

  module Make (G : Owl_parameters.T) = struct
    type 'a prm = 'a prm_

    let map ~f prms =
      { space_cov = Covariance.P.map ~f prms.space_cov
      ; time_cov = Covariance.P.map ~f prms.time_cov
      }


    let fold ?prefix ~init ~f prms =
      let w = with_prefix ?prefix in
      let init = Covariance.P.fold ~prefix:(w "space_cov") ~init ~f prms.space_cov in
      Covariance.P.fold ~prefix:(w "time_cov") ~init ~f prms.time_cov
  end
end
