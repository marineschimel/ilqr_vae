open Base
open Owl_parameters
open Covariance

(** Data type *)

type 'a data =
  { u : AD.t option
  ; z : AD.t option
  ; o : 'a
  }

(** Modules that encapsulate parameter types *)

module Generative_P = struct
  type ('u, 'd, 'l) prm_ =
    { prior : 'u
    ; dynamics : 'd
    ; likelihood : 'l
    }
  [@@deriving accessors ~submodule:A]

  module Make (U : Owl_parameters.T) (D : Owl_parameters.T) (L : Owl_parameters.T) =
  struct
    type 'a prm = ('a U.prm, 'a D.prm, 'a L.prm) prm_

    let map ~f prms =
      { prior = U.map ~f prms.prior
      ; dynamics = D.map ~f prms.dynamics
      ; likelihood = L.map ~f prms.likelihood
      }


    let fold ?prefix ~init ~f prms =
      let w = with_prefix ?prefix in
      let init = U.fold ~prefix:(w "prior") ~init ~f prms.prior in
      let init = D.fold ~prefix:(w "dynamics") ~init ~f prms.dynamics in
      L.fold ~prefix:(w "likelihood") ~init ~f prms.likelihood
  end
end

module Recognition_P = struct
  type ('a, 'u, 'd, 'l) prm_ =
    { generative : ('u, 'd, 'l) Generative_P.prm_ Option.t
    ; space_cov : 'a Covariance.P.prm
    ; time_cov : 'a Covariance.P.prm
    }
  [@@deriving accessors ~submodule:A]

  module Make (U : Owl_parameters.T) (D : Owl_parameters.T) (L : Owl_parameters.T) =
  struct
    module G = Generative_P.Make (U) (D) (L)

    type 'a prm = ('a, 'a U.prm, 'a D.prm, 'a L.prm) prm_

    let map ~f prms =
      { generative = Option.map prms.generative ~f:(fun g -> G.map ~f g)
      ; space_cov = Covariance.P.map ~f prms.space_cov
      ; time_cov = Covariance.P.map ~f prms.time_cov
      }


    let fold ?prefix ~init ~f prms =
      let w = with_prefix ?prefix in
      let init =
        match prms.generative with
        | None -> init
        | Some gen -> G.fold ~prefix:(w "generative") ~init ~f gen
      in
      let init = Covariance.P.fold ~prefix:(w "space_cov") ~init ~f prms.space_cov in
      Covariance.P.fold ~prefix:(w "time_cov") ~init ~f prms.time_cov
  end
end

module VAE_P = struct
  type ('a, 'u, 'd, 'l) prm_ =
    { generative : ('u, 'd, 'l) Generative_P.prm_
    ; recognition : ('a, 'u, 'd, 'l) Recognition_P.prm_
    }
  [@@deriving accessors ~submodule:A]

  module Make (U : Owl_parameters.T) (D : Owl_parameters.T) (L : Owl_parameters.T) =
  struct
    module G = Generative_P.Make (U) (D) (L)
    module R = Recognition_P.Make (U) (D) (L)

    type 'a prm = ('a, 'a U.prm, 'a D.prm, 'a L.prm) prm_

    let map ~f prms =
      { generative = G.map ~f prms.generative; recognition = R.map ~f prms.recognition }


    let fold ?prefix ~init ~f prms =
      let w = with_prefix ?prefix in
      G.fold ~prefix:(w "generative") ~init ~f prms.generative
      |> fun init -> R.fold ~prefix:(w "recognition") ~init ~f prms.recognition
  end
end
