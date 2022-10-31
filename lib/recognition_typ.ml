open Owl_parameters
open Rnn
open Rnn_typ

module type T = sig
  module G : Generative.T
  module P : Owl_parameters.T

  val posterior_mean : ?gen_prms:G.P.p -> P.p -> ('b, G.L.output) Data.t -> AD.t

  val posterior_cov_sample
    :  ?gen_prms:G.P.p
    -> P.p
    -> ('b, G.L.output) Data.t
    -> n_samples:int
    -> AD.t

  val entropy : ?gen_prms:G.P.p -> P.p -> ('b, G.L.output) Data.t -> AD.t
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

module BiRNN_P = struct
  type ('enc, 'con) prm_ =
    { encoder : 'enc
    ; controller : 'con Option.t
    }
  [@@deriving accessors ~submodule:A]

  module Make (Enc : Owl_parameters.T) (Con : Owl_parameters.T) = struct
    type 'a prm = ('a Enc.prm, 'a Con.prm) prm_

    let map ~f prms =
      { encoder = Enc.map ~f prms.encoder
      ; controller = Option.map (fun g -> Con.map ~f g) prms.controller
      }


    let fold ?prefix ~init ~f prms =
      let w = with_prefix ?prefix in
      let init = Enc.fold ~prefix:(w "encoder") ~init ~f prms.encoder in
      match prms.controller with
      | None -> init
      | Some con -> Con.fold ~prefix:(w "controller") ~init ~f con
  end
end
