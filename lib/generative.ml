open Base
include Generative_typ

module Make (U : Prior.T) (D : Dynamics.T) (L : Likelihood.T) = struct
  let _ = assert (D.m = U.m)

  module U = U
  module D = D
  module L = L
  module P = Owl_parameters.Make (P.Make (U.P) (D.P) (L.P))
  module Integrate = Dynamics.Integrate (D)
  open Generative_typ.P

  let n = D.n
  and m = D.m

  let n_beg =
    assert (D.n % D.m = 0);
    n / m


  let integrate ~prms ~u = Integrate.integrate ~prms:prms.dynamics ~u

  let sample ~prms ?(id = 0) ?(pre = false) =
    let integrate = integrate ~prms in
    fun u ->
      let u =
        match u with
        | `prior n_steps -> U.sample ~prms:prms.prior ~n_steps
        | `some u -> u
      in
      let z =
        let u = AD.expand_to_3d u in
        integrate ~u |> AD.squeeze_from_3d
      in
      let o = (if pre then L.pre_sample else L.sample) ~prms:prms.likelihood ~z in
      Data.pack ~id o |> Data.fill ~u ~z


  let log_prior ~prms u = U.logp ~prms:prms.prior u

  let log_likelihood ~prms data =
    L.logp ~prms:prms.likelihood ~z:(Data.z data) ~o:(Data.o data)
end
