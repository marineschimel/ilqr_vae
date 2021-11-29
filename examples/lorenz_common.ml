open Base
open Owl
open Ilqr_vae
open Variational

type setup =
  { n : int
  ; m : int
  ; n_trials : int
  ; n_steps : int
  }

module Make_model (P : sig
  val setup : setup
  val n_beg : int Option.t
end) =
struct
  module U = Priors.Student (struct
    let n_beg = P.n_beg
  end)

  module L = Likelihoods.Gaussian (struct
    let label = "o"
    let normalize_c = false
  end)

  module D = Dynamics.MGU2 (struct
    let phi x = AD.Maths.(AD.requad x - F 1.)
    let d_phi = AD.d_requad
    let sigma x = AD.Maths.sigmoid x

    let d_sigma x =
      let tmp = AD.Maths.(exp (neg x)) in
      AD.Maths.(tmp / sqr (F 1. + tmp))


    let n_beg = P.n_beg
  end)

  module Model =
    VAE (U) (D) (L)
      (struct
        let n = P.setup.n
        let m = P.setup.m
        let n_steps = P.setup.n_steps
        let diag_time_cov = false
        let n_beg = P.n_beg
      end)
end

(* output is k x t x 3 *)
let generate ?(sigma = 10.) ?(rho = 28.) ?(beta = 8. /. 3.) ~n_steps n_trials =
  let tt = n_steps in
  let dt = 0.01 in
  let duration = Float.(dt * of_int Int.(tt - 1)) in
  let tspec = Owl_ode.Types.(T1 { t0 = 0.; duration; dt }) in
  let f x _ =
    let x = Mat.get x 0 0
    and y = Mat.get x 0 1
    and z = Mat.get x 0 2 in
    let xdot = sigma *. (y -. x)
    and ydot = (x *. (rho -. z)) -. y
    and zdot = (x *. y) -. (beta *. z) in
    Mat.(of_array [| xdot; ydot; zdot |] 1 3)
  in
  Array.init n_trials ~f:(fun _ ->
      let x0 = Mat.uniform ~a:(-10.) ~b:10. 1 3 in
      let _, xs = Owl_ode.Ode.odeint (module Owl_ode.Native.D.RK4) f x0 tspec () in
      Arr.reshape xs [| 1; n_steps; 3 |])
  |> Arr.concatenate ~axis:0


(* output is k x t x 3 *)
let generate_from_long ?(sigma = 10.) ?(rho = 28.) ?(beta = 8. /. 3.) ~n_steps n_trials =
  let tt = n_trials * n_steps * 100 in
  let dt = 0.01 in
  let duration = Float.(dt * of_int Int.(tt - 1)) in
  let tspec = Owl_ode.Types.(T1 { t0 = 0.; duration; dt }) in
  let f x _ =
    let x = Mat.get x 0 0
    and y = Mat.get x 0 1
    and z = Mat.get x 0 2 in
    let xdot = sigma *. (y -. x)
    and ydot = (x *. (rho -. z)) -. y
    and zdot = (x *. y) -. (beta *. z) in
    Mat.(of_array [| xdot; ydot; zdot |] 1 3)
  in
  let x0 = Mat.gaussian 1 3 in
  let _, xs = Owl_ode.Ode.odeint (module Owl_ode.Native.D.RK4) f x0 tspec () in
  let all = Arr.reshape xs [| 100 * n_trials; n_steps; 3 |] in
  let ids =
    Array.init (100 * n_trials) ~f:(fun i -> i)
    |> Stats.shuffle
    |> Array.sub ~pos:0 ~len:n_trials
    |> Array.to_list
  in
  Arr.get_fancy [ L ids ] all


(* output is t x 3 *)
let continue_from ?(sigma = 10.) ?(rho = 28.) ?(beta = 8. /. 3.) ~n_steps x0 =
  let tt = n_steps in
  let dt = 0.01 in
  let duration = Float.(dt * of_int Int.(tt - 1)) in
  let tspec = Owl_ode.Types.(T1 { t0 = 0.; duration; dt }) in
  let f x _ =
    let x = Mat.get x 0 0
    and y = Mat.get x 0 1
    and z = Mat.get x 0 2 in
    let xdot = sigma *. (y -. x)
    and ydot = (x *. (rho -. z)) -. y
    and zdot = (x *. y) -. (beta *. z) in
    Mat.(of_array [| xdot; ydot; zdot |] 1 3)
  in
  let _, xs = Owl_ode.Ode.odeint (module Owl_ode.Native.D.RK4) f x0 tspec () in
  Arr.reshape xs [| n_steps; 3 |]
