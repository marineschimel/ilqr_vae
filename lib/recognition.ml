open Base
open Owl
include Recognition_typ

module ILQR
    (U : Prior.T)
    (D : Dynamics.T)
    (L : Likelihood.T) (X : sig
      val conv_threshold : float
      val diag_time_cov : bool
      val n_steps : int
    end) =
struct
  let _ = assert (U.m = D.m)

  module L = L
  module G = Generative.Make (U) (D) (L)
  module P = Owl_parameters.Make (ILQR_P.Make (G.P))
  open ILQR_P

  let n = D.n
  and m = D.m

  let n_steps = X.n_steps

  let n_beg =
    assert (D.n % D.m = 0);
    n / m


  let n_steps_total = n_steps + n_beg - 1
  let linesearch = U.requires_linesearch || D.requires_linesearch || L.requires_linesearch

  let init ?(sigma = 1.) (set : Owl_parameters.setter) =
    { space_cov = Covariance.init ~pin_diag:true ~sigma2:1. set m
    ; time_cov =
        Covariance.init
          ~no_triangle:X.diag_time_cov
          ~pin_diag:false
          ~sigma2:Float.(square sigma)
          set
          n_steps_total
    }


  (* n : dimensionality of state space; m : input dimension *)
  let solve ?u_init ~primal' ~prms data =
    let o = Data.o data in
    let open Generative.P in
    let module M = struct
      type theta = G.P.p

      let primal' = primal'

      let cost ~theta =
        let cost_lik = L.neg_logp_t ~prms:theta.likelihood in
        let cost_liks =
          Array.init n_steps ~f:(fun k -> cost_lik ~o_t:(L.output_slice ~k o))
        in
        let cost_u = U.neg_logp_t ~prms:theta.prior in
        fun ~k ~x ~u ->
          let cost_lik =
            if k < n_beg then AD.F 0. else cost_liks.(k - n_beg) ~k:(k - n_beg) ~z_t:x
          in
          let cost_u = cost_u ~k ~x ~u in
          AD.Maths.(cost_u + cost_lik)


      let m = m
      let n = n

      let rl_u =
        Option.map U.neg_jac_t ~f:(fun neg_jac_t ~theta -> neg_jac_t ~prms:theta.prior)


      let rl_x =
        Option.map L.neg_jac_t ~f:(fun neg_jac_t ~theta ->
            let neg_jac_t = neg_jac_t ~prms:theta.likelihood in
            let neg_jac_ts =
              Array.init n_steps ~f:(fun k -> neg_jac_t ~o_t:(L.output_slice ~k o))
            in
            let tmp = AD.Mat.zeros 1 n in
            fun ~k ~x ~u:_ ->
              if k < n_beg
              then tmp
              else (
                let k = k - n_beg in
                neg_jac_ts.(k) ~k ~z_t:x))


      let rl_xx =
        Option.map L.neg_hess_t ~f:(fun neg_hess_t ~theta ->
            let neg_hess_t = neg_hess_t ~prms:theta.likelihood in
            let neg_hess_ts =
              Array.init n_steps ~f:(fun k -> neg_hess_t ~o_t:(L.output_slice ~k o))
            in
            let tmp = AD.Mat.zeros n n in
            fun ~k ~x ~u:_ ->
              if k < n_beg
              then tmp
              else (
                let k = k - n_beg in
                neg_hess_ts.(k) ~k ~z_t:x))


      let rl_uu =
        Option.map U.neg_hess_t ~f:(fun neg_hess_t ~theta -> neg_hess_t ~prms:theta.prior)


      let rl_ux =
        let tmp = AD.Mat.zeros m n in
        Some (fun ~theta:_ ~k:_ ~x:_ ~u:_ -> tmp)


      let final_cost ~theta:_ ~k:_ ~x:_ = AD.F 0.

      let fl_x =
        let z = AD.Mat.zeros 1 n in
        Some (fun ~theta:_ ~k:_ ~x:_ -> z)


      let fl_xx =
        let z = AD.Mat.zeros n n in
        Some (fun ~theta:_ ~k:_ ~x:_ -> z)


      let dyn ~theta = D.dyn ~theta:theta.dynamics
      let dyn_x = Option.map D.dyn_x ~f:(fun d ~theta -> d ~theta:theta.dynamics)
      let dyn_u = Option.map D.dyn_u ~f:(fun d ~theta -> d ~theta:theta.dynamics)
      let running_loss = cost
      let final_loss = final_cost
    end
    in
    let module IP =
      Dilqr.Default.Make (struct
        include M

        let n_steps = n_steps_total + 1
      end)
    in
    let stop_ilqr loss ~prms =
      let x0, theta = AD.Mat.zeros 1 n, prms in
      let cprev = ref 1E9 in
      fun _k us ->
        let c = loss ~theta x0 us in
        let pct_change = Float.(abs ((c -. !cprev) /. !cprev)) in
        cprev := c;
        Float.(pct_change < X.conv_threshold)
    in
    let us =
      match u_init with
      | None -> List.init n_steps_total ~f:(fun _ -> AD.Mat.zeros 1 m)
      | Some us ->
        List.init n_steps_total ~f:(fun k -> AD.pack_arr (Mat.get_slice [ [ k ] ] us))
    in
    (*
        u0        u1  u2 ......   uT
        x0 = 0    x1  x2 ......   xT xT+1
    *)
    let tau =
      IP.ilqr
        ~linesearch
        ~stop:(stop_ilqr IP.loss ~prms)
        ~us
        ~x0:(AD.Mat.zeros 1 n)
        ~theta:prms
        ()
    in
    let tau = AD.Maths.reshape tau [| n_steps_total + 1; -1 |] in
    AD.Maths.get_slice [ [ 0; -2 ]; [ n; -1 ] ] tau


  (* TODO: still have to implement re-use of [u_init] in this new functor interface... *)
  let posterior_mean ?gen_prms _ =
    let prms = Option.value_exn gen_prms in
    solve ?u_init:None ~primal':(G.P.map ~f:(Owl_parameters.map AD.primal')) ~prms


  (* returns K x T x M array *)
  let posterior_cov_sample ?gen_prms:_ prms =
    let chol_space = Covariance.to_chol_factor prms.space_cov in
    let chol_time_t = Covariance.to_chol_factor prms.time_cov in
    fun ~n_samples ->
      let xi = AD.Mat.(gaussian Int.(n_samples * n_steps_total) m) in
      AD.Maths.(xi *@ chol_space)
      |> fun v ->
      AD.Maths.reshape v [| n_samples; n_steps_total; m |]
      |> fun v ->
      AD.Maths.transpose ~axis:[| 1; 0; 2 |] v
      |> fun v ->
      AD.Maths.reshape v [| n_steps_total; -1 |]
      |> fun v ->
      AD.Maths.(transpose chol_time_t *@ v)
      |> fun v ->
      AD.Maths.reshape v [| n_steps_total; n_samples; m |]
      |> fun v -> AD.Maths.transpose ~axis:[| 1; 0; 2 |] v


  let entropy ?gen_prms:_ prms =
    let mm = Float.of_int m in
    let tt = Float.of_int n_steps_total in
    let dim = Float.(mm * tt) in
    let d_space = Owl_parameters.extract prms.space_cov.d in
    let d_time = Owl_parameters.extract prms.time_cov.d in
    let log_det =
      AD.Maths.(F 2. * ((F mm * sum' (log d_time)) + (F tt * sum' (log d_space))))
    in
    AD.Maths.(F 0.5 * (log_det + (F dim * F Float.(1. + log Const.pi2))))
end

