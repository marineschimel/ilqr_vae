open Base
open Owl
include Recognition_typ
include Encoder_typ
open Rnn
open Owl_parameters

module ILQR
    (U : Prior.T)
    (D : Dynamics.T)
    (L : Likelihood.T) (X : sig
      val conv_threshold : float
      val reuse_u : [ `never | `always | `with_proba of float ]
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


  (* a store for reusing previous solves *)
  let store = Hashtbl.create ~size:10 (module String)

  (* n : dimensionality of state space; m : input dimension *)
  let solve ~primal' ~prms data =
    let o = Data.o data in
    let ext_u = Data.u_ext data in
    let hash = Data.hash data in
    let open Generative.P in
    let module M = struct
      type theta = G.P.p

      let primal' = primal'
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
            let zeros = AD.Mat.zeros 1 n in
            fun ~k ~x ~u:_ ->
              if k < n_beg
              then zeros
              else (
                let k = k - n_beg in
                neg_jac_ts.(k) ~k ~z_t:x))


      let rl_xx =
        Option.map L.neg_hess_t ~f:(fun neg_hess_t ~theta ->
            let neg_hess_t = neg_hess_t ~prms:theta.likelihood in
            let neg_hess_ts =
              Array.init n_steps ~f:(fun k -> neg_hess_t ~o_t:(L.output_slice ~k o))
            in
            let zeros = AD.Mat.zeros n n in
            fun ~k ~x ~u:_ ->
              if k < n_beg
              then zeros
              else (
                let k = k - n_beg in
                neg_hess_ts.(k) ~k ~z_t:x))


      let rl_uu =
        Option.map U.neg_hess_t ~f:(fun neg_hess_t ~theta -> neg_hess_t ~prms:theta.prior)


      let rl_ux =
        let zeros = AD.Mat.zeros m n in
        Some (fun ~theta:_ ~k:_ ~x:_ ~u:_ -> zeros)


      let fl_x =
        let zeros = AD.Mat.zeros 1 n in
        Some (fun ~theta:_ ~k:_ ~x:_ -> zeros)


      let fl_xx =
        let zeros = AD.Mat.zeros n n in
        Some (fun ~theta:_ ~k:_ ~x:_ -> zeros)


      let dyn ~theta =
        let d = D.dyn ~theta:theta.dynamics in
        let driven_d =
          Array.init
            Int.(n_steps_total)
            ~f:(fun k -> d ~ext_u:(Option.map ~f:(AD.Maths.get_slice [ [ k ] ]) ext_u))
        in
        fun ~k ~x ~u -> driven_d.(k) ~k ~x ~u


      let dyn_x =
        Option.map D.dyn_x ~f:(fun d ~theta ->
            let d_t = d ~theta:theta.dynamics in
            let dyn_ts =
              Array.init
                Int.(n_steps_total)
                ~f:(fun k ->
                  d_t ~ext_u:(Option.map ~f:(AD.Maths.get_slice [ [ k ] ]) ext_u))
            in
            fun ~k ~x ~u -> dyn_ts.(k) ~k ~x ~u)


      let dyn_u =
        Option.map D.dyn_u ~f:(fun d ~theta ->
            let d_t = d ~theta:theta.dynamics in
            let dyn_ts =
              Array.init
                Int.(n_steps_total)
                ~f:(fun k ->
                  d_t ~ext_u:(Option.map ~f:(AD.Maths.get_slice [ [ k ] ]) ext_u))
            in
            fun ~k ~x ~u -> dyn_ts.(k) ~k ~x ~u)


      let running_loss ~theta =
        let cost_o =
          let c = L.neg_logp_t ~prms:theta.likelihood in
          Array.init n_steps ~f:(fun k -> c ~o_t:(L.output_slice ~k o))
        in
        let cost_u = U.neg_logp_t ~prms:theta.prior in
        fun ~k ~x ~u ->
          let cost_o =
            if k < n_beg then AD.F 0. else cost_o.(k - n_beg) ~k:(k - n_beg) ~z_t:x
          in
          let cost_u = cost_u ~k ~x ~u in
          AD.Maths.(cost_u + cost_o)


      let final_loss ~theta:_ ~k:_ ~x:_ = AD.F 0.
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
        Float.(pct_change < X.conv_threshold || Int.(_k > 10))
    in
    let us =
      let no_reuse () = List.init n_steps_total ~f:(fun _ -> AD.Mat.zeros 1 m) in
      let reuse () =
        match Hashtbl.find store hash with
        | Some u ->
          List.init n_steps_total ~f:(fun k -> AD.pack_arr (Mat.get_slice [ [ k ] ] u))
        | None -> no_reuse ()
      in
      match X.reuse_u with
      | `always -> reuse ()
      | `never -> no_reuse ()
      | `with_proba p ->
        if Float.(Stats.uniform_rvs ~a:0. ~b:1. < p) then reuse () else no_reuse ()
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
    let tau = AD.Maths.reshape tau [| n_steps_total + 1; n + m |] in
    let u = AD.Maths.get_slice [ [ 0; -2 ]; [ n; -1 ] ] tau in
    (* cache the solution for potential later reuse as initial condition *)
    if Poly.(X.reuse_u <> `never)
    then (
      Hashtbl.remove store hash;
      Hashtbl.add_exn store ~key:hash ~data:(AD.unpack_arr u));
    u


  (* TODO: still have to implement re-use of [u_init] in this new functor interface... *)
  let posterior_mean ?gen_prms _ =
    let prms = Option.value_exn gen_prms in
    solve ~primal':(G.P.map ~f:(Owl_parameters.map AD.primal')) ~prms


  (* returns K x T x M array *)
  let posterior_cov_sample ?gen_prms:_ prms _ =
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


  let entropy ?gen_prms:_ prms _ =
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

module BiRNN
    (Enc : Encoder_T)
    (Con : Encoder_T) (X : sig
      val n_steps : int
      val n : int
      val m : int
    end)
    (U : Prior.T)
    (D : Dynamics.T)
    (L : Likelihood.T) =
struct
  open BiRNN_P
  module G = Generative.Make (U) (D) (L)
  module P = Owl_parameters.Make (BiRNN_P.Make (Enc.P) (Con.P))

  let n_beg =
    assert (X.n % X.m = 0);
    X.n / X.m


  let n_steps_total = X.n_steps + n_beg - 1

  let extract_and_pad data =
    let data = L.extract (Data.o data) in
    let n_col = AD.Mat.col_num data in
    AD.Maths.concatenate ~axis:0 [| AD.Mat.zeros Int.(n_beg - 1) n_col; data |]


  let posterior_us ~prms data =
    match prms.controller with
    | Some prms -> Con.encode ~prms ~input:data
    | None ->
      AD.Mat.zeros n_steps_total X.n, AD.Maths.(F 0.001 * AD.Mat.(ones n_steps_total X.n))


  let posterior_x0s ~prms data =
    let prms = prms.encoder in
    Enc.encode ~prms ~input:data


  let posterior_mean ?gen_prms:_ rec_prms data =
    let d = extract_and_pad data in
    let mean_x0, _ = posterior_x0s ~prms:rec_prms d in
    let mean_us, _ = posterior_us ~prms:rec_prms d in
    AD.Maths.(mean_x0 + mean_us)


  (* returns K x T x M array *)
  let posterior_cov_sample ?gen_prms:_ rec_prms data ~n_samples =
    let d = extract_and_pad data in
    let _, _std_x0 = posterior_x0s ~prms:rec_prms d in
    let _, std_u = posterior_us ~prms:rec_prms d in
    let std_u =
      AD.Maths.split ~axis:0 (Array.init (AD.Mat.row_num std_u) ~f:(fun _ -> 1)) std_u
    in
    let z =
      Array.map std_u ~f:(fun ell_t ->
          let z_t = AD.Maths.(diagm ell_t *@ AD.Mat.gaussian X.m n_samples) in
          let z_t = AD.Maths.transpose z_t in
          AD.Maths.reshape z_t [| 1; n_samples; X.m |])
      |> AD.Maths.concatenate ~axis:0
    in
    let z = AD.Maths.transpose ~axis:[| 1; 0; 2 |] z in
    z


  (*actually still need to add the sample for z0*)

  let entropy ?gen_prms:_ prms data =
    let _, _std_x0 = posterior_x0s ~prms (L.extract (Data.o data)) in
    let _, std_u = posterior_us ~prms (L.extract (Data.o data)) in
    let mm = Float.of_int X.m in
    let tt = Float.of_int n_steps_total in
    let dim = Float.(mm * tt) in
    let std_u =
      AD.Maths.split ~axis:0 (Array.init (AD.Mat.row_num std_u) ~f:(fun _ -> 1)) std_u
    in
    let log_det_term =
      Array.fold std_u ~init:(AD.F 0.) ~f:(fun accu x ->
          AD.Maths.(accu + (F 2. * sum' (log (diag x)))))
    in
    AD.Maths.(F 0.5 * (log_det_term + (F dim * F Float.(1. + log Const.pi2))))
end
