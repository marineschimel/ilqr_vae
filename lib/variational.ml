open Base
open Owl
include Variational_typ
open Covariance
open Priors
open Dynamics
open Likelihoods
open Accessor.O

(* -------------------------------------
   -- iLQR primitive
   ------------------------------------- *)

module ILQR (U : Prior_T) (D : Dynamics_T) (L : Likelihood_T) = struct
  module G = Owl_parameters.Make (Generative_P.Make (U.P) (D.P) (L.P))

  let linesearch = U.requires_linesearch || D.requires_linesearch || L.requires_linesearch

  (* n : dimensionality of state space; m : input dimension *)
  let solve
      ?(conv_threshold = 1E-4)
      ?(n_beg = 1)
      ?saving_iter
      ~u_init
      ~primal'
      ~n
      ~m
      ~n_steps
      ~prms
      data
    =
    let open Generative_P in
    let module M = struct
      type theta = G.p

      let primal' = primal'

      let cost ~theta =
        let cost_lik = L.neg_logp_t ~prms:theta.likelihood in
        let cost_liks =
          Array.init n_steps ~f:(fun k -> cost_lik ~data_t:(L.data_slice ~k data.o))
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
              Array.init n_steps ~f:(fun k -> neg_jac_t ~data_t:(L.data_slice ~k data.o))
            in
            fun ~k ~x ~u:_ ->
              if k < n_beg
              then AD.Mat.zeros 1 n
              else (
                let k = k - n_beg in
                neg_jac_ts.(k) ~k ~z_t:x))


      let rl_xx =
        Option.map L.neg_hess_t ~f:(fun neg_hess_t ~theta ->
            let neg_hess_t = neg_hess_t ~prms:theta.likelihood in
            let neg_hess_ts =
              Array.init n_steps ~f:(fun k -> neg_hess_t ~data_t:(L.data_slice ~k data.o))
            in
            fun ~k ~x ~u:_ ->
              if k < n_beg
              then AD.Mat.zeros n n
              else (
                let k = k - n_beg in
                neg_hess_ts.(k) ~k ~z_t:x))


      let rl_uu =
        Option.map U.neg_hess_t ~f:(fun neg_hess_t ~theta -> neg_hess_t ~prms:theta.prior)


      let rl_ux = Some (fun ~theta:_ ~k:_ ~x:_ ~u:_ -> AD.Mat.zeros m n)
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
    let n_steps = n_steps + n_beg - 1 in
    let module IP =
      Dilqr.Default.Make (struct
        include M

        let n_steps = n_steps + 1
      end)
    in
    let nprev = ref 1E8 in
    let stop_ilqr loss ~prms =
      let x0, theta = AD.Mat.zeros 1 n, prms in
      let cprev = ref 1E9 in
      fun _k us ->
        let c = loss ~theta x0 us in
        let pct_change = Float.(abs ((c -. !cprev) /. !cprev)) in
        cprev := c;
        (* Stdio.printf "\n loss %f || Iter %i \n%!" c _k; *)
        (if Float.(pct_change < conv_threshold) then nprev := Float.(of_int _k));
        Float.(pct_change < conv_threshold)
    in
    let us =
      match u_init with
      | None -> List.init n_steps ~f:(fun _ -> AD.Mat.zeros 1 m)
      | Some us ->
        List.init n_steps ~f:(fun k -> AD.pack_arr (Mat.get_slice [ [ k ] ] us))
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
    let tau = AD.Maths.reshape tau [| n_steps + 1; -1 |] in
    let _ =
      match saving_iter with
      | None -> ()
      | Some file ->
        Mat.save_txt ~out:file ~append:true (Mat.of_array [| !nprev |] 1 (-1))
    in
    AD.Maths.get_slice [ [ 0; -2 ]; [ n; -1 ] ] tau
end

(* -------------------------------------
   -- VAE
   ------------------------------------- *)

module VAE
    (U : Prior_T)
    (D : Dynamics_T)
    (L : Likelihood_T) (X : sig
      val n : int (* state dimension *)

      val m : int (* input dimension *)

      val n_steps : int
      val n_beg : int Option.t
      val diag_time_cov : bool
    end) =
struct
  open X
  module G = Owl_parameters.Make (Generative_P.Make (U.P) (D.P) (L.P))
  module R = Owl_parameters.Make (Recognition_P.Make (U.P) (D.P) (L.P))
  module P = Owl_parameters.Make (VAE_P.Make (U.P) (D.P) (L.P))
  module Integrate = Dynamics.Integrate (D)
  module Ilqr = ILQR (U) (D) (L)
  open VAE_P

  let n_beg = Option.value_map n_beg ~default:1 ~f:(fun i -> i)

  let rec_gen prms =
    match prms.recognition.generative with
    | Some x -> x
    | None -> prms.generative


  let init ?(tie = false) ?(sigma = 1.) gen (set : Owl_parameters.setter) =
    let recognition =
      Recognition_P.
        { generative = (if tie then None else Some gen)
        ; space_cov = Covariance.init ~pin_diag:true ~sigma2:1. set m
        ; time_cov =
            Covariance.init
              ~no_triangle:diag_time_cov
              ~pin_diag:false
              ~sigma2:Float.(square sigma)
              set
              (n_steps + n_beg - 1)
        }
    in
    { generative = gen; recognition }


  let sample_generative ~prms =
    let open Generative_P in
    let u = U.sample ~prms:prms.prior ~n_steps ~m in
    let z = Integrate.integrate ~prms:prms.dynamics ~n ~u:(AD.expand0 u) |> AD.squeeze0 in
    let o = L.sample ~prms:prms.likelihood ~z in
    { u = Some u; z = Some z; o }


  (* NON-DIFFERENTIABLE *)
  let sample_generative_autonomous ~sigma ~prms =
    let open Generative_P in
    let u =
      let u0 = Mat.gaussian ~sigma 1 m in
      let u_rest = Mat.zeros (n_steps - 1) m in
      AD.pack_arr Mat.(u0 @= u_rest)
    in
    let z = Integrate.integrate ~prms:prms.dynamics ~n ~u:(AD.expand0 u) |> AD.squeeze0 in
    let o = L.sample ~prms:prms.likelihood ~z in
    { u = Some u; z = Some z; o }


  let logp ~prms data =
    let prms = prms.generative in
    L.logp ~prms:prms.likelihood ~z:(Option.value_exn data.z) ~data:data.o


  let primal' = G.map ~f:(Owl_parameters.map AD.primal')

  let posterior_mean ?saving_iter ?conv_threshold ~u_init ~prms data =
    Ilqr.solve
      ?saving_iter
      ?conv_threshold
      ~n_beg
      ~u_init
      ~primal'
      ~n
      ~m
      ~n_steps
      ~prms:(rec_gen prms)
      data


  let sample_recognition ~prms =
    let prms = prms.recognition in
    let chol_space = Covariance.to_chol_factor prms.space_cov in
    let chol_time_t = Covariance.to_chol_factor prms.time_cov in
    fun ~mu_u n_samples ->
      let mu_u = AD.Maths.reshape mu_u [| 1; n_steps + n_beg - 1; m |] in
      let xi = AD.Mat.(gaussian Int.(n_samples * (n_steps + n_beg - 1)) m) in
      let z =
        AD.Maths.(xi *@ chol_space)
        |> fun v ->
        AD.Maths.reshape v [| n_samples; n_steps + n_beg - 1; m |]
        |> fun v ->
        AD.Maths.transpose ~axis:[| 1; 0; 2 |] v
        |> fun v ->
        AD.Maths.reshape v [| n_steps + n_beg - 1; -1 |]
        |> fun v ->
        AD.Maths.(transpose chol_time_t *@ v)
        |> fun v ->
        AD.Maths.reshape v [| n_steps + n_beg - 1; n_samples; m |]
        |> fun v -> AD.Maths.transpose ~axis:[| 1; 0; 2 |] v
      in
      AD.Maths.(mu_u + z)


  let predictions ?(pre = true) ~n_samples ~prms mu_u =
    let u = sample_recognition ~prms ~mu_u n_samples in
    let z = Integrate.integrate ~prms:prms.generative.dynamics ~n ~u in
    let z = AD.Maths.get_slice [ []; [ n_beg - 1; -1 ]; [] ] z in
    let u = AD.Maths.get_slice [ []; [ n_beg - 1; -1 ]; [] ] u in
    let o =
      Array.init n_samples ~f:(fun i ->
          let z = AD.Maths.(reshape (get_slice [ [ i ] ] z) [| n_steps; n |]) in
          let o =
            (if pre then L.pre_sample else L.sample) ~prms:prms.generative.likelihood ~z
          in
          o
          |> L.to_mat_list
          |> Array.of_list
          |> Array.map ~f:(fun (label, v) ->
                 label, AD.Maths.reshape v [| 1; AD.Mat.row_num v; AD.Mat.col_num v |]))
    in
    (* for backward compatibility with Marine's previous convention, I need to transpose *)
    let tr = AD.Maths.transpose ~axis:[| 1; 2; 0 |] in
    let o =
      let n_o = Array.length o.(0) in
      Array.init n_o ~f:(fun i ->
          let label, _ = o.(0).(i) in
          ( label
          , o |> Array.map ~f:(fun a -> snd a.(i)) |> AD.Maths.concatenate ~axis:0 |> tr ))
    in
    tr u, tr z, o


  let lik_term ~prms =
    let logp = logp ~prms in
    let dyn = Integrate.integrate ~prms:prms.generative.dynamics in
    fun samples data ->
      let n_samples = (AD.shape samples).(0) in
      let z = dyn ~n ~u:samples in
      let z = AD.Maths.get_slice [ []; [ n_beg - 1; -1 ]; [] ] z in
      let data = { data with z = Some z } in
      AD.Maths.(logp data / F Float.(of_int n_samples))


  let kl_term ~prms =
    match U.kl_to_gaussian with
    | `sampling_based ->
      let logp = U.logp ~prms:prms.generative.prior ~n_steps in
      let logq =
        let c_space = Covariance.to_chol_factor prms.recognition.space_cov in
        let c_time = Covariance.to_chol_factor prms.recognition.time_cov in
        let m_ = AD.Mat.row_num c_space in
        let m = Float.of_int m_ in
        let t = Float.of_int (AD.Mat.row_num c_time) in
        let cst = Float.(m * t * log Const.pi2) in
        let log_det_term =
          let d_space = Owl_parameters.extract prms.recognition.space_cov.d in
          let d_time = Owl_parameters.extract prms.recognition.time_cov.d in
          AD.Maths.(F 2. * ((F m * sum' (log d_time)) + (F t * sum' (log d_space))))
        in
        fun mu_u u ->
          let u_s = AD.shape u in
          assert (Array.length u_s = 3);
          let n_samples = u_s.(0) in
          let du = AD.Maths.(u - AD.expand0 mu_u) in
          (* quadratic term: 
             assuming vec is stacking columns, du = vec(dU) and dU = is T x N
               du^t ((S^t S)⊗(T^t T))^{-1} du
            =  du^t ((S^{-1} S^{-t})⊗(T^{-1} T^{-t})) du 
            =  du^t (S^{-1}⊗T^{-1}) (S^{-t}⊗T^{-t}) du
            =  || (S^{-t}⊗T^{-t}) du ||^2 
            =  || vec(T^{-t} dU S^{-1}) ||^2 *)
          let quadratic_term =
            (* K x T x N *)
            du
            |> AD.Maths.transpose ~axis:[| 1; 0; 2 |]
            |> (fun v -> AD.Maths.reshape v [| n_steps + n_beg - 1; -1 |])
            |> AD.Linalg.linsolve ~typ:`u ~trans:true c_time
            |> (fun v -> AD.Maths.reshape v [| -1; m_ |])
            |> AD.Maths.transpose
            |> AD.Linalg.linsolve ~typ:`u ~trans:true c_space
            |> AD.Maths.l2norm_sqr'
          in
          AD.Maths.(
            F (-0.5)
            * ((F Float.(of_int n_samples) * (F cst + log_det_term)) + quadratic_term))
      in
      fun mu_u u ->
        let u_s = AD.shape u in
        assert (Array.length u_s = 3);
        let n_samples = u_s.(0) in
        (* compute log q(u) - log p(u) *)
        let logqu = logq mu_u u in
        let logpu = logp u in
        AD.Maths.((logqu - logpu) / F Float.(of_int n_samples))
    | `direct f ->
      fun mu_u _ ->
        f
          ~prms:prms.generative.prior
          ~mu:mu_u
          ~space:prms.recognition.space_cov
          ~time:prms.recognition.time_cov


  let elbo ?conv_threshold ~u_init ~n_samples ?(beta = 1.) ~prms =
    let lik_term = lik_term ~prms in
    let kl_term = kl_term ~prms in
    let sample_recognition = sample_recognition ~prms in
    fun data ->
      let mu_u =
        match u_init with
        | `known mu_u -> mu_u
        | `guess u_init -> posterior_mean ?conv_threshold ~u_init ~prms data
      in
      let samples = sample_recognition ~mu_u n_samples in
      let lik_term = lik_term samples data in
      let kl_term = kl_term mu_u samples in
      let elbo = AD.Maths.(lik_term - (F beta * kl_term)) in
      elbo, AD.(unpack_arr (primal' mu_u))


  let elbo_all ~u_init ~n_samples ?beta ~prms data =
    Array.foldi data ~init:(AD.F 0.) ~f:(fun i accu data ->
        let elbo, _ = elbo ~u_init:u_init.(i) ~n_samples ?beta ~prms data in
        AD.Maths.(accu + elbo))


  type u_init =
    [ `known of AD.t option
    | `guess of Mat.mat option
    ]

  let train
      ?(n_samples = fun _ -> 1)
      ?(mini_batch : int Option.t)
      ?max_iter
      ?conv_threshold
      ?(mu_u : u_init Array.t Option.t)
      ?(recycle_u = true)
      ?save_progress_to
      ?in_each_iteration
      ?eta
      ?reg
      ~init_prms
      data
    =
    let n_samples_ = ref (n_samples 1) in
    let n_trials = Array.length data in
    (* make sure all workers have the same data *)
    let data = C.broadcast data in
    (* make sure all workers have different random seeds *)
    C.self_init_rng ();
    let module Packer = Owl_parameters.Packer () in
    let handle = P.pack (module Packer) init_prms in
    let theta, lbound, ubound = Packer.finalize () in
    let theta = AD.unpack_arr theta in
    let us_init =
      match mu_u with
      | Some z -> z
      | None -> Array.create ~len:n_trials (`guess None)
    in
    let adam_loss theta gradient =
      Stdlib.Gc.full_major ();
      let theta = C.broadcast theta in
      let data_batch =
        match mini_batch with
        | None -> data
        | Some size ->
          let ids =
            C.broadcast' (fun () ->
                let ids = Array.mapi data ~f:(fun i _ -> i) in
                Array.permute ids;
                Array.sub ids ~pos:0 ~len:size)
          in
          Array.map ids ~f:(Array.get data)
      in
      let count, loss, g =
        Array.foldi
          data_batch
          ~init:(0, 0., Arr.(zeros (shape theta)))
          ~f:(fun i (accu_count, accu_loss, accu_g) datai ->
            if Int.(i % C.n_nodes = C.rank)
            then (
              try
                let open AD in
                let theta = make_reverse (Arr (Owl.Mat.copy theta)) (AD.tag ()) in
                let prms = P.unpack handle theta in
                let u_init =
                  match us_init.(i) with
                  | `guess z -> `guess z
                  | `known z -> `known (Option.value_exn z)
                in
                let elbo, mu_u =
                  elbo ?conv_threshold ~u_init ~n_samples:!n_samples_ ~prms datai
                in
                if recycle_u
                then (
                  match u_init with
                  | `guess _ -> us_init.(i) <- `guess (Some mu_u)
                  | `known _ -> ());
                let loss = AD.Maths.(neg elbo) in
                (* normalize by the problem size *)
                let loss =
                  AD.Maths.(
                    loss
                    / F
                        Float.(
                          of_int
                            Int.(n_steps * L.size ~prms:init_prms.generative.likelihood)))
                in
                (* optionally add regularizer *)
                let loss =
                  match reg with
                  | None -> loss
                  | Some r -> AD.Maths.(loss + r ~prms)
                in
                reverse_prop (F 1.) loss;
                ( accu_count + 1
                , accu_loss +. unpack_flt loss
                , Owl.Mat.(accu_g + unpack_arr (adjval theta)) )
              with
              | _ ->
                Stdio.printf "Trial %i failed with some exception." i;
                accu_count, accu_loss, accu_g)
            else accu_count, accu_loss, accu_g)
      in
      let total_count = Mpi.reduce_int count Mpi.Int_sum 0 Mpi.comm_world in
      let loss = Mpi.reduce_float loss Mpi.Float_sum 0 Mpi.comm_world in
      Mpi.reduce_bigarray g gradient Mpi.Sum 0 Mpi.comm_world;
      Mat.div_scalar_ gradient Float.(of_int total_count);
      Float.(loss / of_int total_count)
    in
    let stop iter current_loss =
      n_samples_ := n_samples iter;
      Option.iter in_each_iteration ~f:(fun do_this ->
          let prms = P.unpack handle (AD.pack_arr theta) in
          let u_init =
            Array.map us_init ~f:(function
                | `known _ -> None
                | `guess z -> z)
          in
          do_this ~u_init ~prms iter);
      C.root_perform (fun () ->
          Stdio.printf "\r[%05i]%!" iter;
          Option.iter save_progress_to ~f:(fun (loss_every, prms_every, prefix) ->
              let kk = Int.((iter - 1) / loss_every) in
              if Int.((iter - 1) % prms_every) = 0
              then (
                let prefix = Printf.sprintf "%s_%i" prefix kk in
                let prms = P.unpack handle (AD.pack_arr theta) in
                Misc.save_bin ~out:(prefix ^ ".params.bin") prms;
                P.save_to_files ~prefix ~prms);
              if Int.((iter - 1) % loss_every) = 0
              then (
                Stdio.printf "\r[%05i] %.4f%!" iter current_loss;
                let z = [| [| Float.of_int kk; current_loss |] |] in
                Mat.(save_txt ~append:true (of_arrays z) ~out:(prefix ^ ".loss")))));
      match max_iter with
      | Some mi -> iter > mi
      | None -> false
    in
    let _ = Adam.min ?eta ?lb:lbound ?ub:ubound ~stop adam_loss theta in
    theta |> AD.pack_arr |> P.unpack handle


  let recalibrate_uncertainty
      ?n_samples
      ?max_iter
      ?save_progress_to
      ?in_each_iteration
      ?eta
      ~prms
      data
    =
    let n_trials = Array.length data in
    assert (Int.(n_trials % C.n_nodes = 0));
    (* make sure all workers have the same data *)
    let data = C.broadcast data in
    (* make sure all workers have different random seeds *)
    C.self_init_rng ();
    (* get posterior means once and for all *)
    let mu_u =
      Array.mapi data ~f:(fun i data_i ->
          if Int.(i % C.n_nodes = C.rank)
          then `known (Some (posterior_mean ~u_init:None ~prms data_i))
          else `known None)
    in
    (* freeze all parameters except for the posterior uncertainty *)
    let init_prms =
      P.map ~f:Owl_parameters.pin prms
      |> Accessor.map (VAE_P.A.recognition @> Recognition_P.A.space_cov) ~f:(fun _ ->
             prms.recognition.space_cov)
      |> Accessor.map (VAE_P.A.recognition @> Recognition_P.A.time_cov) ~f:(fun _ ->
             prms.recognition.time_cov)
    in
    let recalibrated_prms =
      train
        ?n_samples
        ?max_iter
        ~mu_u
        ?save_progress_to
        ?in_each_iteration
        ?eta
        ~init_prms
        data
    in
    (* pop the uncertainty back in the original prms set *)
    prms
    |> Accessor.map (VAE_P.A.recognition @> Recognition_P.A.space_cov) ~f:(fun _ ->
           recalibrated_prms.recognition.space_cov)
    |> Accessor.map (VAE_P.A.recognition @> Recognition_P.A.time_cov) ~f:(fun _ ->
           recalibrated_prms.recognition.time_cov)


  let check_grad ~prms data n_points file =
    let seed = Random.int 31415 in
    let u_init = Array.map data ~f:(fun _ -> `guess None) in
    let elbo_all ~prms =
      let _ = Owl_stats_prng.init seed in
      elbo_all ~u_init ~n_samples:2 ~beta:1. ~prms
    in
    let module Packer = Owl_parameters.Packer () in
    let handle = P.pack (module Packer) prms in
    let theta, _, _ = Packer.finalize () in
    let theta = AD.unpack_arr theta in
    let loss, true_g =
      let theta = AD.make_reverse (Arr (Mat.copy theta)) (AD.tag ()) in
      let prms = P.unpack handle theta in
      let loss = elbo_all ~prms data in
      AD.reverse_prop (F 1.) loss;
      AD.unpack_flt loss, AD.(unpack_arr (adjval theta))
    in
    let dim = Mat.numel theta in
    let n_points =
      match n_points with
      | `all -> dim
      | `random k -> k
    in
    Array.init dim ~f:(fun i -> i)
    |> Stats.shuffle
    |> Array.sub ~pos:0 ~len:n_points
    |> Array.mapi ~f:(fun k id ->
           Stdio.printf "\rcheck grad: %05i / %05i (out of %i)%!" (k + 1) n_points dim;
           let true_g = Mat.get true_g 0 id in
           let est_g =
             let delta = 1E-6 in
             let theta' = Mat.copy theta in
             Mat.set theta' 0 id (Mat.get theta 0 id +. delta);
             let loss' =
               elbo_all ~prms:(P.unpack handle (Arr theta')) data |> AD.unpack_flt
             in
             Float.((loss' - loss) / delta)
           in
           [| true_g; est_g |])
    |> Mat.of_arrays
    |> Mat.save_txt ~out:file
    |> fun _ -> Stdio.print_endline ""


  let save_data ?prefix data =
    Option.iter data.u ~f:(fun u ->
        Mat.save_txt ~out:(Owl_parameters.with_prefix ?prefix "u") (AD.unpack_arr u));
    Option.iter data.z ~f:(fun z ->
        Mat.save_txt ~out:(Owl_parameters.with_prefix ?prefix "z") (AD.unpack_arr z));
    L.save_data ~prefix:(Owl_parameters.with_prefix ?prefix "o") data.o
end
