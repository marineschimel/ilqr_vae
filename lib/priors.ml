open Base
open Owl
open Covariance
include Priors_typ

(*
module Gaussian = struct
  module P = Owl_parameters.Make (Gaussian_P)
  open Gaussian_P

  let requires_linesearch = false

  let init ?(spatial_std = 1.) ?(first_bin = 1.) ~m (set : Owl_parameters.setter) =
    let spatial_stds = Mat.create 1 m spatial_std in
    { spatial_stds = set ~above:1E-3 (AD.pack_arr spatial_stds)
    ; first_bin = set ~above:1E-5 (AD.F first_bin)
    }


  let spatial_stds ~prms = Owl_parameters.extract prms.spatial_stds

  (* returns a column vector *)
  let temporal_stds ~prms ~n_steps =
    let fb = Owl_parameters.extract prms.first_bin in
    let t1 = AD.Maths.(fb * AD.Mat.ones 1 1) in
    let t2 = AD.Mat.ones (n_steps - 1) 1 in
    AD.Maths.concat ~axis:0 t1 t2


  let kl_to_gaussian =
    let kl ~prms ~mu ~space ~time =
      let ell_q_s = Covariance.to_chol_factor space in
      let ell_q_t = Covariance.to_chol_factor time in
      let m = AD.Mat.row_num ell_q_s in
      let t = AD.Mat.row_num ell_q_t in
      let mm = Float.of_int m in
      let tt = Float.of_int t in
      let dim = Float.(mm * tt) in
      let ell_p_s = Owl_parameters.extract prms.spatial_stds in
      let ell_p_t = temporal_stds ~prms ~n_steps:t in
      let quadratic_term = AD.Maths.(l2norm_sqr' (F 1. / ell_p_t * mu / ell_p_s)) in
      let trace_term =
        let s = AD.Maths.(l2norm_sqr' (ell_q_s / ell_p_s)) in
        let t = AD.Maths.(l2norm_sqr' (ell_q_t / transpose ell_p_t)) in
        AD.Maths.(s * t)
      in
      let logdet_term =
        let log_det_q =
          let d_space = Owl_parameters.extract space.d in
          let d_time = Owl_parameters.extract time.d in
          AD.Maths.(F 2. * ((F mm * sum' (log d_time)) + (F tt * sum' (log d_space))))
        in
        let log_det_p =
          let d_space = ell_p_s in
          let d_time = ell_p_t in
          AD.Maths.(F 2. * ((F mm * sum' (log d_time)) + (F tt * sum' (log d_space))))
        in
        AD.Maths.(log_det_p - log_det_q)
      in
      AD.Maths.(F 0.5 * (trace_term + logdet_term + quadratic_term - F dim))
    in
    `direct kl


  let sample ~prms ~n_steps ~m =
    let ell_t = temporal_stds ~prms ~n_steps in
    let ell_s = Owl_parameters.extract prms.spatial_stds in
    let xi = AD.(Mat.gaussian n_steps m) in
    AD.Maths.(ell_t * xi * ell_s)


  let neg_logp_t ~prms =
    let fb = Owl_parameters.extract prms.first_bin in
    let ell_s = Owl_parameters.extract prms.spatial_stds in
    let m = AD.Mat.numel ell_s in
    let cst = Float.(of_int m * log Const.pi2) in
    fun ~k ~x:_ ~u ->
      let sigma = if k = 0 then AD.Maths.(fb * ell_s) else ell_s in
      AD.Maths.(F 0.5 * (F cst + (F 2. * sum' (log sigma)) + l2norm_sqr' (u / sigma)))


  let neg_jac_t =
    let jac_t ~prms =
      let fb = Owl_parameters.extract prms.first_bin in
      let ell_s = Owl_parameters.extract prms.spatial_stds in
      fun ~k ~x:_ ~u ->
        let sigma = if k = 0 then AD.Maths.(fb * ell_s) else ell_s in
        AD.Maths.(u / sqr sigma)
    in
    Some jac_t


  let neg_hess_t =
    let hess_t ~prms =
      let fb = Owl_parameters.extract prms.first_bin in
      let ell_p_s = Owl_parameters.extract prms.spatial_stds in
      fun ~k ~x:_ ~u:_ ->
        let sigma = if k = 0 then AD.Maths.(fb * ell_p_s) else ell_p_s in
        AD.Maths.(diagm (F 1. / sqr sigma))
    in
    Some hess_t


  (* shouldn't ever need this *)
  let logp ~prms:_ = assert false
end
*)

module Gaussian (X : sig
  val n_beg : int Option.t
end) =
struct
  module P = Owl_parameters.Make (Gaussian_P)
  open Gaussian_P

  let requires_linesearch = false

  let init ?(spatial_std = 1.) ?(first_bin = 1.) ~m (set : Owl_parameters.setter) =
    let spatial_stds = Mat.create 1 m spatial_std in
    { spatial_stds = set ~above:1E-3 (AD.pack_arr spatial_stds)
    ; first_bin = set ~above:1E-5 (AD.F first_bin)
    }


  let spatial_stds ~prms = Owl_parameters.extract prms.spatial_stds
  let n_beg = Option.value_map X.n_beg ~default:1 ~f:(fun i -> i)

  (* returns a column vector *)
  let temporal_stds ~prms ~n_steps =
    let fb = Owl_parameters.extract prms.first_bin in
    let t1 = AD.Maths.(fb * AD.Mat.ones n_beg 1) in
    let t2 = AD.Mat.ones (n_steps - n_beg) 1 in
    AD.Maths.concat ~axis:0 t1 t2


  let kl_to_gaussian =
    let kl ~prms ~mu ~space ~time =
      let ell_q_s = Covariance.to_chol_factor space in
      let ell_q_t = Covariance.to_chol_factor time in
      let m = AD.Mat.row_num ell_q_s in
      let t = AD.Mat.row_num ell_q_t in
      let mm = Float.of_int m in
      let tt = Float.of_int t in
      let dim = Float.(mm * tt) in
      let ell_p_s = Owl_parameters.extract prms.spatial_stds in
      let ell_p_t = temporal_stds ~prms ~n_steps:t in
      let quadratic_term = AD.Maths.(l2norm_sqr' (F 1. / ell_p_t * mu / ell_p_s)) in
      let trace_term =
        let s = AD.Maths.(l2norm_sqr' (ell_q_s / ell_p_s)) in
        let t = AD.Maths.(l2norm_sqr' (ell_q_t / transpose ell_p_t)) in
        AD.Maths.(s * t)
      in
      let logdet_term =
        let log_det_q =
          let d_space = Owl_parameters.extract space.d in
          let d_time = Owl_parameters.extract time.d in
          AD.Maths.(F 2. * ((F mm * sum' (log d_time)) + (F tt * sum' (log d_space))))
        in
        let log_det_p =
          let d_space = ell_p_s in
          let d_time = ell_p_t in
          AD.Maths.(F 2. * ((F mm * sum' (log d_time)) + (F tt * sum' (log d_space))))
        in
        AD.Maths.(log_det_p - log_det_q)
      in
      AD.Maths.(F 0.5 * (trace_term + logdet_term + quadratic_term - F dim))
    in
    `direct kl


  let sample ~prms ~n_steps ~m =
    let ell_t = temporal_stds ~prms ~n_steps in
    let ell_s = Owl_parameters.extract prms.spatial_stds in
    let xi = AD.(Mat.gaussian n_steps m) in
    AD.Maths.(ell_t * xi * ell_s)


  let neg_logp_t ~prms =
    let fb = Owl_parameters.extract prms.first_bin in
    let ell_s = Owl_parameters.extract prms.spatial_stds in
    let m = AD.Mat.numel ell_s in
    let cst = Float.(of_int m * log Const.pi2) in
    fun ~k ~x:_ ~u ->
      let sigma = if k < n_beg then AD.Maths.(fb * ell_s) else ell_s in
      AD.Maths.(F 0.5 * (F cst + (F 2. * sum' (log sigma)) + l2norm_sqr' (u / sigma)))


  let neg_jac_t =
    let jac_t ~prms =
      let fb = Owl_parameters.extract prms.first_bin in
      let ell_s = Owl_parameters.extract prms.spatial_stds in
      fun ~k ~x:_ ~u ->
        let sigma = if k < n_beg then AD.Maths.(fb * ell_s) else ell_s in
        AD.Maths.(u / sqr sigma)
    in
    Some jac_t


  let neg_hess_t =
    let hess_t ~prms =
      let fb = Owl_parameters.extract prms.first_bin in
      let ell_p_s = Owl_parameters.extract prms.spatial_stds in
      fun ~k ~x:_ ~u:_ ->
        let sigma = if k < n_beg then AD.Maths.(fb * ell_p_s) else ell_p_s in
        AD.Maths.(diagm (F 1. / sqr sigma))
    in
    Some hess_t


  (* shouldn't ever need this *)
  let logp ~prms:_ = assert false
end

module Student (X : sig
  val n_beg : int Option.t
end) =
struct
  module P = Owl_parameters.Make (Student_P)
  open Student_P

  let requires_linesearch = true

  let init
      ?(pin_std = false)
      ?(spatial_std = 1.)
      ?(nu = 10.)
      ~m
      (set : Owl_parameters.setter)
    =
    let spatial_stds = Mat.create 1 m spatial_std in
    { spatial_stds =
        (if pin_std then Owl_parameters.pinned else set)
          ~above:1E-3
          (AD.pack_arr spatial_stds)
    ; nu = set ~above:2.0 (AD.F nu)
    ; first_step = set ~above:1E-3 (AD.pack_arr spatial_stds)
    }


  let spatial_stds ~prms = Owl_parameters.extract prms.spatial_stds
  let kl_to_gaussian = `sampling_based

  let get_eff_prms ~prms =
    let s = Owl_parameters.extract prms.spatial_stds in
    let nu = Owl_parameters.extract prms.nu in
    let sigma = AD.Maths.(sqrt ((nu - F 2.) / nu) * s) in
    nu, sigma


  (* non-differentiable *)
  let sample ~prms ~n_steps ~m =
    let nu, sigma = get_eff_prms ~prms in
    let i =
      match X.n_beg with
      | None -> 1
      | Some i -> i
    in
    let sigma0 = Owl_parameters.extract prms.first_step in
    let xi = Mat.(gaussian Int.(n_steps - i) m * AD.unpack_arr sigma) in
    let u = Stats.chi2_rvs ~df:(AD.unpack_flt nu) in
    let z = Float.(sqrt (AD.unpack_flt nu / u)) in
    let z = Mat.(z $* xi) in
    let z0 = Mat.(gaussian i m * AD.unpack_arr sigma0) in
    AD.pack_arr (Mat.concatenate ~axis:0 [| z0; z |])


  let neg_logp_t ~prms =
    let nu, sigma = get_eff_prms ~prms in
    let m = AD.Mat.numel sigma in
    let m_half = AD.F Float.(of_int m / 2.) in
    let nu_half = AD.Maths.(F 0.5 * nu) in
    let nu_plus_m_half = AD.Maths.(F 0.5 * (nu + F Float.(of_int m))) in
    let sigma0 = Owl_parameters.extract prms.first_step in
    let cst0 = Float.(of_int m * log Const.pi2) in
    let cst =
      let cst1 = AD.Maths.(AD.loggamma nu_half - AD.loggamma nu_plus_m_half) in
      let cst2 = AD.Maths.(m_half * log (F Const.pi * nu)) in
      let cst3 = AD.Maths.(sum' (log sigma)) in
      AD.Maths.(cst1 + cst2 + cst3)
    in
    fun ~k ~x:_ ~u ->
      let stu =
        let utilde = AD.Maths.(u / sigma) in
        AD.Maths.(cst + (nu_plus_m_half * log (F 1. + (l2norm_sqr' utilde / nu))))
      in
      match X.n_beg with
      | None ->
        if k = 0
        then
          AD.Maths.(
            F 0.5 * (F cst0 + (F 2. * sum' (log sigma0)) + l2norm_sqr' (u / sigma0)))
        else stu
      | Some nb ->
        if k < nb
        then
          AD.Maths.(
            F 0.5 * (F cst0 + (F 2. * sum' (log sigma0)) + l2norm_sqr' (u / sigma0)))
        else stu


  let neg_jac_t =
    let jac_t ~prms =
      let nu, sigma = get_eff_prms ~prms in
      let m = AD.Mat.numel sigma in
      let nu_plus_m_half = AD.Maths.(F 0.5 * (nu + F Float.(of_int m))) in
      let sigma2 = AD.Maths.sqr sigma in
      fun ~k ~x:_ ~u ->
        let stu =
          let tmp =
            let utilde = AD.Maths.(u / sigma) in
            AD.Maths.(F 1. + (l2norm_sqr' utilde / nu))
          in
          let tmp' = AD.Maths.(F 2. * u / sigma2 / nu) in
          AD.Maths.(nu_plus_m_half * tmp' / tmp)
        in
        match X.n_beg with
        | None ->
          if k = 0
          then (
            let var0 = Owl_parameters.extract prms.first_step |> AD.Maths.sqr in
            AD.Maths.(u / var0))
          else stu
        | Some nb ->
          if k < nb
          then (
            let var0 = Owl_parameters.extract prms.first_step |> AD.Maths.sqr in
            AD.Maths.(u / var0))
          else stu
    in
    Some jac_t


  let neg_hess_t =
    let hess_t ~prms =
      let nu, sigma = get_eff_prms ~prms in
      let m = AD.Mat.numel sigma in
      let nu_plus_m_half = AD.Maths.(F 0.5 * (nu + F Float.(of_int m))) in
      let sigma2 = AD.Maths.sqr sigma in
      fun ~k ~x:_ ~u ->
        let stu =
          let u_over_s = AD.Maths.(u / sigma) in
          let tau = AD.Maths.(F 1. + (l2norm_sqr' u_over_s / nu)) in
          let cst = AD.Maths.(F 2. * nu_plus_m_half / nu / sqr tau) in
          let term1 = AD.Maths.(diagm (tau / sigma2)) in
          let term2 = AD.Maths.(F 2. * (transpose u_over_s *@ u_over_s) / nu) in
          AD.Maths.(cst * (term1 - term2))
        in
        match X.n_beg with
        | None ->
          if k = 0
          then (
            let var0 = Owl_parameters.extract prms.first_step |> AD.Maths.sqr in
            AD.Maths.(diagm (F 1. / var0)))
          else stu
        | Some nb ->
          if k < nb
          then (
            let var0 = Owl_parameters.extract prms.first_step |> AD.Maths.sqr in
            AD.Maths.(diagm (F 1. / var0)))
          else stu
    in
    Some hess_t


  let logp ~prms ~n_steps =
    let nu, sigma = get_eff_prms ~prms in
    let sigma0 = Owl_parameters.extract prms.first_step in
    let m = AD.Mat.numel sigma in
    let m_half = AD.F Float.(of_int m / 2.) in
    let nu_half = AD.Maths.(F 0.5 * nu) in
    let nu_plus_m_half = AD.Maths.(m_half + nu_half) in
    let i =
      match X.n_beg with
      | Some i -> i
      | None -> 1
    in
    let cst0 =
      AD.Maths.(
        F (Float.of_int i)
        * (F Float.(of_int m * log Const.pi2) + (F 2. * sum' (log sigma0))))
    in
    let cst =
      let cst1 = AD.Maths.(AD.loggamma nu_half - AD.loggamma nu_plus_m_half) in
      let cst2 = AD.Maths.(m_half * log (F Const.pi * nu)) in
      let cst3 = AD.Maths.(sum' (log sigma)) in
      AD.Maths.(F Float.(of_int n_steps) * (cst1 + cst2 + cst3))
    in
    fun u ->
      (* u is K x T x M *)
      let u_s = AD.shape u in
      let n_samples = u_s.(0) in
      let u0 =
        AD.Maths.get_slice [ []; [ 0; i - 1 ]; [] ] u
        |> fun v -> AD.Maths.reshape v [| -1; m |]
      in
      let u = AD.Maths.get_slice [ []; [ i; -1 ]; [] ] u in
      assert (Array.length u_s = 3);
      let cst0 = AD.Maths.(F Float.(of_int n_samples) * cst0) in
      let cst = AD.Maths.(F Float.(of_int n_samples) * cst) in
      let u = AD.Maths.reshape u [| -1; m |] in
      let first_term = AD.Maths.(F 0.5 * (cst0 + l2norm_sqr' (u0 / sigma0))) in
      let rest =
        AD.Maths.(
          cst + (nu_plus_m_half * sum' (log (F 1. + (sum ~axis:1 (sqr (u / sigma)) / nu)))))
      in
      AD.Maths.(neg (first_term + rest))
end
