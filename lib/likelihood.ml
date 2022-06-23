open Base
open Owl
include Likelihood_typ

module Arr_output (Label : sig
  val label : String.t
end) =
struct
  type output_t = AD.t
  type output = AD.t

  let save_output ?zip ?prefix o =
    Misc.save_mat
      ?zip
      ~out:(Owl_parameters.with_prefix ?prefix Label.label)
      (AD.unpack_arr o)


  let output_slice ~k o = AD.Maths.get_slice [ [ k ] ] o
  let numel = AD.Mat.numel

  let stats o =
    let k = Array.length o in
    let c f k =
      let tmp =
        Array.fold o ~init:None ~f:(fun accu x ->
            match accu with
            | None -> Some (f x)
            | Some a -> Some AD.Maths.(a + f x))
      in
      AD.Maths.(Option.value_exn tmp / F Float.(of_int k))
    in
    let o_avg = c Fn.id k in
    let o_var = c (fun x -> AD.Maths.(sqr (x - o_avg))) (max 1 (k - 1)) in
    o_avg, o_var
end

module Gaussian (X : sig
  include Dims_T

  val label : string
  val normalize_c : bool
end) =
struct
  module P = Owl_parameters.Make (Gaussian_P)
  open Gaussian_P

  include Arr_output (struct
    let label = X.label
  end)

  let requires_linesearch = false
  let label = X.label
  let n = X.n

  let init ?(sigma2 = 1.) ?(bias = 0.) (set : Owl_parameters.setter) =
    { c = set (AD.Mat.gaussian ~sigma:Float.(1. / sqrt (of_int X.n)) X.n_output X.n)
    ; c_mask = None
    ; bias = set (AD.Mat.create 1 X.n_output bias)
    ; variances = set ~above:0.001 (AD.pack_arr (Mat.create 1 X.n_output sigma2))
    }


  let unpack_c ~prms =
    let c = Owl_parameters.extract prms.c in
    let c =
      match prms.c_mask with
      | None -> c
      | Some cm -> AD.Maths.(c * cm)
    in
    if X.normalize_c then AD.Maths.(c / sqrt (sum ~axis:1 (sqr c))) else c


  let pre_sample ~prms ~z =
    let bias = Owl_parameters.extract prms.bias in
    let c = unpack_c ~prms in
    (* z is T x M *)
    AD.Maths.(bias + (z *@ transpose c))


  let sample ~prms ~z =
    let mu = pre_sample ~prms ~z in
    let res =
      let xi = AD.Arr.(gaussian (shape mu)) in
      AD.Maths.(xi * sqrt (Owl_parameters.extract prms.variances))
    in
    AD.Maths.(mu + res)


  let neg_logp_t ~prms ~o_t =
    let bias = Owl_parameters.extract prms.bias in
    let variances = Owl_parameters.extract prms.variances in
    let c = unpack_c ~prms in
    let c_t = AD.Maths.transpose c in
    let n_out = Float.of_int X.n_output in
    let cst = AD.F Float.(n_out * log Const.pi2) in
    let sum_log_var = AD.Maths.(sum' (log variances)) in
    fun ~k:_ ~z_t ->
      let mu_t = AD.Maths.(bias + (z_t *@ c_t)) in
      assert (Poly.(AD.shape mu_t = AD.shape o_t));
      AD.Maths.(F 0.5 * (cst + sum_log_var + sum' (sqr (o_t - mu_t) / variances)))


  let neg_jac_t =
    let neg_jac_t ~prms ~o_t =
      let bias = Owl_parameters.extract prms.bias in
      let variances = Owl_parameters.extract prms.variances in
      let c = unpack_c ~prms in
      let c_t = AD.Maths.transpose c in
      let c_inv_variances = AD.Maths.(F 1. / transpose variances * c) in
      fun ~k:_ ~z_t ->
        (*z = 1*N, c = 0*N, data = 1*0*)
        let mu_t = AD.Maths.(bias + (z_t *@ c_t)) in
        AD.Maths.((mu_t - o_t) *@ c_inv_variances)
    in
    Some neg_jac_t


  let neg_hess_t =
    let neg_hess_t ~prms ~o_t:_ =
      let variances = Owl_parameters.extract prms.variances in
      let c = unpack_c ~prms in
      let c_inv_variances = AD.Maths.(F 1. / transpose variances * c) in
      let tmp = AD.Maths.(transpose c *@ c_inv_variances) in
      (*z = 1*N, c = 0*N, data = 1*0*)
      fun ~k:_ ~z_t:_ -> tmp
    in
    Some neg_hess_t


  let logp ~prms ~o =
    let variances = Owl_parameters.extract prms.variances in
    let bias = Owl_parameters.extract prms.bias in
    let c = unpack_c ~prms in
    let c_t = AD.Maths.transpose c in
    let sum_log_var = AD.Maths.(sum' (log variances)) in
    let o = AD.expand_to_3d o in
    fun ~z ->
      let z_s = AD.shape z in
      assert (Array.length z_s = 3);
      let n_samples = z_s.(0) in
      let n_steps = z_s.(1) in
      let z = AD.Maths.reshape z [| -1; X.n |] in
      let mu = AD.Maths.(bias + (z *@ c_t)) in
      let diff =
        let mu = AD.Maths.reshape mu [| n_samples; n_steps; X.n_output |] in
        AD.Maths.(reshape (o - mu) [| -1; X.n_output |])
      in
      let cst =
        AD.F Float.(of_int Int.(n_samples * n_steps * X.n_output) * log Const.pi2)
      in
      AD.Maths.(
        F (-0.5)
        * (cst
          + (F Float.(of_int Int.(n_steps * n_samples)) * sum_log_var)
          + sum' (sqr diff / variances)))
end

module Poisson (X : sig
  include Dims_T

  val label : string
  val dt : AD.t
  val link_function : AD.t -> AD.t
  val d_link_function : AD.t -> AD.t
  val d2_link_function : AD.t -> AD.t
end) =
struct
  module P = Owl_parameters.Make (Poisson_P)
  open Poisson_P
  open X

  include Arr_output (struct
    let label = X.label
  end)

  let requires_linesearch = true
  let label = X.label
  let n = X.n

  let init (set : Owl_parameters.setter) =
    { c = set (AD.Mat.gaussian ~sigma:Float.(1. / sqrt (of_int n)) n_output n)
    ; c_mask = None
    ; bias = set (AD.Mat.zeros 1 n_output)
    ; gain = set (AD.Mat.ones 1 n_output)
    }


  let unpack_c ~prms =
    let c = Owl_parameters.extract prms.c in
    match prms.c_mask with
    | None -> c
    | Some cm -> AD.Maths.(c * cm)


  let pre_sample_before_link_function ~prms ~z =
    let c = unpack_c ~prms in
    let bias = Owl_parameters.extract prms.bias in
    (* z is T x M *)
    AD.Maths.((z *@ transpose c) + bias)


  let pre_sample ~prms ~z = link_function (pre_sample_before_link_function ~prms ~z)

  let sample ~prms ~z =
    let t = AD.Mat.row_num z in
    (* z is T x M *)
    let mu = pre_sample ~prms ~z in
    let spikes =
      Owl_distribution_generic.poisson_rvs ~mu:(AD.unpack_arr AD.Maths.(mu * dt)) ~n:1
    in
    AD.Maths.reshape (AD.pack_arr spikes) [| t; -1 |]


  let logfact k =
    let rec iter k accu =
      if k <= 1 then accu else iter (k - 1) Float.(accu + log (of_int k))
    in
    iter k 0.


  (* redefine the link_function to include a safe floor *)
  let link_function x = AD.Maths.(AD.F 1E-3 + link_function x)

  let neg_logp_t ~prms =
    let bias = Owl_parameters.extract prms.bias in
    let gain = Owl_parameters.extract prms.gain in
    let c = unpack_c ~prms in
    let c_t = AD.Maths.transpose c in
    fun ~o_t ->
      let logfact =
        AD.pack_arr (Mat.map (fun x -> logfact Int.(of_float x)) (AD.unpack_arr o_t))
      in
      fun ~k:_ ~z_t ->
        let rate_t = AD.Maths.(dt * gain * link_function ((z_t *@ c_t) + bias)) in
        let log_rate_t = AD.Maths.(log rate_t) in
        assert (Poly.(AD.shape log_rate_t = AD.shape o_t));
        AD.Maths.(sum' (rate_t + logfact - (o_t * log_rate_t)))


  let d_log_link_function x = AD.Maths.(d_link_function x / link_function x)

  let d2_log_link_function x =
    let lx = link_function x in
    let dlx = d_link_function x in
    let ddlx = d2_link_function x in
    AD.Maths.(((ddlx * lx) - sqr dlx) / sqr lx)


  let neg_jac_t =
    (* dlogp/dz *)
    let neg_jac_t ~prms =
      let c = unpack_c ~prms in
      let c_t = AD.Maths.transpose c in
      let bias = Owl_parameters.extract prms.bias in
      let gain = Owl_parameters.extract prms.gain in
      fun ~o_t ~k:_ ~z_t ->
        let a = AD.Maths.(bias + (z_t *@ c_t)) in
        let tmp1 = AD.Maths.(dt * gain * d_link_function a) in
        let tmp2 = AD.Maths.(o_t * d_log_link_function a) in
        AD.Maths.((tmp1 - tmp2) *@ c)
    in
    Some neg_jac_t


  let neg_hess_t =
    let neg_hess_t ~prms =
      let c = unpack_c ~prms in
      let c_t = AD.Maths.transpose c in
      let bias = Owl_parameters.extract prms.bias in
      let gain = Owl_parameters.extract prms.gain in
      fun ~o_t ~k:_ ~z_t ->
        let a = AD.Maths.(bias + (z_t *@ c_t)) in
        let tmp1 = AD.Maths.(dt * gain * d2_link_function a) in
        let tmp2 = AD.Maths.(o_t * d2_log_link_function a) in
        AD.Maths.(transpose c * (tmp1 - tmp2) *@ c)
    in
    Some neg_hess_t


  let logp ~prms =
    let c = unpack_c ~prms in
    let c_t = AD.Maths.transpose c in
    let bias = Owl_parameters.extract prms.bias in
    let gain = Owl_parameters.extract prms.gain in
    fun ~o ->
      let logfact =
        AD.pack_arr (Mat.map (fun x -> logfact Int.(of_float x)) (AD.unpack_arr o))
      in
      let o = AD.expand_to_3d o in
      fun ~z ->
        let z_s = AD.shape z in
        assert (Array.length z_s = 3);
        let n_samples = z_s.(0) in
        let z = AD.Maths.reshape z [| -1; X.n |] in
        let rates = AD.Maths.(dt * gain * link_function ((z *@ c_t) + bias)) in
        let rates = AD.Maths.(reshape rates [| n_samples; -1; X.n_output |]) in
        let log_rates = AD.Maths.(log rates) in
        AD.Maths.(
          sum' ((o * log_rates) - rates) - (F Float.(of_int n_samples) * sum' logfact))
end

module Pair (L1 : T) (L2 : T) = struct
  module P =
    Owl_parameters.Make
      (Pair_P.Make
         (struct
           include L1.P

           let label = L1.label
         end)
         (struct
           include L2.P

           let label = L2.label
         end))

  open Pair_P

  type output_t = (L1.output_t, L2.output_t) prm_
  type output = (L1.output, L2.output) prm_

  let requires_linesearch = L1.requires_linesearch || L2.requires_linesearch
  let label = Printf.sprintf "pair(%s-%s)" L1.label L2.label

  let n =
    assert (L1.n = L2.n);
    L1.n


  let save_output ?zip ?prefix o =
    L1.save_output ?zip ?prefix o.fst;
    L2.save_output ?zip ?prefix o.snd


  let output_slice ~k o =
    { fst = L1.output_slice ~k o.fst; snd = L2.output_slice ~k o.snd }


  let numel x = L1.numel x.fst + L2.numel x.snd

  let stats o =
    let mean_1, var_1 = L1.stats (Array.map o ~f:(fun x -> x.fst)) in
    let mean_2, var_2 = L2.stats (Array.map o ~f:(fun x -> x.snd)) in
    { fst = mean_1; snd = mean_2 }, { fst = var_1; snd = var_2 }


  let pre_sample ~prms ~z =
    { fst = L1.pre_sample ~prms:prms.fst ~z; snd = L2.pre_sample ~prms:prms.snd ~z }


  let sample ~prms ~z =
    { fst = L1.sample ~prms:prms.fst ~z; snd = L2.sample ~prms:prms.snd ~z }


  let add f1 f2 ~prms =
    let f1 = f1 ~prms:prms.fst in
    let f2 = f2 ~prms:prms.snd in
    fun ~o_t ->
      let f1 = f1 ~o_t:o_t.fst in
      let f2 = f2 ~o_t:o_t.snd in
      fun ~k ~z_t -> AD.Maths.(f1 ~k ~z_t + f2 ~k ~z_t)


  let neg_logp_t = add L1.neg_logp_t L2.neg_logp_t

  let neg_jac_t =
    match L1.neg_jac_t, L2.neg_jac_t with
    | Some f1, Some f2 -> Some (add f1 f2)
    | _ -> None


  let neg_hess_t =
    match L1.neg_hess_t, L2.neg_hess_t with
    | Some f1, Some f2 -> Some (add f1 f2)
    | _ -> None


  let logp ~prms =
    let f1 = L1.logp ~prms:prms.fst in
    let f2 = L2.logp ~prms:prms.snd in
    fun ~o ->
      let f1 = f1 ~o:o.fst in
      let f2 = f2 ~o:o.snd in
      fun ~z -> AD.Maths.(f1 ~z + f2 ~z)
end
