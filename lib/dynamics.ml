open Base
open Owl
open Owl_parameters
include Dynamics_typ

module Integrate (D : T) = struct
  let integrate ~prms ~ext_u =
    let dyn_k = D.dyn ~theta:prms ~ext_u in
    let n = D.n in
    fun ~u ->
      (* assume u is n_samples x n_steps x m *)
      assert (Array.length (AD.shape u) = 3);
      assert (AD.(shape u).(2) = D.m);
      let u = AD.Maths.transpose ~axis:[| 1; 0; 2 |] u in
      (* now u is T x K x M *)
      let n_steps = AD.(shape u).(0) in
      let n_samples = AD.(shape u).(1) in
      let x0 = AD.Mat.zeros n_samples n in
      let us =
        let u = AD.Maths.reshape u [| n_steps; -1 |] in
        AD.Maths.split ~axis:0 (Array.init n_steps ~f:(fun _ -> 1)) u
        |> Array.map ~f:(fun v -> AD.Maths.reshape v [| n_samples; -1 |])
        |> Array.to_list
      in
      let rec dyn k x xs us =
        match us with
        | [] -> xs
        | u :: unexts ->
          let new_x = dyn_k ~k ~x ~u in
          dyn (k + 1) new_x (new_x :: xs) unexts
      in
      dyn 0 x0 [] us
      |> List.rev_map ~f:(fun v -> AD.Maths.reshape v [| 1; n_samples; n |])
      |> Array.of_list
      |> AD.Maths.concatenate ~axis:0 (* T x K x N *)
      (* result KxTxN *)
      |> AD.Maths.transpose ~axis:[| 1; 0; 2 |]
end

(* this is a wrapper that normalises the B input matrix, 
   and automates the handling of the first few  "special" time bins
   during which the dynamics are different (to enable setting the
   initial condition in an unrestricted way). *)
let wrapper ~n ~m =
  assert (n % m = 0);
  let n_beg = n / m in
  let beg_bs =
    Array.init n_beg ~f:(fun k ->
        Mat.init_2d m n (fun i j -> if j / n_beg = k && j % m = i then 1. else 0.)
        |> AD.pack_arr)
  in
  let id_n = AD.Mat.eye n in
  let scale_factor = AD.F Float.(sqrt (of_int n / of_int m)) in
  ( n_beg
  , function
    | None ->
      let b = AD.Mat.eye n in
      b, Fn.id, Fn.id, Fn.id, Fn.id
    | Some b ->
      let b =
        let b = extract b in
        AD.Maths.(scale_factor * b / sqrt (sum ~axis:1 (sqr b)))
      in
      let u_eff u = AD.Maths.(u *@ b) in
      let dyn_wrap f ~k ~x ~u =
        if k < n_beg then AD.Maths.(x + (u *@ beg_bs.(k))) else f ~k ~x ~u
      in
      let dyn_x_wrap f ~k ~x ~u = if k < n_beg then id_n else f ~k ~x ~u in
      let dyn_u_wrap f ~k ~x ~u = if k < n_beg then beg_bs.(k) else f ~k ~x ~u in
      b, u_eff, dyn_wrap, dyn_x_wrap, dyn_u_wrap )


module Nonlinear (X : sig
  include Dims_T

  val phi : [ `linear | `nonlinear of (AD.t -> AD.t) * (AD.t -> AD.t) ]
end) =
struct
  module P = Owl_parameters.Make (Nonlinear_Init_P)
  open Nonlinear_Init_P
  open X

  let n = X.n
  and m = X.m

  let n_beg, wrapper = wrapper ~n ~m

  let phi, d_phi, requires_linesearch =
    match phi with
    | `linear -> (fun x -> x), (fun x -> AD.Arr.(ones (shape x))), false
    | `nonlinear (f, df) -> f, df, true


  let init ?(radius = 0.1) (set : Owl_parameters.setter) =
    let sigma = Float.(radius / sqrt (of_int n)) in
    { a = set (AD.Mat.gaussian ~sigma n n)
    ; bias = set (AD.Mat.zeros 1 n)
    ; b =
        (if n = m
        then None
        else Some (set (AD.Mat.gaussian ~sigma:Float.(1. / sqrt (of_int m)) m n)))
    }


  let dyn ~theta ~ext_u =
    match ext_u with |Some ext_u ->
    let _, u_eff, dyn_wrap, _, _ = wrapper theta.b in
    let a = extract theta.a in
    let bias = extract theta.bias in
    dyn_wrap (fun ~k:_ ~x ~u -> AD.Maths.((phi x *@ a) + u_eff u + bias + ))


  let dyn_x =
    Some
      (fun ~theta ->
        let _, _, _, dyn_x_wrap, _ = wrapper theta.b in
        let a = extract theta.a in
        dyn_x_wrap (fun ~k:_ ~x ~u:_ -> AD.Maths.(transpose (d_phi x) * a)))


  let dyn_u =
    Some
      (fun ~theta ->
        let b, _, _, _, dyn_u_wrap = wrapper theta.b in
        dyn_u_wrap (fun ~k:_ ~x:_ ~u:_ -> b))
end
(* 
module Linear (Dims : Dims_T) = struct
  module P = Owl_parameters.Make (Linear_P)
  open Linear_P
  include Dims

  let requires_linesearch = false
  let n_beg, wrapper = wrapper ~n ~m

  (* alpha is the spectral abscissa of the equivalent continuous-time system
     beta is the spectral radius of the random S *)
  let init ~dt_over_tau ~alpha ~beta (set : Owl_parameters.setter) =
    let d =
      let tmp = Float.(exp (-2. * dt_over_tau * (1.0 - alpha))) in
      Mat.create 1 n Float.(tmp / (1. - tmp)) |> AD.pack_arr
    in
    let u = AD.Mat.eye n in
    let q =
      let s = Mat.(Float.(beta * dt_over_tau / sqrt (2. * of_int n)) $* gaussian n n) in
      Linalg.D.expm Mat.(s - transpose s) |> AD.pack_arr
    in
    let b = if n = m then None else Some (set (AD.Mat.gaussian m n)) in
    { d = set ~above:1E-5 d; u = set u; q = set q; b }


  let unpack_a ~prms =
    let q =
      let q, r = AD.Linalg.qr (extract prms.q) in
      let r = AD.Maths.diag r in
      AD.Maths.(q * signum r)
    in
    let u =
      let q, r = AD.Linalg.qr (extract prms.u) in
      let r = AD.Maths.diag r in
      AD.Maths.(q * signum r)
    in
    let d = extract prms.d in
    let dp1_sqrt_inv = AD.Maths.(F 1. / sqrt (F 1. + d)) in
    let d_sqrt = AD.Maths.(sqrt d) in
    AD.Maths.(u * d_sqrt *@ (q * dp1_sqrt_inv) *@ transpose u)


  let dyn ~theta =
    let a = unpack_a ~prms:theta in
    let _, u_eff, dyn_wrap, _, _ = wrapper theta.b in
    dyn_wrap (fun ~k:_ ~x ~u -> AD.Maths.((x *@ a) + u_eff u))


  let dyn_x =
    Some
      (fun ~theta ->
        let a = unpack_a ~prms:theta in
        let _, _, _, dyn_x_wrap, _ = wrapper theta.b in
        dyn_x_wrap (fun ~k:_ ~x:_ ~u:_ -> a))


  let dyn_u =
    Some
      (fun ~theta ->
        let b, _, _, _, dyn_u_wrap = wrapper theta.b in
        dyn_u_wrap (fun ~k:_ ~x:_ ~u:_ -> b))
end

module Nonlinear (X : sig
  include Dims_T

  val phi : [ `linear | `nonlinear of (AD.t -> AD.t) * (AD.t -> AD.t) ]
end) =
struct
  module P = Owl_parameters.Make (Nonlinear_Init_P)
  open Nonlinear_Init_P
  open X

  let n = X.n
  and m = X.m

  let n_beg, wrapper = wrapper ~n ~m

  let phi, d_phi, requires_linesearch =
    match phi with
    | `linear -> (fun x -> x), (fun x -> AD.Arr.(ones (shape x))), false
    | `nonlinear (f, df) -> f, df, true


  let init ?(radius = 0.1) (set : Owl_parameters.setter) =
    let sigma = Float.(radius / sqrt (of_int n)) in
    { a = set (AD.Mat.gaussian ~sigma n n)
    ; bias = set (AD.Mat.zeros 1 n)
    ; b =
        (if n = m
        then None
        else Some (set (AD.Mat.gaussian ~sigma:Float.(1. / sqrt (of_int m)) m n)))
    }


  let dyn ~theta =
    let _, u_eff, dyn_wrap, _, _ = wrapper theta.b in
    let a = extract theta.a in
    let bias = extract theta.bias in
    dyn_wrap (fun ~k:_ ~x ~u -> AD.Maths.((phi x *@ a) + u_eff u + bias))


  let dyn_x =
    Some
      (fun ~theta ->
        let _, _, _, dyn_x_wrap, _ = wrapper theta.b in
        let a = extract theta.a in
        dyn_x_wrap (fun ~k:_ ~x ~u:_ -> AD.Maths.(transpose (d_phi x) * a)))


  let dyn_u =
    Some
      (fun ~theta ->
        let b, _, _, _, dyn_u_wrap = wrapper theta.b in
        dyn_u_wrap (fun ~k:_ ~x:_ ~u:_ -> b))
end

module MGU (X : sig
  include Dims_T

  val phi : AD.t -> AD.t
  val d_phi : AD.t -> AD.t
  val sigma : AD.t -> AD.t
  val d_sigma : AD.t -> AD.t
end) =
struct
  module P = Owl_parameters.Make (MGU_P)
  open MGU_P
  open X

  let n = X.n
  let m = X.m
  let n_beg, wrapper = wrapper ~n ~m
  let requires_linesearch = true

  let init (set : Owl_parameters.setter) =
    (*h = size 1xN
     x = size 1xN (x = Bu)
     h = size 1xK 
     f = size of h so 1xN
     Wf = NxN
     B = MxN *)
    { wf = set (AD.Mat.zeros m n)
    ; wh = set (AD.Mat.gaussian m n)
    ; bh = set (AD.Mat.zeros 1 n)
    ; bf = set (AD.Mat.zeros 1 n)
    ; uh = set (AD.Mat.zeros n n)
    ; uf = set (AD.Mat.zeros n n)
    }


  let default_regularizer ?(lambda = 1.) prms =
    let uh = extract prms.uh in
    let uf = extract prms.uf in
    let z = Float.(lambda / of_int Int.(n * n)) in
    AD.Maths.(F z * (l2norm_sqr' uh + l2norm_sqr' uf))


  let dyn ~theta =
    let wh, _, dyn_wrap, _, _ = wrapper (Some theta.wh) in
    let wf = Owl_parameters.extract theta.wf in
    let bh = Owl_parameters.extract theta.bh in
    let bf = Owl_parameters.extract theta.bf in
    let uh = Owl_parameters.extract theta.uh in
    let uf = Owl_parameters.extract theta.uf in
    dyn_wrap (fun ~k:_ ~x ~u ->
        let h_pred = x in
        let f = sigma AD.Maths.((u *@ wf) + bf + (h_pred *@ uf)) in
        let h_hat =
          let hf = AD.Maths.(h_pred * f) in
          phi AD.Maths.((u *@ wh) + bh + (hf *@ uh))
        in
        AD.Maths.(((F 1. - f) * h_pred) + (f * h_hat)))


  let dyn_x =
    Some
      (fun ~theta ->
        let wh, _, _, dyn_x_wrap, _ = wrapper (Some theta.wh) in
        let wf = Owl_parameters.extract theta.wf in
        let bh = Owl_parameters.extract theta.bh in
        let bf = Owl_parameters.extract theta.bf in
        let uh = Owl_parameters.extract theta.uh in
        let uf = Owl_parameters.extract theta.uf in
        dyn_x_wrap (fun ~k:_ ~x ~u ->
            let h_pred = x in
            let f_pre = AD.Maths.((u *@ wf) + bf + (h_pred *@ uf)) in
            let f = sigma f_pre in
            let h_hat_pre =
              let hf = AD.Maths.(h_pred * f) in
              AD.Maths.((u *@ wh) + bh + (hf *@ uh))
            in
            let h_hat = phi h_hat_pre in
            AD.Maths.(
              diagm (F 1. - f)
              - (uf * ((h_pred - h_hat) * d_sigma f_pre))
              + (((transpose f * uh) + (uf *@ (transpose (h_pred * d_sigma f_pre) * uh)))
                * (f * d_phi h_hat_pre)))))


  let dyn_u =
    Some
      (fun ~theta ->
        let wh, _, _, _, dyn_u_wrap = wrapper (Some theta.wh) in
        let wf = Owl_parameters.extract theta.wf in
        let bh = Owl_parameters.extract theta.bh in
        let bf = Owl_parameters.extract theta.bf in
        let uh = Owl_parameters.extract theta.uh in
        let uf = Owl_parameters.extract theta.uf in
        dyn_u_wrap (fun ~k:_ ~x ~u ->
            let h_pred = x in
            let f_pre = AD.Maths.((u *@ wf) + bf + (h_pred *@ uf)) in
            let f = sigma f_pre in
            let h_hat_pre =
              let hf = AD.Maths.(h_pred * f) in
              AD.Maths.((u *@ wh) + bh + (hf *@ uh))
            in
            let h_hat = phi h_hat_pre in
            AD.Maths.(
              (wf * (d_sigma f_pre * (h_hat - h_pred)))
              + ((wh + (wf *@ (transpose (h_pred * d_sigma f_pre) * uh)))
                * (f * d_phi h_hat_pre)))))
end

module MGU2 (X : sig
  include Dims_T

  val phi : AD.t -> AD.t
  val d_phi : AD.t -> AD.t
  val sigma : AD.t -> AD.t
  val d_sigma : AD.t -> AD.t
end) =
struct
  module P = Owl_parameters.Make (MGU2_P)
  open MGU2_P
  open X

  let requires_linesearch = true
  let n = X.n
  let m = X.m
  let n_beg, wrapper = wrapper ~n ~m

  let init (set : Owl_parameters.setter) =
    (* h : size 1xN
       x : size 1xN (x = Bu)
       h : size 1xK 
       f : size of h so 1xN *)
    { wh = set (AD.Mat.gaussian m n)
    ; bh = set (AD.Mat.zeros 1 n)
    ; bf = set (AD.Mat.zeros 1 n)
    ; uh = set (AD.Mat.zeros n n)
    ; uf = set (AD.Mat.zeros n n)
    }


  let default_regularizer ?(lambda = 1.) prms =
    let uh = extract prms.uh in
    let uf = extract prms.uf in
    let z = Float.(lambda / of_int Int.(n * n)) in
    AD.Maths.(F z * (l2norm_sqr' uh + l2norm_sqr' uf))


  let dyn ~theta =
    let wh, _, dyn_wrap, _, _ = wrapper (Some theta.wh) in
    let bh = Owl_parameters.extract theta.bh in
    let bf = Owl_parameters.extract theta.bf in
    let uh = Owl_parameters.extract theta.uh in
    let uf = Owl_parameters.extract theta.uf in
    dyn_wrap (fun ~k:_ ~x ~u ->
        let h_pred = x in
        let f = sigma AD.Maths.(bf + (h_pred *@ uf)) in
        let h_hat =
          let hf = AD.Maths.(h_pred * f) in
          AD.Maths.(phi AD.Maths.(bh + (hf *@ uh)) + (u *@ wh))
        in
        AD.Maths.(((F 1. - f) * h_pred) + (f * h_hat)))


  let dyn_x =
    Some
      (fun ~theta ->
        let wh, _, _, dyn_x_wrap, _ = wrapper (Some theta.wh) in
        let bf = Owl_parameters.extract theta.bf in
        let bh = Owl_parameters.extract theta.bh in
        let uh = Owl_parameters.extract theta.uh in
        let uf = Owl_parameters.extract theta.uf in
        dyn_x_wrap (fun ~k:_ ~x ~u ->
            let h_pred = x in
            let f_pre = AD.Maths.(bf + (h_pred *@ uf)) in
            let f = sigma f_pre in
            let h_hat_pre =
              let hf = AD.Maths.(h_pred * f) in
              AD.Maths.(bh + (hf *@ uh))
            in
            let h_hat = AD.Maths.(phi h_hat_pre + (u *@ wh)) in
            AD.Maths.(
              diagm (F 1. - f)
              - (uf * ((h_pred - h_hat) * d_sigma f_pre))
              + (((transpose f * uh) + (uf *@ (transpose (h_pred * d_sigma f_pre) * uh)))
                * (f * d_phi h_hat_pre)))))


  let dyn_u =
    Some
      (fun ~theta ->
        let wh, _, _, _, dyn_u_wrap = wrapper (Some theta.wh) in
        let bf = Owl_parameters.extract theta.bf in
        let uf = Owl_parameters.extract theta.uf in
        dyn_u_wrap (fun ~k:_ ~x ~u:_ ->
            let h_pred = x in
            let f_pre = AD.Maths.(bf + (h_pred *@ uf)) in
            let f = sigma f_pre in
            AD.Maths.(wh * f)))
end *)
