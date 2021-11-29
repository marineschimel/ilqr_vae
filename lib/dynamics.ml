open Base
open Owl
include Dynamics_typ

module Integrate (D : Dynamics_T) = struct
  let integrate ~prms =
    let dyn_k = D.dyn ~theta:prms in
    fun ~n ~u ->
      (* assume u is n_samples x n_steps x m *)
      assert (Array.length (AD.shape u) = 3);
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
        | [] -> List.rev xs
        | u :: unexts ->
          let new_x = dyn_k ~k ~x ~u in
          dyn (k + 1) new_x (new_x :: xs) unexts
      in
      dyn 0 x0 [] us
      |> Array.of_list
      |> Array.map ~f:(fun v -> AD.Maths.reshape v [| 1; n_samples; n |])
      |> AD.Maths.concatenate ~axis:0 (* T x K x N *)
      |> AD.Maths.transpose ~axis:[| 1; 0; 2 |]

  (* result KxTxN *)
end

let b_rescaled b =
  Option.map b ~f:(function b ->
      let b = Owl_parameters.extract b in
      AD.Maths.(b / sqrt (sum ~axis:0 (sqr b))))


module Linear (X : sig
  val n_beg : int Option.t
end) =
struct
  module P = Owl_parameters.Make (Linear_P)
  open Linear_P

  let requires_linesearch = false

  (* alpha is the spectral abscissa of the equivalent continuous-time system
     beta is the spectral radius of the random S *)
  let init ~dt_over_tau ~alpha ~beta (set : Owl_parameters.setter) n m =
    (* exp (dt_over_tau * (W-I))
       where W = alpha*I + S *)
    let d =
      let tmp = Float.(exp (-2. * dt_over_tau * (1.0 - alpha))) in
      Mat.init_2d 1 n (fun _ _ -> Float.(tmp / (1. - tmp)))
    in
    let u = AD.Mat.eye n in
    let q =
      let s = Mat.(Float.(beta * dt_over_tau / sqrt (2. * of_int n)) $* gaussian n n) in
      Linalg.D.expm Mat.(s - transpose s)
    in
    let b = if n = m then None else Some (set (AD.Mat.gaussian m n)) in
    { d = set ~above:1E-5 (AD.pack_arr d); u = set u; q = set (AD.pack_arr q); b }


  let unpack_a ~prms =
    let q =
      let q, r = AD.Linalg.qr (Owl_parameters.extract prms.q) in
      let r = AD.Maths.diag r in
      AD.Maths.(q * signum r)
    in
    let u =
      let q, r = AD.Linalg.qr (Owl_parameters.extract prms.u) in
      let r = AD.Maths.diag r in
      AD.Maths.(q * signum r)
    in
    let d = Owl_parameters.extract prms.d in
    let dp1_sqrt_inv = AD.Maths.(F 1. / sqrt (F 1. + d)) in
    let d_sqrt = AD.Maths.(sqrt d) in
    AD.Maths.(u * d_sqrt *@ (q * dp1_sqrt_inv) *@ transpose u)


  let generate_bs ~n ~m =
    Option.map X.n_beg ~f:(fun nb ->
        let nr = n / nb in
        assert (nr = m);
        ( nb
        , Array.init nb ~f:(fun i ->
              let inr = i * nr in
              let rnr = n - ((i + 1) * nr) in
              AD.Maths.(
                transpose
                  (concatenate
                     ~axis:0
                     [| AD.Mat.zeros inr m; AD.Mat.eye nr; AD.Mat.zeros rnr m |]))) ))


  let extract_b ~theta ~n =
    match b_rescaled theta.b with
    | None -> AD.Mat.(eye n)
    | Some b -> b


  let dyn ~theta =
    let a = unpack_a ~prms:theta in
    let n = AD.Mat.row_num a in
    let b = extract_b ~theta ~n in
    let m = AD.Mat.row_num b in
    let beg_bs = generate_bs ~n ~m in
    let default x u = AD.Maths.((x *@ a) + (u *@ b)) in
    fun ~k ~x ~u ->
      match beg_bs with
      | None -> default x u
      | Some (i, beg_b) -> if k < i then AD.Maths.(x + (u *@ beg_b.(k))) else default x u


  let dyn_x =
    (* Marine to check this *)
    let dyn_x ~theta =
      let a = unpack_a ~prms:theta in
      let n = AD.Mat.row_num a in
      let id_n = AD.Mat.eye n in
      fun ~k ~x:_ ~u:_ ->
        match X.n_beg with
        | None -> a
        | Some i -> if k < i then id_n else a
    in
    Some dyn_x


  let dyn_u =
    (* Marine to check this *)
    let dyn_u ~theta =
      let q = Owl_parameters.extract theta.q in
      let n = AD.Mat.row_num q in
      let b = extract_b ~theta ~n in
      let m = AD.Mat.row_num b in
      let beg_bs = generate_bs ~n ~m in
      fun ~k ~x:_ ~u:_ ->
        match beg_bs with
        | None -> b
        | Some (i, beg_b) -> if k < i then beg_b.(k) else b
    in
    Some dyn_u
end

module Nonlinear (X : sig
  val phi : [ `linear | `nonlinear of (AD.t -> AD.t) * (AD.t -> AD.t) ]
  val n_beg : int Option.t
end) =
struct
  module P = Owl_parameters.Make (Nonlinear_Init_P)
  open Nonlinear_Init_P
  open X

  let phi, d_phi, requires_linesearch =
    match phi with
    | `linear -> (fun x -> x), (fun x -> AD.Arr.(ones (shape x))), false
    | `nonlinear (f, df) -> f, df, true


  let init ?(radius = 0.1) ~n ~m (set : Owl_parameters.setter) =
    let sigma = Float.(radius / sqrt (of_int n)) in
    { a = set (AD.Mat.gaussian ~sigma n n)
    ; bias = set (AD.Mat.zeros 1 n)
    ; b = Some (set (AD.Mat.gaussian ~sigma:Float.(1. / sqrt (of_int m)) m n))
    }


  let generate_bs ~n ~m =
    Option.map X.n_beg ~f:(fun nb ->
        let nr = n / nb in
        assert (nr = m);
        ( nb
        , Array.init nb ~f:(fun i ->
              let inr = i * nr in
              let rnr = n - ((i + 1) * nr) in
              AD.Maths.(
                transpose
                  (concatenate
                     ~axis:0
                     [| AD.Mat.zeros inr m; AD.Mat.eye nr; AD.Mat.zeros rnr m |]))) ))


  let u_eff ~prms =
    match b_rescaled prms.b with
    | None -> fun u -> u
    | Some b -> fun u -> AD.Maths.(u *@ b)


  let dyn ~theta =
    let a = Owl_parameters.extract theta.a in
    let bias = Owl_parameters.extract theta.bias in
    let n = AD.Mat.row_num a in
    let m =
      match theta.b with
      | None -> n
      | Some b -> AD.Mat.row_num (Owl_parameters.extract b)
    in
    let u_eff = u_eff ~prms:theta in
    let beg_bs = generate_bs ~n ~m in
    let default x u = AD.Maths.((phi x *@ a) + u_eff u + bias) in
    fun ~k ~x ~u ->
      match beg_bs with
      | None -> default x u
      | Some (nb, beg_b) ->
        if Int.(k < nb) then AD.Maths.(x + (u *@ beg_b.(k))) else default x u


  let dyn_x =
    let dyn_x ~theta =
      let a = Owl_parameters.extract theta.a in
      let n = AD.Mat.row_num a in
      let id_n = AD.Mat.eye n in
      let default x = AD.Maths.(transpose (d_phi x) * a) in
      fun ~k ~x ~u:_ ->
        match X.n_beg with
        | None -> default x
        | Some nb -> if Int.(k < nb) then id_n else default x
    in
    Some dyn_x


  let dyn_u =
    let dyn_u ~theta =
      let n = AD.Mat.row_num (Owl_parameters.extract theta.a) in
      let b =
        match b_rescaled theta.b with
        | None -> AD.Mat.eye n
        | Some b -> b
      in
      let m = AD.Mat.row_num b in
      let beg_bs = generate_bs ~n ~m in
      fun ~k ~x:_ ~u:_ ->
        match beg_bs with
        | None -> b
        | Some (nb, beg_b) -> if Int.(k < nb) then beg_b.(k) else b
    in
    Some dyn_u
end

module MGU (X : sig
  val phi : AD.t -> AD.t
  val d_phi : AD.t -> AD.t
  val sigma : AD.t -> AD.t
  val d_sigma : AD.t -> AD.t
  val n_beg : int Option.t
end) =
struct
  module P = Owl_parameters.Make (MGU_P)
  open MGU_P
  open X

  let requires_linesearch = true

  let init ~n ~m (set : Owl_parameters.setter) =
    (*h = size 1xN
     x = size 1xN (x = Bu)
     h = size 1xK 
     f = size of h so 1xN
     Wf = NxN
     B = MxN *)
    { wf = set (AD.Mat.zeros m n)
    ; wh = set (AD.Mat.gaussian m n)
    ; bh = set (AD.Mat.zeros 1 n)
    ; bf = set (AD.Mat.create 1 n 3.)
    ; uh = set (AD.Mat.zeros n n)
    ; uf = set (AD.Mat.zeros n n)
    }


  let with_wh_rescaled theta =
    { theta with
      wh =
        Owl_parameters.map
          (fun wh -> AD.Maths.(wh / sqrt (sum ~axis:0 (sqr wh))))
          theta.wh
    }


  let generate_bs ~n ~m =
    Option.map X.n_beg ~f:(fun nb ->
        let nr = n / nb in
        assert (nr = m);
        ( nb
        , Array.init nb ~f:(fun i ->
              let inr = i * nr in
              let rnr = n - ((i + 1) * nr) in
              AD.Maths.(
                transpose
                  (concatenate
                     ~axis:0
                     [| AD.Mat.zeros inr m; AD.Mat.eye nr; AD.Mat.zeros rnr m |]))) ))


  let dyn ~theta =
    let theta = with_wh_rescaled theta in
    let wh = Owl_parameters.extract theta.wh in
    let wf = Owl_parameters.extract theta.wf in
    let bh = Owl_parameters.extract theta.bh in
    let bf = Owl_parameters.extract theta.bf in
    let uh = Owl_parameters.extract theta.uh in
    let uf = Owl_parameters.extract theta.uf in
    let n = AD.Mat.col_num bh in
    let m = AD.Mat.row_num wh in
    let beg_bs = generate_bs ~n ~m in
    let default x u =
      let h_pred = x in
      let f = sigma AD.Maths.((u *@ wf) + bf + (h_pred *@ uf)) in
      let h_hat =
        let hf = AD.Maths.(h_pred * f) in
        phi AD.Maths.((u *@ wh) + bh + (hf *@ uh))
      in
      AD.Maths.(((F 1. - f) * h_pred) + (f * h_hat))
    in
    fun ~k ~x ~u ->
      match beg_bs with
      | None -> default x u
      | Some (nb, beg_b) ->
        if k < nb then AD.Maths.(x + (u *@ beg_b.(k))) else default x u


  let dyn_x =
    let _dyn_x ~theta =
      let theta = with_wh_rescaled theta in
      let wh = Owl_parameters.extract theta.wh in
      let wf = Owl_parameters.extract theta.wf in
      let bh = Owl_parameters.extract theta.bh in
      let bf = Owl_parameters.extract theta.bf in
      let uh = Owl_parameters.extract theta.uh in
      let uf = Owl_parameters.extract theta.uf in
      let n = AD.Mat.col_num bh in
      let id_n = AD.Mat.eye n in
      let default x u =
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
            * (f * d_phi h_hat_pre)))
      in
      fun ~k ~x ~u ->
        match X.n_beg with
        | None -> default x u
        | Some i -> if k < i then id_n else default x u
    in
    Some _dyn_x


  let dyn_u =
    let _dyn_u ~theta =
      let theta = with_wh_rescaled theta in
      let wh = Owl_parameters.extract theta.wh in
      let wf = Owl_parameters.extract theta.wf in
      let bh = Owl_parameters.extract theta.bh in
      let bf = Owl_parameters.extract theta.bf in
      let uh = Owl_parameters.extract theta.uh in
      let uf = Owl_parameters.extract theta.uf in
      let m = AD.Mat.row_num wh in
      let n = AD.Mat.col_num bh in
      let beg_bs = generate_bs ~n ~m in
      let default x u =
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
            * (f * d_phi h_hat_pre)))
      in
      fun ~k ~x ~u ->
        match beg_bs with
        | None -> default x u
        | Some (nb, beg_b) -> if k < nb then beg_b.(k) else default x u
    in
    Some _dyn_u
end

module MGU2 (X : sig
  val phi : AD.t -> AD.t
  val d_phi : AD.t -> AD.t
  val sigma : AD.t -> AD.t
  val d_sigma : AD.t -> AD.t
  val n_beg : int Option.t
end) =
struct
  module P = Owl_parameters.Make (MGU2_P)
  open MGU2_P
  open X

  let requires_linesearch = true

  let init ~n ~m (set : Owl_parameters.setter) =
    (* h : size 1xN
       x : size 1xN (x = Bu)
       h : size 1xK 
       f : size of h so 1xN *)
    { wh = set (AD.Mat.gaussian m n)
    ; bh = set (AD.Mat.zeros 1 n)
    ; bf = set (AD.Mat.create 1 n 3.)
    ; uh = set (AD.Mat.zeros n n)
    ; uf = set (AD.Mat.zeros n n)
    }


  let with_wh_rescaled theta =
    { theta with
      wh =
        Owl_parameters.map
          (fun wh -> AD.Maths.(wh / sqrt (sum ~axis:0 (sqr wh))))
          theta.wh
    }


  let generate_bs ~n ~m =
    Option.map X.n_beg ~f:(fun nb ->
        let nr = n / nb in
        assert (nr = m);
        ( nb
        , Array.init nb ~f:(fun i ->
              let inr = i * nr in
              let rnr = n - ((i + 1) * nr) in
              AD.Maths.(
                transpose
                  (concatenate
                     ~axis:0
                     [| AD.Mat.zeros inr m; AD.Mat.eye nr; AD.Mat.zeros rnr m |]))) ))


  let dyn ~theta =
    let theta = with_wh_rescaled theta in
    let wh = Owl_parameters.extract theta.wh in
    let bh = Owl_parameters.extract theta.bh in
    let bf = Owl_parameters.extract theta.bf in
    let uh = Owl_parameters.extract theta.uh in
    let uf = Owl_parameters.extract theta.uf in
    let n = AD.Mat.col_num bh in
    let m = AD.Mat.row_num wh in
    let beg_bs = generate_bs ~n ~m in
    let default x u =
      let h_pred = x in
      let f = sigma AD.Maths.(bf + (h_pred *@ uf)) in
      let h_hat =
        let hf = AD.Maths.(h_pred * f) in
        AD.Maths.(phi AD.Maths.(bh + (hf *@ uh)) + (u *@ wh))
      in
      AD.Maths.(((F 1. - f) * h_pred) + (f * h_hat))
    in
    fun ~k ~x ~u ->
      match beg_bs with
      | None -> default x u
      | Some (nb, beg_b) ->
        if k < nb then AD.Maths.(x + (u *@ beg_b.(k))) else default x u


  let dyn_x =
    let _dyn_x ~theta =
      let theta = with_wh_rescaled theta in
      let wh = Owl_parameters.extract theta.wh in
      let bf = Owl_parameters.extract theta.bf in
      let bh = Owl_parameters.extract theta.bh in
      let uh = Owl_parameters.extract theta.uh in
      let uf = Owl_parameters.extract theta.uf in
      let n = AD.Mat.col_num bh in
      let id_n = AD.Mat.eye n in
      let default x u =
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
            * (f * d_phi h_hat_pre)))
      in
      fun ~k ~x ~u ->
        match X.n_beg with
        | None -> default x u
        | Some i -> if k < i then id_n else default x u
    in
    Some _dyn_x


  let dyn_u =
    let _dyn_u ~theta =
      let theta = with_wh_rescaled theta in
      let wh = Owl_parameters.extract theta.wh in
      let bf = Owl_parameters.extract theta.bf in
      let bh = Owl_parameters.extract theta.bh in
      let uf = Owl_parameters.extract theta.uf in
      let m = AD.Mat.row_num wh in
      let n = AD.Mat.col_num bh in
      let beg_bs = generate_bs ~n ~m in
      let default x =
        let h_pred = x in
        let f_pre = AD.Maths.(bf + (h_pred *@ uf)) in
        let f = sigma f_pre in
        AD.Maths.(wh * f)
      in
      fun ~k ~x ~u:_ ->
        match beg_bs with
        | None -> default x
        | Some (nb, beg_b) -> if k < nb then beg_b.(k) else default x
    in
    Some _dyn_u
end
