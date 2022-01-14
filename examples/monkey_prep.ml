open Base
open Owl
open Ilqr_vae
open Owl_parameters
open Accessor.O

type 'a o =
  { spikes : 'a
  ; t_mov : int
  ; hand_vel : 'a
  }

let in_data_dir = Cmdargs.in_dir "-data"
(*these arrays have shape n_trials * n_time * n_channels*)

(*to be able to read the spikes I added 0.5 to all of them before putting them in npy arrays so we need to undo that first *)

(*here, trials are of length 250 : the first 50ms (e.g 10 time bins are before target onset) then the rest is before go cue and move onset
the diff_move_tgt file tells us how long has elapsed (in seconds) between tgt onset and move onset
In terms of time bins we can get those by dividing by dt, and we want to keep only trials where this difference is < 190 time bins*)
let spikes =
  Arr.load_npy (in_data_dir "all_spikes_float.npy")
  |> fun z -> Arr.(z -$ 0.5) |> C.broadcast


let hand_vel = Arr.load_npy (in_data_dir "hand_vel.npy") |> C.broadcast
let diff_mov_target = Arr.load_npy (in_data_dir "diff_move_tgt.npy") |> C.broadcast

(*in the training set we have : train behaviour, train spikes (all of them) and in the test set we want to evaluate both prediction of heldout spikes and of heldout behaviour *)
let in_dir = Cmdargs.in_dir "-d"
let n = Cmdargs.(get_int "-n" |> force ~usage:"-n [n]")
let m = Cmdargs.(get_int "-m" |> force ~usage:"-m [m]")
let n_beg = n / m

let logfact k =
  let rec iter k accu =
    if k <= 1 then accu else iter (k - 1) Float.(accu + log (of_int k))
  in
  iter k 0.


let log_of_2 = AD.Maths.(log (F 2.))
let init_prms_file = Cmdargs.(get_string "-init_prms")
let n_decay = Cmdargs.(get_int "-n_decay" |> default 1)
let prev_iter = Cmdargs.(get_int "-prev_iter" |> default 0)
let eta = Cmdargs.(get_float "-eta" |> default 0.01)
let reuse = Cmdargs.get_string "-reuse"
let untie = Cmdargs.check "-untie"
let smooth = Cmdargs.check "-smooth"
let optimal_lag, optimal_bin_lag = 0., 0
let n_trials = (Arr.shape spikes).(0)
let n_steps = (Arr.shape spikes).(1)
let n_train_neurons = (Arr.shape spikes).(2) - 45
let n_test_neurons = 45
let _ = Stdio.printf "|| n_train %i, n_test %i || \n %!" n_train_neurons n_test_neurons
let n_neurons = n_train_neurons + n_test_neurons
let n_hand = 2

type setup =
  { n : int
  ; m : int
  ; n_trials : int
  ; n_steps : int
  ; n_neural : int
  ; n_hand : int
  }

let dt = 0.005

let compute_R2 ~lambda x x' =
  let y = x' in
  let nx = Mat.col_num x in
  let c =
    let xt = Mat.transpose x in
    let xtx_inv = Linalg.D.linsolve Mat.((xt *@ x) + (lambda $* eye nx)) (Mat.eye nx) in
    Mat.(xtx_inv *@ xt *@ y)
  in
  let new_x = Mat.(x *@ c) in
  let residuals = Mat.(new_x - x') |> Mat.l2norm_sqr' in
  let sstot = Mat.(x' - mean ~axis:0 x') |> Mat.l2norm_sqr' in
  1. -. (residuals /. sstot), c


let setup =
  C.broadcast' (fun () ->
      match reuse with
      | Some _ -> Misc.load_bin (in_dir "setup.bin")
      | None ->
        let s = { n; m; n_trials; n_steps; n_neural = n_neurons; n_hand } in
        Misc.save_bin ~out:(in_dir "setup.bin") s;
        s)


(* ----------------------------------------- 
   -- Define model
   ----------------------------------------- *)

module U = Prior.Student (struct
  let n_beg = n_beg
  let m = m
end)

module L = Likelihood.Poisson (struct
  let label = "neural"
  let dt = AD.F dt
  let link_function = AD.Maths.exp
  let d_link_function = AD.Maths.exp
  let d2_link_function = AD.Maths.exp
  let n_output = n_neurons
  let n = n
end)

module D = Dynamics.MGU2 (struct
  let n = n
  let m = m
  let phi x = AD.Maths.(AD.requad x - F 1.)
  let d_phi = AD.d_requad
  let sigma x = AD.Maths.sigmoid x

  let d_sigma x =
    let tmp = AD.Maths.(exp (neg x)) in
    AD.Maths.(tmp / sqr (F 1. + tmp))


  let n_beg = Some n_beg
end)

module X = struct
  let n = setup.n
  let m = setup.m
  let n_steps = setup.n_steps
  let diag_time_cov = false
  let n_beg = Some n_beg
end

module Model = Default.Model (U) (D) (L) (X)

(* ----------------------------------------- 
   -- Fetch the data and process slightly
   ----------------------------------------- *)

let squash x = Mat.(signum x * log (1. $+ abs x))
let unsquash x = Mat.(signum x * (exp (abs x) -$ 1.))

let train_data, test_data =
  let data =
    Array.init n_trials ~f:(fun i ->
        let x =
          Arr.get_slice [ [ i ] ] spikes |> fun z -> Arr.reshape z [| -1; n_neurons |]
        in
        let y =
          Arr.get_slice [ [ i ] ] hand_vel |> fun z -> Arr.reshape z [| -1; n_hand |]
        in
        let t_mov =
          Arr.get_slice [ [ i ] ] diff_mov_target
          |> fun z -> Arr.sum' z |> fun t -> Float.(t /. 0.05) |> Int.of_float
        in
        (* let o = AD.pack_arr x in *)
        let o = { spikes = AD.pack_arr x; hand_vel = AD.pack_arr y; t_mov } in
        Data.pack o)
  in
  Data.split_and_distribute
    ~reuse:false
    ~prefix:(in_dir "data")
    ~train:n_trials
    (lazy data)


(* ----------------------------------------- 
   -- Initialise parameters and train
   ----------------------------------------- *)
open Model

let init_prms =
  C.broadcast' (fun () ->
      match init_prms_file with
      | Some f -> Misc.load_bin f
      | None ->
        C.broadcast' (fun () ->
            let generative =
              let prior = U.init ~spatial_std:1.0 ~nu:20. learned in
              let dynamics = D.init learned in
              let likelihood = L.init learned in
              Generative.P.{ prior; dynamics; likelihood }
            in
            let recognition = R.init learned in
            Model.init generative recognition))


let regularizer ~prms =
  D.default_regularizer ~lambda:1E-5 prms.Vae.P.generative.Generative.P.dynamics


let train_data_no_id = Array.map train_data ~f:(fun x -> Data.pack (Data.o x).spikes)

let final_prms =
  let in_each_iteration ~prms k =
    (* if Int.(k % 10 = 0) then Model.P.save_to_files ~prefix:(in_dir "final") ~prms; *)
    if Int.(k % 200 = 0)
    then Model.save_results ~prefix:(in_dir "final") ~prms train_data_no_id
  in
  Model.train
    ~n_posterior_samples:(fun _ -> 10)
    ~max_iter:Cmdargs.(get_int "-max_iter" |> default 40000)
    ~save_progress_to:(10, 200, in_dir "progress")
    ~in_each_iteration
    ~learning_rate:(`of_iter (fun k -> Float.(0.004 / (1. + sqrt (of_int k / 1.)))))
    ~regularizer
    ~init_prms
    train_data_no_id
