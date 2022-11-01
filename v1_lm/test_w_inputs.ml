open Base
open Owl
open Ilqr_vae
open Vae
open Owl_parameters
open Accessor.O

let in_data_dir = Printf.sprintf "/home/mmcs3/rds/rds-t2-cs156-T7o4pEA8QoU/v1_lm/data/%s"
(*these arrays have shape n_trials * n_time * n_channels*)

(*to be able to read the spikes I added 0.5 to all of them before putting them in npy arrays so we need to undo that first *)

let lm =
  Arr.load_npy (in_data_dir "all_lm_cells.npy")
  |> fun z ->
  let _ = Stdio.printf "%i %i %i" (Arr.shape z).(0) (Arr.shape z).(1) (Arr.shape z).(2) in
  Arr.(z -$ 0.5) |> fun z -> Arr.transpose z ~axis:[| 0; 2; 1 |] |> C.broadcast


let v1 =
  Arr.load_npy (in_data_dir "all_v1_cells.npy")
  |> fun z ->
  let _ = Stdio.printf "%i %i %i" (Arr.shape z).(0) (Arr.shape z).(1) (Arr.shape z).(2) in
  Arr.(z -$ 0.5) |> fun z -> Arr.transpose z ~axis:[| 0; 2; 1 |] |> C.broadcast


let laser_times =
  Arr.load_npy (in_data_dir "all_times.npy")
  |> fun z ->
  let _ = Stdio.printf "%i" (Arr.shape z).(0) in
  z |> C.broadcast


(* 
let v1 =
  Arr.load_npy (in_data_dir "all_v1_cells.npy") |> fun z -> Arr.(z -$ 0.5) |> C.broadcast
 *)

(*in the training set we have : train behaviour, train spikes (all of them) and in the test set we want to evaluate both prediction of heldout spikes and of heldout behaviour *)
let in_dir = Cmdargs.in_dir "-d"
let n = Cmdargs.(get_int "-n" |> force ~usage:"-n [n]")
let m = Cmdargs.(get_int "-m" |> force ~usage:"-m [m]")
let n_cpus = Cmdargs.(get_int "-n_cpus" |> default 76)
let n_fish = Cmdargs.(get_int "-n_fish" |> default 50)
let n_decay = Cmdargs.(get_int "-n_decay" |> default 1)
let n_samples = Cmdargs.(get_int "-n_samples" |> default 1)
let prev_iter = Cmdargs.(get_int "-prev_iter" |> default 0)
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
let optimal_lag, optimal_bin_lag = 0., 0
let n_trials = (Arr.shape lm).(0)
let n_steps = (Arr.shape lm).(1)
let n_train_neurons = (Arr.shape lm).(2)
let _ = Stdio.printf "|| n_train %i %i %i || \n %!" n_trials n_steps n_train_neurons
let n_neurons = n_train_neurons
let batch_size = Int.(n_trials / n_cpus)

type setup =
  { n : int
  ; m : int
  ; n_trials : int
  ; n_steps : int
  ; n_neural : int
  }

let dt = 0.02

let setup =
  C.broadcast' (fun () ->
      let s = { n; m; n_trials; n_steps; n_neural = n_neurons } in
      Misc.save_bin ~out:(in_dir "setup.bin") s;
      s)


(* ----------------------------------------- 
   -- Define model
   ----------------------------------------- *)

module type Setup = sig
  val n : int
  val m : int
  val n_output : int
  val n_trials : int
  val n_steps : int
end

module S = struct
  let n = n
  let m = m
  let n_trials = n_trials
  let n_steps = n_steps
  let n_output = n_neurons
end
(* ----------------------------------------- 
     -- Define model
     ----------------------------------------- *)

module U = Prior.Student (struct
  include S

  let n_beg = n_beg
end)

module L = Likelihood.Poisson (struct
  include S

  let label = "neural"
  let dt = AD.F dt
  let link_function = AD.Maths.exp
  let d_link_function = AD.Maths.exp
  let d2_link_function = AD.Maths.exp
end)

let p x = AD.Maths.(AD.requad x - F 1.)
let d_p x = AD.d_requad x

module D = Dynamics.Driven_Nonlinear (struct
  include S

  let phi = `nonlinear (p, d_p)
  let m_ext = 1

  let u_ext =
    Array.init
      Int.(n_beg + 100)
      ~f:(fun i -> if Int.(i < 75 + n_beg) && Int.(i > 50 + n_beg) then 1. else 0.)
    |> fun z -> Mat.of_array z (-1) 1 |> AD.pack_arr
end)

include Default.Model (U) (D) (L) (S)

(* module Model = Default.Model (U) (D) (L) (S) *)
(* ----------------------------------------- 
   -- Fetch the data and process slightly
   ----------------------------------------- *)

let train_data, test_data =
  let data =
    lazy
      (Array.init n_trials ~f:(fun i ->
           let x =
             Arr.get_slice [ [ i ]; []; [] ] lm
             (* |> Arr.transpose ~axis:[| 0; 2; 1 |] *)
             |> fun z -> Arr.reshape z [| -1; n_neurons |]
           in
           let _ =
             if Int.(i < 10)
             then Mat.save_txt ~out:(in_dir (Printf.sprintf "train_output_%i" i)) x
           in
           let laser_u =
             let t_laser = Arr.(sum' (get_slice [ [ i ] ] laser_times)) in
             let laser_idx = Int.(of_float Float.(50. *. t_laser)) in
             let _ = Stdio.printf "%i %!" laser_idx in
             Array.init n_steps ~f:(fun i ->
                 if Int.(laser_idx >= i) && Int.(laser_idx < i + 8)
                 then (
                   let _ = Stdio.printf "tree %!" in
                   Mat.ones 1 1)
                 else Mat.zeros 1 1)
             |> Mat.concatenate ~axis:0
           in
           let o = AD.pack_arr x in
           let dat = Data.pack o in
           let dat = Data.fill dat ~u:(AD.pack_arr laser_u) ~z:(AD.pack_arr laser_u) in
           let _ =
             if Int.(i < 40)
             then Mat.save_txt ~out:(in_dir (Printf.sprintf "train_laser_%i" i)) laser_u
           in
           dat))
  in
  Data.split_and_distribute
    ~reuse:false
    ~prefix:(in_dir "data")
    ~train:Int.(2 * n_trials / 3)
    data


let _ = C.print_endline "Data generated and broadcasted and saved."

(* ----------------------------------------- 
   -- Initialise parameters and train
   ----------------------------------------- *)

let init_prms =
  C.broadcast' (fun () ->
      match init_prms_file with
      | Some f -> Misc.load_bin f
      | None ->
        let generative =
          let n = S.n
          and m = S.m in
          let prior = U.init ~spatial_std:1.0 ~nu:10. learned in
          let likelihood : L.P.p = L.init learned in
          let dynamics = D.init learned in
          Generative.P.{ prior; dynamics; likelihood }
        in
        let recognition = R.init learned in
        Model.init generative recognition)


(* let _ = test_preds init_prms "trained_fin" train_data *)

let final_prms =
  let in_each_iteration ~prms k =
    if Int.(k % 200 = 0)
    then C.root_perform (fun () -> Model.P.save_to_files ~prefix:(in_dir "final") prms);
    if Int.(k % 2000 = 0)
    then
      Model.save_results
        ~prefix:(in_dir (Printf.sprintf "train_%i" k))
        ~prms
        ~n_to_save:1
        (Array.sub train_data ~pos:0 ~len:2)
    (* if Int.((k % 500 = 0) && (k> 100))
      then (test_preds prms "test" test_data;
    test_preds prms "train" train_data) *)
  in
  Model.train
    ~mini_batch:1
    ~n_posterior_samples:(fun _ -> n_samples)
    ~max_iter:Cmdargs.(get_int "-max_iter" |> default 16500)
    ~save_progress_to:(1, 2000, in_dir "progress") (* ~conv_threshold:1E-6 *)
    ~in_each_iteration
      (* ~recycle_u:false *)
      (* ~regularizer:reg *)
    ~learning_rate:
      (`of_iter
        (fun k ->
          if k < Int.(of_float (0.01 *. Float.of_int prev_iter))
          then 0.
          else Float.(eta / (1. + sqrt (of_int Int.(k + prev_iter) / of_int n_decay)))))
    ~init_prms
    train_data
