open Base
open Owl
open Ilqr_vae
open Vae
open Owl_parameters
open Accessor.O

let in_data_dir = Printf.sprintf "/home/mmcs3/rds/rds-t2-cs156-T7o4pEA8QoU/v1_lm/data/%s"
let in_dir = Cmdargs.in_dir "-d"

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


let init_prms_file = Cmdargs.(get_string "-init_prms" |> force ~usage:"-init_prms")
let n = Cmdargs.(get_int "-n" |> force ~usage:"-n [n]")
let m = Cmdargs.(get_int "-m" |> force ~usage:"-m [m]")
let n_cpus = Cmdargs.(get_int "-n_cpus" |> default 76)
let n_fish = Cmdargs.(get_int "-n_fish" |> default 50)
let n_decay = Cmdargs.(get_int "-n_decay" |> default 1)
let n_samples = Cmdargs.(get_int "-n_samples" |> default 1)
let prev_iter = Cmdargs.(get_int "-prev_iter" |> default 0)
let n_beg = n / m
let log_of_2 = AD.Maths.(log (F 2.))
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

(* module D = Dynamics.MGU2 (struct
  include S

  let phi x = AD.Maths.(AD.requad x - F 1.)
  let d_phi x = AD.d_requad x
  let sigma x = AD.Maths.sigmoid x
  let d_sigma x = AD.Maths.(exp (neg x) / sqr (F 1. + exp (neg x)))
  let n_beg = n_beg
end) *)
let p x = AD.Maths.(AD.requad x - F 1.)
let d_p x = AD.d_requad x

(* module D = Dynamics.Nonlinear (struct
  include S

  let phi = `nonlinear (p, d_p)
end) *)

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

(* ----------------------------------------- 
   -- Initialise parameters and train
   ----------------------------------------- *)

let prms = C.broadcast' (fun () -> Misc.load_bin init_prms_file)

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
  Data.split_and_distribute ~reuse:false ~prefix:(in_dir "data") ~train:n_trials data


(* let data_to_fit = Misc.load_bin (in_dir "data.train.bin") *)

let data_to_fit = Array.sub train_data ~pos:0 ~len:400

let fitted_data =
  let open Vae_typ.P in
  Array.foldi data_to_fit ~init:[] ~f:(fun i accu dat_trial ->
      if Int.(i % C.n_nodes = C.rank)
      then (
        let mu =
          Model.R.posterior_mean ~gen_prms:prms.generative prms.recognition dat_trial
        in
        let true_u = Data.u dat_trial in
        let u = AD.Maths.(reshape (get_slice [ [] ] mu) [| -1; m |]) in
        let id = Data.id dat_trial in
        let sample = Model.G.sample ~prms:prms.generative in
        let data = sample ~id ~pre:true (`some u) in
        (* let z = Data.z data in
        let o = Data.o data in
        let u = Data.u data in
        let remove_n_beg = AD.Maths.get_slice [ [ Model.G.n_beg - 1; -1 ] ] in
        let fitted_dat = Data.fill ~u ~z o in
        let o = Data.o data in *)
        data :: accu)
      else accu)
  |> C.gather
  |> fun v -> C.broadcast' (fun _ -> Array.of_list (List.concat (Array.to_list v)))


let _ = Data.save ~prefix:(in_dir "analyses/fit") G.L.save_output fitted_data
let _ = Data.save ~prefix:(in_dir "analyses/data") G.L.save_output data_to_fit

(*  *)
let n_train_neurons = 9

let test_preds ~label ~n_train_neurons (prms : Model.P.p) data =
  let prepared_data =
    Array.map data ~f:(fun d ->
        let o = Data.o d in
        let u = Data.u d in
        let train_o = AD.Maths.get_slice [ []; [ 0; n_train_neurons - 1 ] ] o in
        let test_o =
          if n_train_neurons = n_neurons
          then train_o
          else AD.Maths.get_slice [ []; [ n_train_neurons; -1 ] ] o
        in
        Data.fill ~u ~z:test_o (Data.pack train_o))
  in
  let n_test_neurons = n_neurons - n_train_neurons in
  let prms = C.broadcast prms in
  let in_dir' s = in_dir Printf.(sprintf "%s" s) in
  let masked_likelihood ~prms =
    let open Likelihood.Poisson_P in
    let c = Owl_parameters.extract prms.c in
    let bias = Owl_parameters.extract prms.bias in
    let gain = Owl_parameters.extract prms.gain in
    Likelihood.Poisson_P.
      { c = pinned (AD.Maths.get_slice [ [ 0; n_train_neurons - 1 ] ] c)
      ; bias = pinned (AD.Maths.get_slice [ []; [ 0; n_train_neurons - 1 ] ] bias)
      ; c_mask = None
      ; gain = pinned (AD.Maths.get_slice [ []; [ 0; n_train_neurons - 1 ] ] gain)
      }
  in
  let masked_prms =
    C.broadcast' (fun () ->
        let generative =
          let n = S.n
          and m = S.m in
          let prior = prms.generative.prior in
          let likelihood : L.P.p = masked_likelihood ~prms:prms.generative.likelihood in
          let dynamics = prms.generative.dynamics in
          Generative.P.{ prior; dynamics; likelihood }
        in
        let recognition = prms.recognition in
        Model.init generative recognition)
  in
  Array.foldi prepared_data ~init:[] ~f:(fun i accu dat_trial ->
      if Int.(i % C.n_nodes = C.rank)
      then (
        let _ = Stdio.printf "%i %!" i in
        (*partial_test_data has the partial spikes as outputs and the full spikes as latent*)
        try
          let open Likelihood.Poisson_P in
          let mu =
            Model.R.posterior_mean
              ~gen_prms:masked_prms.generative
              prms.recognition
              dat_trial
          in
          let sample = Model.G.sample ~prms:prms.generative in
          let data = sample ~id:(Data.id dat_trial) ~pre:true (`some mu) in
          let fitted_o =
            AD.Maths.get_slice
              [ [ Model.G.n_beg - 1; -1 ]; [ -n_test_neurons; -1 ] ]
              (Data.o data)
          in
          let true_o = Data.z dat_trial in
          let flat_rate =
            Mat.mean ~axis:0 (AD.unpack_arr true_o)
            |> fun z ->
            Mat.((z * ones (AD.Mat.row_num true_o) (AD.Mat.col_num true_o)) +$ 0.00001)
            |> AD.pack_arr
          in
          (AD.Maths.(F dt * fitted_o), flat_rate, true_o) :: accu
        with
        | e ->
          Stdio.printf
            "Trial %i failed with some exception in compute likelihood val : %s"
            i
            (Exn.to_string e);
          accu)
      else accu)
  |> C.gather
  |> fun v ->
  C.root_perform (fun () ->
      (* try *)
      let v = v |> Array.to_list |> List.concat |> Array.of_list in
      let _ = Stdio.printf "post v %!" in
      let pred_lambda =
        Array.map v ~f:(fun (a, _, _) -> a) |> AD.Maths.concatenate ~axis:0
      in
      let _ = Mat.save_txt ~out:(in_dir' "LL_pred_lambda") (AD.unpack_arr pred_lambda) in
      let true_spikes =
        Array.map v ~f:(fun (_, _, a) -> a) |> AD.Maths.concatenate ~axis:0
      in
      let _ = Mat.save_txt ~out:(in_dir' "LL_true_spikes") (AD.unpack_arr true_spikes) in
      let flat_rate =
        Mat.mean ~axis:0 (AD.unpack_arr true_spikes)
        |> fun z ->
        Mat.(
          (z * ones (AD.Mat.row_num true_spikes) (AD.Mat.col_num true_spikes))
          +$ 0.00000001)
        |> AD.pack_arr
      in
      let ll_neuron =
        AD.Maths.((true_spikes * log pred_lambda) - pred_lambda) |> AD.Maths.sum'
      in
      let ll_flat =
        AD.Maths.((true_spikes * log flat_rate) - flat_rate) |> AD.Maths.sum'
      in
      let n_sp = AD.Maths.sum' true_spikes in
      let normalized_ll = AD.Maths.((ll_neuron - ll_flat) / n_sp / log_of_2) in
      Mat.save_txt
        ~append:true
        ~out:(in_dir' label)
        (Mat.of_array [| AD.unpack_flt normalized_ll |] 1 (-1)))


(* with
      | e -> Stdio.printf "error %s" (Exn.to_string e)) *)

let n_trials = 399

let _ =
  test_preds
    ~label:"LL_heldout"
    ~n_train_neurons:8
    prms
    (Array.sub data_to_fit ~pos:0 ~len:n_trials);
  test_preds
    ~n_train_neurons:10
    ~label:"LL_full"
    prms
    (Array.sub data_to_fit ~pos:0 ~len:n_trials)
