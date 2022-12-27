open Base
open Owl
open Ilqr_vae
open Vae
open Owl_parameters
open Accessor.O

let in_data_dir = Printf.sprintf "/home/mmcs3/rds/rds-t2-cs156-T7o4pEA8QoU/v1_lm/data/%s"
let in_dir = Cmdargs.in_dir "-d"
let include_stim = Cmdargs.check "-include_stim"
let include_laser = Cmdargs.check "-include_laser"

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
let go_ids = Arr.load_npy (in_data_dir "go_ids.npy")
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
let lm_only = Cmdargs.check "-lm"
let batch_size = Int.(n_trials / n_cpus)
let n_lm_neurons = (Arr.shape lm).(2)
let n_v1_neurons = (Arr.shape v1).(2)
let n_neurons = if lm_only then n_lm_neurons else n_v1_neurons
let neural_obs = if lm_only then lm else v1
let _ = Stdio.printf "%i %i %!" n_lm_neurons n_v1_neurons

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

let p x = AD.Maths.(AD.requad x - F 1.)
let d_p x = AD.d_requad x

module D = Dynamics.Nonlinear (struct
  include S

  let phi = `nonlinear (p, d_p)
  let m_ext = 2
end)

(* let u_stim =
  Array.init n_trials ~f:(fun n ->
      Array.init
        Int.(n_beg + 100)
        ~f:(fun i ->
          if Int.(i < 75 + n_beg) && Int.(i > 50 + n_beg)
          then Arr.get_slice [ [ n ]; [] ] go_ids
          else Mat.zeros 1 2)
      |> fun z -> Mat.concatenate ~axis:0 z |> AD.pack_arr) *)
let u_ext =
  Array.init n_trials ~f:(fun n ->
      let laser_u =
        let t_laser = Arr.(sum' (get_slice [ [ n ] ] laser_times)) in
        let laser_idx = Int.(of_float Float.(50. *. t_laser)) in
        Array.init
          Int.(n_steps + n_beg)
          ~f:(fun i ->
            if Int.(laser_idx >= i + n_beg) && Int.(laser_idx < i + 8 + n_beg)
            then Mat.ones 1 1
            else Mat.zeros 1 1)
        |> Mat.concatenate ~axis:0
      in
      let stim_u =
        Array.init
          Int.(n_beg + 100)
          ~f:(fun i ->
            if Int.(i < 75 + n_beg) && Int.(i > 50 + n_beg)
            then Arr.get_slice [ [ n ]; [] ] go_ids
            else Mat.zeros 1 2)
        |> fun z -> Mat.concatenate ~axis:0 z
      in
      let ext_u =
        if include_laser && include_stim
        then AD.Maths.concatenate ~axis:1 [| AD.pack_arr stim_u; AD.pack_arr laser_u |]
        else if include_laser
        then AD.pack_arr laser_u
        else if include_stim
        then AD.pack_arr stim_u
        else AD.Mat.zeros Int.(n_beg + 100) 2
      in
      ext_u)


let u_info =
  Array.init n_trials ~f:(fun n ->
      let laser_u =
        let t_laser = Arr.(sum' (get_slice [ [ n ] ] laser_times)) in
        let laser_idx = Int.(of_float Float.(50. *. t_laser)) in
        Array.init
          Int.(n_steps + n_beg)
          ~f:(fun i ->
            if Int.(laser_idx >= i + n_beg) && Int.(laser_idx < i + 8 + n_beg)
            then Mat.ones 1 1
            else Mat.zeros 1 1)
        |> Mat.concatenate ~axis:0
      in
      let stim_u =
        Array.init
          Int.(n_beg + 100)
          ~f:(fun i ->
            if Int.(i < 75 + n_beg) && Int.(i > 50 + n_beg)
            then Arr.get_slice [ [ n ]; [] ] go_ids
            else Mat.zeros 1 2)
        |> fun z -> Mat.concatenate ~axis:0 z
      in
      AD.pack_arr stim_u, AD.pack_arr laser_u)


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
             Arr.get_slice [ [ i ]; []; [] ] neural_obs
             (* |> Arr.transpose ~axis:[| 0; 2; 1 |] *)
             |> fun z -> Arr.reshape z [| -1; n_neurons |]
           in
           let _ =
             if Int.(i < 10)
             then Mat.save_txt ~out:(in_dir (Printf.sprintf "train_output_%i" i)) x
           in
           let extu = Some u_ext.(i) in
           let o = AD.pack_arr x in
           let dat =
             Data.pack ~ext_u:extu o
             (*here we specify the observations + known external inputs if any*)
           in
           let stim_u, laser_u = u_info.(i) in
           let dat = Data.fill dat ~u:laser_u ~z:stim_u in
           let _ =
             if Int.(i < 40)
             then
               Mat.save_txt
                 ~out:(in_dir (Printf.sprintf "ext_stim_%i" i))
                 (AD.unpack_arr u_ext.(i))
           in
           dat))
  in
  Data.split_and_distribute
    ~reuse:false
    ~prefix:(in_dir "data")
    ~train:Int.(2 * S.n_trials / 3)
    data


(*decide how many of the observations we try to fit*)
(* let data_to_fit = Array.sub test_data ~pos:0 ~len:100 *)

(*this is a function to compute the log likelihood of the data (on all neurons or heldout neurons)
We might want to also compute how well the inferred inputs match the true inputs there*)
let test_preds ~label ~n_train_neurons (prms : Model.P.p) data =
  let prepared_data =
    Array.mapi data ~f:(fun i d ->
        let o = Data.o d in
        let laser_u = Data.u d in
        let stim_u = Data.z d in
        (*here if we train on some neurons on will then test on other (heldout) neurons*)
        let train_o = AD.Maths.get_slice [ []; [ 0; n_train_neurons - 1 ] ] o in
        let test_o =
          if n_train_neurons = n_neurons
          then train_o
          else AD.Maths.get_slice [ []; [ n_train_neurons; -1 ] ] o
        in
        let ext_u = Some u_ext.(i) in
        Data.fill
          ~u:(AD.Maths.concatenate ~axis:1 [| laser_u; stim_u |])
          ~z:test_o
          (Data.pack ~ext_u train_o))
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
        (* try *)
        let open Likelihood.Poisson_P in
        let mu =
          Model.R.posterior_mean
            ~gen_prms:masked_prms.generative
            prms.recognition
            dat_trial
        in
        let sample = Model.G.sample ~prms:prms.generative in
        let u = Data.u dat_trial in
        let ext_u =
          if not (include_stim || include_laser)
          then None
          else if include_stim
          then
            if not include_laser
            then Some (AD.Maths.get_slice [ []; [ 0; 1 ] ] u)
            else Some u
          else Some (AD.Maths.get_slice [ []; [ 1; -1 ] ] u)
        in
        let data = sample ~id:(Data.id dat_trial) ~ext_u ~pre:true (`some mu) in
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
        (AD.Maths.(F dt * fitted_o), flat_rate, true_o) :: accu)
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


let n_trials = 100

let _ =
  test_preds ~label:"LL_heldout" ~n_train_neurons:Int.(n_train_neurons - 2) prms test_data;
  test_preds ~n_train_neurons ~label:"LL_full" prms test_data


let fitted_data ~n_trials data =
  let open Vae_typ.P in
  Array.foldi (Array.sub data ~pos:0 ~len:n_trials) ~init:[] ~f:(fun i accu dat_trial ->
      if Int.(i % C.n_nodes = C.rank)
      then (
        let mu =
          Model.R.posterior_mean ~gen_prms:prms.generative prms.recognition dat_trial
        in
        let true_u = Data.u dat_trial in
        let u = AD.Maths.(reshape (get_slice [ [] ] mu) [| -1; m |]) in
        let id = Data.id dat_trial in
        let sample = Model.G.sample ~prms:prms.generative in
        let u = Data.u dat_trial in
        let data = sample ~id ~ext_u:(Some u_ext.(i)) ~pre:true (`some mu) in
        data :: accu)
      else accu)
  |> C.gather
  |> fun v -> C.broadcast' (fun _ -> Array.of_list (List.concat (Array.to_list v)))


let _ =
  Data.save
    ~prefix:(in_dir "analyses/train_fit")
    G.L.save_output
    (fitted_data ~n_trials:100 train_data);
  Data.save
    ~prefix:(in_dir "analyses/test_fit")
    G.L.save_output
    (fitted_data ~n_trials:100 test_data)


let _ =
  Data.save
    ~prefix:(in_dir "analyses/train_data")
    G.L.save_output
    (Array.sub ~pos:0 ~len:100 train_data);
  Data.save
    ~prefix:(in_dir "analyses/test_data")
    G.L.save_output
    (Array.sub ~pos:0 ~len:100 test_data)

(* open Base
open Owl
open Ilqr_vae
open Vae
open Owl_parameters
open Accessor.O

let in_data_dir = Printf.sprintf "/home/mmcs3/rds/rds-t2-cs156-T7o4pEA8QoU/v1_lm/data/%s"
let in_dir = Cmdargs.in_dir "-d"
let include_stim = Cmdargs.check "-include_stim"
let include_laser = Cmdargs.check "-include_laser"

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
let go_ids = Arr.load_npy (in_data_dir "go_ids.npy")
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
let lm_only = Cmdargs.check "-lm"
let batch_size = Int.(n_trials / n_cpus)
let n_lm_neurons = (Arr.shape lm).(2)
let n_v1_neurons = (Arr.shape v1).(2)
let n_neurons = if lm_only then n_lm_neurons else n_v1_neurons
let neural_obs = if lm_only then lm else v1
let _ = Stdio.printf "%i %i %!" n_lm_neurons n_v1_neurons

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
  let m_ext = if include_stim & include_laser then 3 else 2
end) *)

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

module D = Dynamics.Nonlinear (struct
  include S

  let phi = `nonlinear (p, d_p)
  let m_ext = 2
end)

let u_stim =
  Array.init n_trials ~f:(fun n ->
      Array.init
        Int.(n_beg + 100)
        ~f:(fun i ->
          if Int.(i < 75 + n_beg) && Int.(i > 50 + n_beg)
          then Arr.get_slice [ [ n ]; [] ] go_ids
          else Mat.zeros 1 2)
      |> fun z -> Mat.concatenate ~axis:0 z |> AD.pack_arr)


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
             Arr.get_slice [ [ i ]; []; [] ] neural_obs
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
             Array.init
               Int.(n_steps + n_beg)
               ~f:(fun i ->
                 if Int.(i >= Int.(laser_idx + n_beg)) && Int.(i < laser_idx + 8 + n_beg)
                 then (
                   let _ = Stdio.printf "tree %!" in
                   Mat.ones 1 1)
                 else Mat.zeros 1 1)
             |> Mat.concatenate ~axis:0
           in
           let o = AD.pack_arr x in
           let ext_u =
             if (not include_stim) && not include_laser
             then None
             else if include_stim
             then Some u_stim.(i)
             else Some (AD.pack_arr laser_u)
             (* (Some u_stim) *)
           in
           let dat = Data.pack ~ext_u o in
           let dat = Data.fill dat ~u:(AD.pack_arr laser_u) ~z:u_stim.(i) in
           let _ =
             if Int.(i < 40)
             then Mat.save_txt ~out:(in_dir (Printf.sprintf "train_laser_%i" i)) laser_u
           in
           dat))
  in
  Data.split_and_distribute ~reuse:false ~prefix:(in_dir "data") ~train:n_trials data


(* let data_to_fit = Misc.load_bin (in_dir "data.train.bin") *)

let data_to_fit = Array.sub train_data ~pos:0 ~len:800

let test_preds ~label ~n_train_neurons (prms : Model.P.p) data =
  let prepared_data =
    Array.map data ~f:(fun d ->
        let o = Data.o d in
        let laser_u = Data.u d in
        let stim_u = Data.z d in
        let train_o = AD.Maths.get_slice [ []; [ 0; n_train_neurons - 1 ] ] o in
        let test_o =
          if n_train_neurons = n_neurons
          then train_o
          else AD.Maths.get_slice [ []; [ n_train_neurons; -1 ] ] o
        in
        let ext_u =
          if (not include_stim) && not include_laser
          then None
          else if include_stim
          then Some stim_u
          else Some laser_u
        in
        Data.fill
          ~u:(AD.Maths.concatenate ~axis:1 [| laser_u; stim_u |])
          ~z:test_o
          (Data.pack ~ext_u train_o))
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
        (* try *)
        let open Likelihood.Poisson_P in
        let mu =
          Model.R.posterior_mean
            ~gen_prms:masked_prms.generative
            prms.recognition
            dat_trial
        in
        let sample = Model.G.sample ~prms:prms.generative in
        let u = Data.u dat_trial in
        let ext_u =
          if not (include_stim || include_laser)
          then None
          else if include_stim
          then
            if not include_laser
            then Some (AD.Maths.get_slice [ []; [ 0; 1 ] ] u)
            else Some u
          else Some (AD.Maths.get_slice [ []; [ 1; -1 ] ] u)
        in
        let data = sample ~id:(Data.id dat_trial) ~ext_u ~pre:true (`some mu) in
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
        (* with
        | e ->
          Stdio.printf
            "Trial %i failed with some exception in compute likelihood val : %s"
            i
            (Exn.to_string e); *)
        (* accu) *))
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

let n_trials = 600

let _ =
  test_preds
    ~label:"LL_heldout"
    ~n_train_neurons:Int.(n_train_neurons - 2)
    prms
    (Array.sub data_to_fit ~pos:0 ~len:n_trials);
  test_preds
    ~n_train_neurons
    ~label:"LL_full"
    prms
    (Array.sub data_to_fit ~pos:0 ~len:n_trials)


let fitted_data =
  let open Vae_typ.P in
  Array.foldi (Array.sub data_to_fit ~pos:0 ~len:100) ~init:[] ~f:(fun i accu dat_trial ->
      if Int.(i % C.n_nodes = C.rank)
      then (
        let mu =
          Model.R.posterior_mean ~gen_prms:prms.generative prms.recognition dat_trial
        in
        let true_u = Data.u dat_trial in
        let u = AD.Maths.(reshape (get_slice [ [] ] mu) [| -1; m |]) in
        let id = Data.id dat_trial in
        let sample = Model.G.sample ~prms:prms.generative in
        let u = Data.u dat_trial in
        let ext_u =
          if not (include_stim || include_laser)
          then None
          else if include_stim
          then
            if not include_laser
            then Some (AD.Maths.get_slice [ []; [ 0; 1 ] ] u)
            else Some u
          else Some (AD.Maths.get_slice [ []; [ 1; -1 ] ] u)
        in
        let data = sample ~id ~ext_u ~pre:true (`some mu) in
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

let _ =
  Data.save
    ~prefix:(in_dir "analyses/data")
    G.L.save_output
    (Array.sub ~pos:0 ~len:100 data_to_fit)

(*let _ =
  Model.save_results
    ~prefix:(in_saving_dir (Printf.sprintf "tmp/fin"))
    ~n_to_save:n_tot_trials
    ~prms:final_prms
    train_data
    
    
    let _ =
      C.root_perform (fun () ->
          Array.init n_tot_trials ~f:(fun i ->
              let m =
                Mat.load_txt (in_saving_dir (Printf.sprintf "tmp/fin.mean.o.%i.tail" i))
              in
              Arr.reshape m [| 1; Mat.row_num m; Mat.col_num m |])
          |> Arr.concatenate ~axis:0
          |> fun a ->
          Arr.save_npy (in_saving_dir (Printf.sprintf "fin_1.mean.tail")) a;
          Array.init n_tot_trials ~f:(fun i ->
              let m = Mat.load_txt (in_saving_dir (Printf.sprintf "tmp/fin.mean.u.%i" i)) in
              Arr.reshape m [| 1; Mat.row_num m; Mat.col_num m |])
          |> Arr.concatenate ~axis:0
          |> fun a ->
          Arr.save_npy (in_saving_dir (Printf.sprintf "fin.mean.u")) a;
          Array.init
            Int.(n_tot_trials / 100)
            ~f:(fun i ->
              let m = Mat.load_txt (in_saving_dir (Printf.sprintf "tmp/fin.mean.z.%i" i)) in
              Arr.reshape m [| 1; Mat.row_num m; Mat.col_num m |])
          |> Arr.concatenate ~axis:0
          |> fun a -> Arr.save_npy (in_saving_dir (Printf.sprintf "part_fin.mean.z")) a)
          *) *)
