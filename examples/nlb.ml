open Base
open Owl
open Ilqr_vae
open Variational
open Owl_parameters
open Accessor.O

let in_data_dir = Cmdargs.in_dir "-data"
(*these arrays have shape n_trials * n_time * n_channels*)

(*to be able to read the spikes I added 0.5 to all of them before putting them in npy arrays so we need to undo that first *)

let train_behaviour = Arr.load_npy (in_data_dir "train_beh.npy") |> C.broadcast
let val_behaviour = Arr.load_npy (in_data_dir "eval_beh.npy") |> C.broadcast

let train_spikes_in =
  Arr.load_npy (in_data_dir "train_spikes_in.npy")
  |> fun z -> Arr.(z -$ 0.5) |> C.broadcast


let eval_spikes_in =
  Arr.load_npy (in_data_dir "eval_spikes_in.npy")
  |> fun z -> Arr.(z -$ 0.5) |> C.broadcast


let test_spikes_in =
  Arr.load_npy (in_data_dir "test_spikes.npy") |> fun z -> Arr.(z -$ 0.5) |> C.broadcast


let train_spikes_out =
  Arr.load_npy (in_data_dir "train_spikes_out.npy")
  |> fun z -> Arr.(z -$ 0.5) |> C.broadcast


let eval_spikes_out =
  Arr.load_npy (in_data_dir "eval_spikes_out.npy")
  |> fun z -> Arr.(z -$ 0.5) |> C.broadcast


let train_spikes = Arr.concatenate ~axis:2 [| train_spikes_in; train_spikes_out |]
let val_spikes = Arr.concatenate ~axis:2 [| eval_spikes_in; eval_spikes_out |]

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
let n_trials = (Arr.shape train_spikes).(0)
let n_test_trials = (Arr.shape val_spikes).(0)
let n_steps = (Arr.shape train_spikes).(1) - optimal_bin_lag
let n_train_neurons = (Arr.shape train_spikes).(2) - 45
let n_test_neurons = 45
let _ = Stdio.printf "|| n_train %i, n_test %i || \n %!" n_train_neurons n_test_neurons
let n_neurons = n_train_neurons + n_test_neurons
let n_hand = (Arr.shape train_behaviour).(2)

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
      | Some _ -> Misc.read_bin (in_dir "setup.bin")
      | None ->
        let s = { n; m; n_trials; n_steps; n_neural = n_neurons; n_hand } in
        Misc.save_bin ~out:(in_dir "setup.bin") s;
        s)


(* ----------------------------------------- 
   -- Define model
   ----------------------------------------- *)

module U = Priors.Student (struct
  let n_beg = Some n_beg
end)

module L = Likelihoods.Poisson (struct
  let label = "neural"
  let dt = AD.F dt
  let link_function = AD.Maths.exp
  let d_link_function = AD.Maths.exp
  let d2_link_function = AD.Maths.exp
end)

module D = Dynamics.MGU2 (struct
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

module Model = VAE (U) (D) (L) (X)
module Model_neural = VAE (U) (D) (L) (X)

(* ----------------------------------------- 
   -- Fetch the data and process slightly
   ----------------------------------------- *)

let squash x = Mat.(signum x * log (1. $+ abs x))
let unsquash x = Mat.(signum x * (exp (abs x) -$ 1.))

let train_data =
  Array.init n_trials ~f:(fun i ->
      let x =
        Arr.get_slice [ [ i ] ] train_spikes |> fun z -> Arr.reshape z [| -1; n_neurons |]
      in
      let y =
        Arr.get_slice [ [ i ] ] train_behaviour |> fun z -> Arr.reshape z [| -1; n_hand |]
      in
      let open Likelihoods.Pair_P in
      { u = None; z = None; o = { fst = AD.pack_arr x; snd = AD.pack_arr y } })
  |> C.broadcast


let val_data =
  Array.init n_test_trials ~f:(fun i ->
      let x =
        Arr.get_slice [ [ i ] ] val_spikes |> fun z -> Arr.reshape z [| -1; n_neurons |]
      in
      let y =
        Arr.get_slice [ [ i ] ] val_behaviour |> fun z -> Arr.reshape z [| -1; n_hand |]
      in
      let open Likelihoods.Pair_P in
      { u = None; z = None; o = { fst = AD.pack_arr x; snd = AD.pack_arr y } })
  |> C.broadcast


let test_data =
  Array.init n_test_trials ~f:(fun i ->
      let x =
        Arr.get_slice [ [ i ] ] test_spikes_in
        |> fun z -> Arr.reshape z [| -1; n_train_neurons |]
      in
      { u = None; z = None; o = AD.pack_arr x })
  |> C.broadcast


let all_trials = Array.concat [ train_data; val_data ]
let idces = Array.init (n_trials + n_test_trials) ~f:(fun i -> i)
let _ = C.root_perform (fun () -> Misc.save_bin ~out:(in_dir "indices") idces)

let train_data_full =
  Array.map (Array.sub idces ~pos:0 ~len:n_trials) ~f:(fun i -> all_trials.(i))


let val_data_full =
  Array.map (Array.sub idces ~pos:n_trials ~len:n_test_trials) ~f:(fun i ->
      all_trials.(i))


let train_data =
  Array.map train_data_full ~f:(fun d ->
      let o = d.o.fst in
      { u = None; z = None; o })


let val_data =
  Array.map val_data_full ~f:(fun d ->
      let o = d.o.fst in
      { u = None; z = None; o })


let saving_some =
  C.root_perform (fun () ->
      Array.iteri (Array.sub ~pos:0 ~len:10 train_data_full) ~f:(fun i trial ->
          let dat = trial.o.snd in
          Mat.save_txt
            ~out:(in_dir (Printf.sprintf "train_hand_pos_%i" i))
            (Mat.cumsum ~axis:0 (AD.unpack_arr dat));
          Mat.save_txt
            ~out:(in_dir (Printf.sprintf "train_hand_%i" i))
            (AD.unpack_arr dat)));
  C.root_perform (fun () ->
      Array.iteri (Array.sub ~pos:0 ~len:10 val_data_full) ~f:(fun i trial ->
          let dat = trial.o.snd in
          Mat.save_txt
            ~out:(in_dir (Printf.sprintf "val_hand_pos_%i" i))
            (Mat.cumsum ~axis:0 (AD.unpack_arr dat));
          Mat.save_txt ~out:(in_dir (Printf.sprintf "val_hand_%i" i)) (AD.unpack_arr dat)))


let _ =
  C.root_perform (fun () ->
      Misc.save_bin ~out:(in_dir "train_data_full") train_data_full;
      Misc.save_bin ~out:(in_dir "val_data_full") val_data_full)


let _ = C.print_endline "Data generated and broadcasted and saved."

(* ----------------------------------------- 
   -- Initialise parameters and train
   ----------------------------------------- *)

let (init_prms : Model.P.p) =
  C.broadcast' (fun () ->
      match init_prms_file with
      | Some f -> Misc.read_bin f
      | None ->
        let generative_prms =
          match reuse with
          | Some f ->
            let (prms : Model.P.p) = Misc.read_bin (in_dir f) in
            prms.generative
          | None ->
            let n = setup.n
            and m = setup.m in
            let prior = U.init ~spatial_std:1.0 ~nu:20. ~m learned in
            (* let prior = Priors.Gaussian.init ~spatial_std:1. ~first_bin:1. ~m learned in *)
            let dynamics = D.init ~n ~m learned in
            let likelihood : L.P.p =
              { c =
                  learned
                    (AD.Mat.gaussian ~sigma:Float.(0.1 / sqrt (of_int n)) n_neurons n)
              ; c_mask = None
              ; bias = learned (AD.Mat.ones 1 n_neurons)
              ; gain = learned (AD.Mat.ones 1 n_neurons)
              }
            in
            Generative_P.{ prior; dynamics; likelihood }
        in
        Model.init ~tie:true generative_prms learned)


let save_results ?u_init prefix prms data =
  let file s = prefix ^ "." ^ s in
  let prms = C.broadcast prms in
  C.root_perform (fun () ->
      Misc.save_bin ~out:(file "params.bin") prms;
      Model.P.save_to_files ~prefix ~prms);
  Array.iteri data ~f:(fun i dat_trial ->
      if Int.(i % C.n_nodes = C.rank)
      then (
        try
          let u_init =
            match u_init with
            | None -> None
            | Some u -> u.(i)
          in
          Option.iter u_init ~f:(fun u ->
              Owl.Mat.save_txt ~out:(file (Printf.sprintf "u_init_%i" i)) u);
          let mu = Model.posterior_mean ~conv_threshold:1E-6 ~u_init ~prms dat_trial in
          let us, zs, os = Model.predictions ~n_samples:100 ~prms mu in
          let process label a =
            let a = AD.unpack_arr a in
            Owl.Arr.(mean ~axis:2 a @|| var ~axis:2 a)
            |> (fun z -> Owl.Arr.reshape z [| setup.n_steps; -1 |])
            |> Mat.save_txt ~out:(file (Printf.sprintf "predicted_%s_%i" label i))
          in
          process "u" us;
          process "z" zs;
          Array.iter ~f:(fun (label, x) -> process label x) os
        with
        | _ -> Stdio.printf "Trial %i failed with some exception in save_results." i))


(* for testing : if arm velocity is available, compute the R^2 to that. If not just compute the log likelihood of the whole train as well as the LL per spike for held out neurons and 
the LL per spike*)
let compute_val_metrics (prms : Model.P.p) label data =
  let _ = C.print "start compute test" in
  let prms = C.broadcast prms in
  let in_dir' s = in_dir Printf.(sprintf "%s_%s" label s) in
  let masked_likelihood ~prms =
    let open Likelihoods.Poisson_P in
    let c = Owl_parameters.extract prms.c in
    let bias = Owl_parameters.extract prms.bias in
    let gain = Owl_parameters.extract prms.gain in
    Likelihoods.Poisson_P.
      { c = pinned (AD.Maths.get_slice [ [ 0; n_train_neurons - 1 ] ] c)
      ; bias = pinned (AD.Maths.get_slice [ []; [ 0; n_train_neurons - 1 ] ] bias)
      ; c_mask = None
      ; gain = pinned (AD.Maths.get_slice [ []; [ 0; n_train_neurons - 1 ] ] gain)
      }
  in
  let masked_prms =
    C.broadcast' (fun () ->
        VAE_P.
          { generative =
              { prior = prms.generative.prior
              ; dynamics = prms.generative.dynamics
              ; likelihood = masked_likelihood ~prms:prms.generative.likelihood
              }
          ; recognition =
              prms.recognition
              |> Accessor.map Recognition_P.A.generative ~f:(fun _ -> None)
          })
  in
  let data_held d =
    let open Likelihoods.Pair_P in
    let o = d.o in
    let neural = o.fst in
    let new_o =
      AD.Maths.concatenate
        ~axis:1
        [| AD.Maths.get_slice [ []; [ 0; n_train_neurons - 1 ] ] neural |]
    in
    { z = None; u = None; o = new_o }
  in
  Array.foldi data ~init:[] ~f:(fun i accu dat_trial ->
      if Int.(i % C.n_nodes = C.rank)
      then (
        try
          let open Likelihoods.Poisson_P in
          let data_withheld = data_held dat_trial in
          let mu =
            Model.posterior_mean
              ~conv_threshold:1E-6
              ~u_init:None
              ~prms:masked_prms
              data_withheld
          in
          let _, zs, os = Model.predictions ~n_samples:1000 ~prms mu in
          let z_mean =
            Owl.Arr.(mean ~axis:2 (AD.unpack_arr zs))
            |> fun z ->
            z |> (fun z -> Owl.Arr.reshape z [| (Arr.shape z).(0); -1 |]) |> AD.pack_arr
          in
          let pred_lambdas =
            let c = Owl_parameters.extract prms.generative.likelihood.c in
            let bias = Owl_parameters.extract prms.generative.likelihood.bias in
            let gain = Owl_parameters.extract prms.generative.likelihood.gain in
            AD.Maths.(F dt * gain * exp ((z_mean *@ transpose c) + bias))
          in
          let pred_lambda =
            AD.Maths.get_slice [ []; [ 0 - n_test_neurons; -1 ] ] pred_lambdas
          in
          (*we get the LL score : this is summed across neurons, and normalized by substracting the LL of a flat mean rate and dividing by total number of spikes of a neurons *)
          let true_spikes =
            Arr.get_slice
              [ []; [ 0 - n_test_neurons; -1 ] ]
              (AD.unpack_arr dat_trial.o.fst)
            |> fun z -> Arr.reshape z [| -1; n_test_neurons |] |> AD.pack_arr
          in
          let flat_rate =
            Mat.mean ~axis:0 (AD.unpack_arr true_spikes)
            |> fun z ->
            Mat.(
              (z * ones (AD.Mat.row_num true_spikes) (AD.Mat.col_num true_spikes))
              +$ 0.00001)
            |> AD.pack_arr
          in
          (pred_lambda, flat_rate, true_spikes) :: accu
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
      try
        let v = v |> Array.to_list |> List.concat |> Array.of_list in
        let _ = Stdio.printf "post v %!" in
        let pred_lambda =
          Array.map v ~f:(fun (a, _, _) -> a) |> AD.Maths.concatenate ~axis:0
        in
        let _ = Stdio.printf "pred_lambda %!" in
        (* let flat_rate =
        Array.map v ~f:(fun (_, a, _) -> a) |> AD.Maths.concatenate ~axis:0
      in *)
        let true_spikes =
          Array.map v ~f:(fun (_, _, a) -> a) |> AD.Maths.concatenate ~axis:0
        in
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
          ~out:(in_dir' "LL_neuron")
          (Mat.of_array [| AD.unpack_flt normalized_ll |] 1 (-1))
      with
      | _ -> ())


let test_preds (prms : Model.P.p) label data =
  let prms = C.broadcast prms in
  try
    let in_dir' s = in_dir Printf.(sprintf "%s_%s" label s) in
    let masked_likelihood ~prms =
      let open Likelihoods.Poisson_P in
      let c = Owl_parameters.extract prms.c in
      let bias = Owl_parameters.extract prms.bias in
      let gain = Owl_parameters.extract prms.gain in
      Likelihoods.Poisson_P.
        { c = pinned (AD.Maths.get_slice [ [ 0; n_train_neurons - 1 ] ] c)
        ; bias = pinned (AD.Maths.get_slice [ []; [ 0; n_train_neurons - 1 ] ] bias)
        ; c_mask = None
        ; gain = pinned (AD.Maths.get_slice [ []; [ 0; n_train_neurons - 1 ] ] gain)
        }
    in
    let masked_prms =
      C.broadcast' (fun () ->
          VAE_P.
            { generative =
                { prior = prms.generative.prior
                ; dynamics = prms.generative.dynamics
                ; likelihood = masked_likelihood ~prms:prms.generative.likelihood
                }
            ; recognition =
                prms.recognition
                |> Accessor.map Recognition_P.A.generative ~f:(fun _ -> None)
            })
    in
    Array.foldi data ~init:[] ~f:(fun i accu dat_trial ->
        if Int.(i % C.n_nodes = C.rank)
        then (
          try
            let open Likelihoods.Poisson_P in
            let data_withheld = dat_trial in
            let mu =
              Model.posterior_mean
                ~conv_threshold:1E-6
                ~u_init:None
                ~prms:masked_prms
                data_withheld
            in
            let _, zs, os = Model.predictions ~n_samples:1000 ~prms mu in
            let z_mean =
              Owl.Arr.(mean ~axis:2 (AD.unpack_arr zs))
              |> fun z ->
              z |> (fun z -> Owl.Arr.reshape z [| (Arr.shape z).(0); -1 |]) |> AD.pack_arr
            in
            let pred_lambdas =
              let c = Owl_parameters.extract prms.generative.likelihood.c in
              let bias = Owl_parameters.extract prms.generative.likelihood.bias in
              let gain = Owl_parameters.extract prms.generative.likelihood.gain in
              AD.Maths.(F dt * gain * exp ((z_mean *@ transpose c) + bias))
            in
            AD.Arr.reshape pred_lambdas [| 1; -1; n_neurons |] :: accu
          with
          | e ->
            Stdio.printf
              "Trial %i failed with some exception in compute likelihood val : %s"
              i
              (Exn.to_string e);
            accu)
        else accu)
    |> C.gather
    |> Array.to_list
    |> List.concat
    |> Array.of_list
    |> fun x ->
    AD.Maths.concatenate ~axis:0 x
    |> AD.unpack_arr
    |> fun z ->
    Mat.save_txt
      ~out:(in_dir "test_pred_0")
      (Arr.reshape (Arr.get_slice [ [ 0 ] ] z) [| -1; n_neurons |]);
    z |> Arr.save_npy ~out:(in_dir "test_predictions")
  with
  | _ -> ()


let save_reg_r2 ~lambda prefix prms data val_data =
  let open Likelihoods.Pair_P in
  let file s = prefix ^ "." ^ s in
  C.root_perform (fun () ->
      Misc.save_bin ~out:(file "params.bin") prms;
      Model.P.save_to_files ~prefix ~prms);
  Array.foldi data ~init:[] ~f:(fun i accu dat_trial ->
      if Int.(i % C.n_nodes = C.rank)
      then (
        try
          let open Likelihoods.Pair_P in
          let (data_neural : L.data data) = { u = None; z = None; o = dat_trial.o.fst } in
          let data_hand = { u = None; z = None; o = dat_trial.o.snd } in
          let mu =
            Model.posterior_mean
              ~saving_iter:(file (Printf.sprintf "iter_%i" i))
              ~conv_threshold:1E-4
              ~u_init:None
              ~prms
              data_neural
          in
          let us, zs, os = Model.predictions ~n_samples:100 ~prms mu in
          let process label a =
            let a = AD.unpack_arr a in
            Owl.Arr.(mean ~axis:2 a @|| var ~axis:2 a)
            |> (fun z -> Owl.Arr.reshape z [| setup.n_steps; -1 |])
            |> Mat.save_txt ~out:(file (Printf.sprintf "predicted_%s_%i" label i))
          in
          (* process "u" us;
          process "z" zs;
          Array.iter ~f:(fun (label, x) -> process label x) os; *)
          let mean_z =
            let z = AD.unpack_arr zs in
            Owl.Arr.(mean ~axis:2 z)
            |> fun z -> Owl.Arr.reshape z [| setup.n_steps; -1 |] |> AD.pack_arr
          in
          let mean_z =
            let c = Owl_parameters.extract prms.generative.likelihood.c in
            let bias = Owl_parameters.extract prms.generative.likelihood.bias in
            let gain = Owl_parameters.extract prms.generative.likelihood.gain in
            AD.Maths.(gain * exp ((mean_z *@ transpose c) + bias)) |> AD.unpack_arr
          in
          (*get slice to have just the movement phase, so *)
          let hand =
            Mat.get_slice [ [ optimal_bin_lag; -1 ] ] (AD.unpack_arr data_hand.o)
          in
          let mean_z = Mat.get_slice [ [ 0; -1 - optimal_bin_lag ] ] mean_z in
          (hand, mean_z) :: accu
        with
        | e ->
          Stdio.printf
            "Trial %i failed with some exception %s in reg."
            i
            (Exn.to_string e);
          accu)
      else accu)
  |> C.gather
  |> fun v ->
  try
    let latent_neural, true_hand =
      C.broadcast' (fun () ->
          (* v is an array of lists *)
          let v = v |> Array.to_list |> List.concat |> Array.of_list in
          let true_hand = Array.map v ~f:fst |> Mat.concatenate ~axis:0 in
          let latent_neural = Array.map v ~f:snd |> Mat.concatenate ~axis:0 in
          Mat.save_txt ~out:(in_dir "pred_neural") latent_neural;
          Mat.save_txt ~out:(in_dir "true_train_hand") true_hand;
          latent_neural, true_hand)
    in
    let _, c =
      C.broadcast' (fun () ->
          let x = latent_neural in
          let y = true_hand in
          compute_R2 ~lambda x y)
    in
    (* let train_r2, _ =
            compute_R2
              (Mat.load_txt (in_dir "pred_neural"))
              (unsquash (Mat.load_txt (in_dir "true_train_hand")))
          in *)
    Array.foldi val_data ~init:[] ~f:(fun i accu dat_trial ->
        if Int.(i % C.n_nodes = C.rank)
        then
          let open Likelihoods.Pair_P in
          let data_neural = { u = None; z = None; o = dat_trial.o.fst } in
          let data_hand = { u = None; z = None; o = dat_trial.o.snd } in
          let mu =
            Model.posterior_mean ~conv_threshold:1E-4 ~u_init:None ~prms data_neural
          in
          let us, zs, os = Model.predictions ~n_samples:100 ~prms mu in
          (* let process label a =
            let a = AD.unpack_arr a in
            Owl.Arr.(mean ~axis:2 a @|| var ~axis:2 a)
            |> (fun z -> Owl.Arr.reshape z [| setup.n_steps; -1 |])
            |> Mat.save_txt ~out:(file (Printf.sprintf "predicted_%s_%i" label i))
          in
          process "u" us; *)
          (* process "z" zs;
          Array.iter ~f:(fun (label, x) -> process label x) os; *)
          let mean_z =
            let z = AD.unpack_arr zs in
            Owl.Arr.(mean ~axis:2 z)
            |> fun z -> Owl.Arr.reshape z [| setup.n_steps; -1 |] |> AD.pack_arr
          in
          let mean_z =
            let c = Owl_parameters.extract prms.generative.likelihood.c in
            let bias = Owl_parameters.extract prms.generative.likelihood.bias in
            let gain = Owl_parameters.extract prms.generative.likelihood.gain in
            AD.Maths.(gain * exp ((mean_z *@ transpose c) + bias)) |> AD.unpack_arr
          in
          let hand =
            Mat.get_slice [ [ optimal_bin_lag; -1 ] ] (AD.unpack_arr data_hand.o)
          in
          let mean_z = Mat.get_slice [ [ 0; -1 - optimal_bin_lag ] ] mean_z in
          (hand, mean_z) :: accu
        else accu)
    |> C.gather
    |> fun v ->
    C.root_perform (fun () ->
        (* v is an array of lists *)
        try
          let v = v |> Array.to_list |> List.concat |> Array.of_list in
          let true_hand = Array.map v ~f:fst |> Mat.concatenate ~axis:0 in
          let latent_neural = Array.map v ~f:snd |> Mat.concatenate ~axis:0 in
          let pred_hand = Mat.(latent_neural *@ c) in
          let r2 =
            1. -. (Mat.(mean' (sqr (true_hand - pred_hand))) /. Mat.var' true_hand)
          in
          Mat.(
            save_txt
              ~append:true
              ~out:(file (Printf.sprintf "regressed_R2_%f" lambda))
              (of_array [| r2 |] 1 1))
        with
        | _ -> ())
  with
  | _ -> ()


let reg ~(prms : Model.P.p) =
  let z = Float.(1E-5 / of_int Int.(setup.n * setup.n)) in
  let part1 = AD.Maths.(F z * l2norm_sqr' (extract prms.generative.dynamics.wh)) in
  (*let part2 = AD.Maths.(F z * l2norm_sqr' (extract prms.generative.dynamics.wf)) in *)
  let part3 = AD.Maths.(F z * l2norm_sqr' (extract prms.generative.dynamics.uh)) in
  let part4 = AD.Maths.(F z * l2norm_sqr' (extract prms.generative.dynamics.uf)) in
  AD.Maths.(part1 + part3 + part4)


let _ =
  let _ = Stdio.printf "predsss %!" in
  test_preds init_prms "test" [| test_data.(0) |]


let final_prms =
  let in_each_iteration ~u_init ~prms k =
    if Int.(k % 10 = 0) then Model.P.save_to_files ~prefix:(in_dir "final") ~prms;
    if Int.(k % 200 = 0)
    then (
      save_results (in_dir "final") prms train_data;
      save_results (in_dir "val") prms val_data;
      compute_val_metrics prms "train" train_data_full;
      compute_val_metrics prms "val" val_data_full)
    (* test_preds prms "test" test_data; *)
    (* Array.iter [| 1E-2; 1E-3; 1E-4; 1E-5; 1E-6; 1E-7 |] ~f:(fun lambda ->
          save_reg_r2 ~lambda (in_dir "reg_R2") prms train_data_full val_data_full) *)
    (*reg seems to cause specifically issues*)
  in
  Model.train
    ~n_samples:(fun _ -> 100)
    ~mini_batch:228
    ~max_iter:Cmdargs.(get_int "-max_iter" |> default 30000)
    ~conv_threshold:1E-6
    ~save_progress_to:(1, 1000, in_dir "progress")
    ~in_each_iteration
    ~recycle_u:false
    ~reg
    ~eta:
      (`of_iter
        (fun k ->
          if k < Int.(of_float (0.01 *. Float.of_int prev_iter))
          then 0.
          else Float.(eta / (1. + sqrt (of_int Int.(k + prev_iter) / of_int n_decay)))))
    ~init_prms
    train_data


let _ = save_results (in_dir "final") final_prms train_data
