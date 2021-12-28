open Base
open Owl
open Ilqr_vae
open Owl_parameters
open Lorenz_common

let dir = Cmdargs.(get_string "-d" |> force ~usage:"-d [dir]")
let in_dir = Printf.sprintf "%s/%s" dir
let reuse_data = Cmdargs.check "-reuse_data"

module S = struct
  let n = 20
  let m = 5
  let n_trials = 2
  let n_steps = 100
  let n_output = 3
  let noise_std = 0.1
end

open Make_model (S)

let train_data, test_data =
  let data =
    lazy
      (Lorenz_common.generate_from_long ~n_steps:S.n_steps (2 * S.n_trials)
      |> (fun v -> Arr.reshape v [| -1; 3 |])
      |> (fun v -> Arr.((v - mean ~axis:0 v) / sqrt (var ~axis:0 v)))
      |> (fun v -> Arr.reshape v [| -1; S.n_steps; 3 |])
      |> (fun v ->
           Array.init (2 * S.n_trials) ~f:(fun k -> Arr.(squeeze (get_slice [ [ k ] ] v))))
      |> Array.map ~f:(fun z ->
             let o = AD.pack_arr Arr.(z + gaussian ~sigma:S.noise_std (shape z)) in
             Data.pack o))
  in
  Data.split_and_distribute
    ~reuse:reuse_data
    ~prefix:(in_dir "data")
    ~train:S.n_trials
    data


(* save the data as text files we can plot in gnuplot later *)
let _ =
  Data.save ~prefix:(in_dir "train") L.save_output train_data;
  Data.save ~prefix:(in_dir "test") L.save_output test_data


(* ----------------------------------------- 
   -- Initialise parameters and train
   ----------------------------------------- *)

let init_prms =
  C.broadcast' (fun () ->
      let generative =
        let prior = U.init ~spatial_std:1.0 ~nu:20. learned in
        let dynamics = D.init learned in
        let likelihood = L.init ~sigma2:Float.(square S.noise_std) learned in
        Generative.P.{ prior; dynamics; likelihood }
      in
      let recognition = R.init learned in
      Model.init generative recognition)


let _ = Model.save_results ~prefix:(in_dir "init") ~prms:init_prms train_data

let final_prms =
  let in_each_iteration ~prms k =
    if Int.(k % 200 = 0) then Model.save_results ~prefix:(in_dir "final") ~prms train_data
  in
  Model.train
    ~n_posterior_samples:(fun k -> if k < 200 then 1 else 1)
    ~max_iter:Cmdargs.(get_int "-max_iter" |> default 40000)
    ~save_progress_to:(10, 200, in_dir "progress")
    ~in_each_iteration
    ~learning_rate:(`of_iter (fun k -> Float.(0.004 / (1. + sqrt (of_int k / 1.)))))
    ~regularizer
    ~init_prms
    train_data


let _ =
  Model.save_results ~prefix:(in_dir "final.train") ~prms:final_prms train_data;
  Model.save_results ~prefix:(in_dir "final.test") ~prms:final_prms test_data

