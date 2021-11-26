open Base
open Owl
open Ilqr_vae
open Variational
open Owl_parameters
open Lorenz_common

let dir = Cmdargs.(get_string "-d" |> force ~usage:"-d [dir]")
let in_dir = Printf.sprintf "%s/%s" dir
let reuse = Cmdargs.get_string "-reuse"

let setup =
  C.broadcast' (fun () ->
      match reuse with
      | Some _ -> Misc.read_bin (in_dir "setup.bin")
      | None ->
        let s = { n = 20; m = 5; n_trials = 112; n_steps = 100 } in
        Misc.save_bin ~out:(in_dir "setup.bin") s;
        s)


let n_output = 3
let noise_std = 0.1

module M = Make_model (struct
  let setup = setup
  let n_beg = Some (setup.n / setup.m)
end)

open M

let reg ~(prms : Model.P.p) =
  let z = Float.(1e-5 / of_int Int.(setup.n * setup.n)) in
  let part1 = AD.Maths.(F z * l2norm_sqr' (extract prms.generative.dynamics.uh)) in
  let part2 = AD.Maths.(F z * l2norm_sqr' (extract prms.generative.dynamics.uf)) in
  AD.Maths.(part1 + part2)


(* ----------------------------------------- 
   -- Generate Lorenz data
   ----------------------------------------- *)

let _ = C.print_endline "Data generation..."

(* generate training and test data right away *)
let data =
  C.broadcast' (fun () ->
      match reuse with
      | Some _ -> Misc.read_bin (in_dir "train_data.bin")
      | None ->
        let data =
          Lorenz_common.generate_from_long ~n_steps:setup.n_steps (2 * setup.n_trials)
          |> (fun v -> Arr.reshape v [| -1; 3 |])
          |> (fun v -> Arr.((v - mean ~axis:0 v) / sqrt (var ~axis:0 v)))
          |> (fun v -> Arr.reshape v [| -1; setup.n_steps; 3 |])
          |> (fun v ->
               Array.init (2 * setup.n_trials) ~f:(fun k ->
                   Arr.(squeeze (get_slice [ [ k ] ] v))))
          |> Array.map ~f:(fun z ->
                 let o = Arr.(z + gaussian ~sigma:noise_std (shape z)) in
                 (* here I'm hijacking z to store the Lorenz traj *)
                 { u = None; z = Some (AD.pack_arr z); o = AD.pack_arr o })
        in
        let train_data = Array.sub data ~pos:0 ~len:setup.n_trials in
        let test_data = Array.sub data ~pos:setup.n_trials ~len:setup.n_trials in
        Misc.save_bin ~out:(in_dir "train_data.bin") train_data;
        Misc.save_bin ~out:(in_dir "test_data.bin") test_data;
        let save_data label data =
          Array.iteri data ~f:(fun i data ->
              let file label' = in_dir (Printf.sprintf "%s_data_%s_%i" label label' i) in
              Option.iter data.z ~f:(fun z ->
                  Mat.save_txt ~out:(file "latent") (AD.unpack_arr z));
              L.save_data ~prefix:(file "o") data.o)
        in
        save_data "train" train_data;
        save_data "test" test_data;
        train_data)


let _ = C.print_endline "Data generated and broadcast."

(* ----------------------------------------- 
   -- Initialise parameters and train
   ----------------------------------------- *)

let init_prms =
  C.broadcast' (fun () ->
      let generative_prms =
        match reuse with
        | Some f ->
          let (prms : Model.P.p) = Misc.read_bin (in_dir f) in
          prms.generative
        | None ->
          let n = setup.n
          and m = setup.m in
          (* let prior = U.init ~spatial_std:1.0 ~first_bin:1. ~m learned in *)
          let prior = U.init ~spatial_std:1.0 ~nu:20. ~m learned in
          let dynamics = D.init ~radius:0.05 ~n ~m learned in
          let likelihood = L.init ~sigma2:Float.(square noise_std) ~n ~n_output learned in
          Generative_P.{ prior; dynamics; likelihood }
      in
      Model.init ~tie:true generative_prms learned)


let save_results ?u_init prefix prms data =
  let prms = C.broadcast prms in
  let file s = prefix ^ "." ^ s in
  C.root_perform (fun () ->
      Misc.save_bin ~out:(file "params.bin") prms;
      Model.P.save_to_files ~prefix ~prms);
  Array.iteri data ~f:(fun i dat_trial ->
      if Int.(i % C.n_nodes = C.rank)
      then (
        let u_init =
          match u_init with
          | None -> None
          | Some u -> u.(i)
        in
        Option.iter u_init ~f:(fun u ->
            Owl.Mat.save_txt ~out:(file (Printf.sprintf "u_init_%i" i)) u);
        let mu = Model.posterior_mean ~u_init ~prms dat_trial in
        Owl.Mat.save_txt
          ~out:(file (Printf.sprintf "posterior_u_%i" i))
          (AD.unpack_arr mu);
        let us, zs, os = Model.predictions ~n_samples:100 ~prms mu in
        let process label a =
          let a = AD.unpack_arr a in
          Owl.Arr.(mean ~axis:2 a @|| var ~axis:2 a)
          |> (fun z -> Owl.Arr.reshape z [| setup.n_steps; -1 |])
          |> Mat.save_txt ~out:(file (Printf.sprintf "predicted_%s_%i" label i))
        in
        process "u" us;
        process "z" zs;
        Array.iter ~f:(fun (label, x) -> process label x) os))


let _ = save_results (in_dir "init") init_prms data

let final_prms =
  let in_each_iteration ~u_init ~prms k =
    if Int.(k % 200 = 0) then save_results ~u_init (in_dir "final") prms data
  in
  Model.train
    ~n_samples:(fun k -> if k < 200 then 1 else 1)
    ~max_iter:Cmdargs.(get_int "-max_iter" |> default 40000)
    ~conv_threshold:1E-4
    ~recycle_u:false
    ~save_progress_to:(10, 200, in_dir "progress")
    ~in_each_iteration
    ~eta:
      (`of_iter (fun k -> Float.(0.004 / (1. + sqrt (of_int k / 1.)))))
      (* 0.01 for Gaussian prior *)
    ~init_prms
    ~reg
    data


let _ = save_results (in_dir "final") final_prms data
