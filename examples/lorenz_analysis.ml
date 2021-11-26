open Base
open Owl
open Ilqr_vae
open Variational
open Lorenz_common

let dir = Cmdargs.(get_string "-d" |> force ~usage:"-d [dir]")
let in_dir = Printf.sprintf "%s/%s" dir
let setup = C.broadcast' (fun () -> Misc.read_bin (in_dir "../setup.bin"))
let _ = C.print_endline (Printf.sprintf "m = %i" setup.m)

module Model = Make_model (struct
  let setup = setup
end)

(* ----------------------------------------- 
   -- Retrieve trained parameters
   ----------------------------------------- *)

let (prms_final : Model.P.p) =
  C.broadcast' (fun () ->
      try Misc.read_bin (in_dir "../final.params.bin.freeze") with
      | _ -> Misc.read_bin (in_dir "../final.params.bin"))


let (prms_init : Model.P.p) =
  C.broadcast' (fun () -> Misc.read_bin (in_dir "../init.params.bin"))


(* ----------------------------------------- 
   -- Extrapolation test:
   -- get posterior mean, zero out the last half
   -- and run the dynamics
   ----------------------------------------- *)

let do_for prefix prms =
  assert (Int.(setup.n_steps = 50 + 50));
  Misc.read_bin (in_dir "../test_data.bin")
  |> Array.iteri ~f:(fun i data ->
         if Int.(i % C.n_nodes = C.rank)
         then (
           Stdio.printf "\rTesting [%04i]%!" i;
           let u =
             let mu_u = Model.posterior_mean ~u_init:None ~prms data |> AD.unpack_arr in
             Mat.save_txt
               ~out:(in_dir Printf.(sprintf "%s_extrapolation_posterior_u_%i" prefix i))
               mu_u;
             let mask = Mat.(ones 50 1 @= zeros 50 1) in
             AD.pack_arr Mat.(mask * mu_u)
           in
           let module D = Dynamics.Integrate (D) in
           let z =
             D.integrate ~prms:prms.generative.dynamics ~n:setup.n ~u:(AD.expand0 u)
           in
           let o = L.pre_sample ~prms:prms.generative.likelihood ~z:(AD.squeeze0 z) in
           Mat.save_txt
             ~out:(in_dir Printf.(sprintf "%s_extrapolation_%i" prefix i))
             (AD.unpack_arr o)))


let _ =
  do_for "init" prms_init;
  do_for "final" prms_final


(* ----------------------------------------- 
   -- Long autonomous trajectory test
   ----------------------------------------- *)

let _ =
  let module D = Dynamics.Integrate (D) in
  let u = AD.pack_arr Mat.(gaussian ~sigma:5. 1 setup.m @= zeros 9999 setup.m) in
  let z = D.integrate ~prms:prms_final.generative.dynamics ~n:setup.n ~u:(AD.expand0 u) in
  let o = L.pre_sample ~prms:prms_final.generative.likelihood ~z:(AD.squeeze0 z) in
  Mat.save_txt
    ~out:(in_dir Printf.(sprintf "long_autonomous_%i" C.rank))
    (AD.unpack_arr o)


(* ----------------------------------------- 
   -- R2_k à la Hernandez et al
   ----------------------------------------- *)

let do_for prefix prms =
  let module D = Dynamics.Integrate (D) in
  Misc.read_bin (in_dir "../test_data.bin")
  |> Array.mapi ~f:(fun i data ->
         if Int.(i % C.n_nodes = C.rank)
         then (
           let true_o = Option.value_exn data.z |> AD.unpack_arr in
           let mean_o = Mat.mean ~axis:0 true_o in
           (* get the posterior *)
           let u = Model.posterior_mean ~u_init:None ~prms data |> AD.unpack_arr in
           (* for each t from 1 to T-2, integrate after zeroing out all inputs after t *)
           let pred_os =
             Array.init (setup.n_steps - 2) ~f:(fun t ->
                 let uz =
                   let uz = Mat.copy u in
                   Mat.set_slice
                     [ [ t + 1; -1 ] ]
                     uz
                     Arr.(zeros (shape (Mat.get_slice [ [ Int.(t + 1); -1 ] ] uz)));
                   AD.pack_arr uz
                 in
                 let z =
                   D.integrate
                     ~prms:prms.generative.dynamics
                     ~n:setup.n
                     ~u:(AD.expand0 uz)
                 in
                 let o =
                   L.pre_sample ~prms:prms.generative.likelihood ~z:(AD.squeeze0 z)
                   |> AD.unpack_arr
                 in
                 Mat.save_txt
                   ~out:(in_dir Printf.(sprintf "predicted_o_%i_after_%i" i t))
                   o;
                 o)
           in
           let r2k =
             Mat.init 1 50 (fun k ->
                 let tmp =
                   Array.init
                     (setup.n_steps - k - 2)
                     ~f:(fun t ->
                       let true_o = Mat.get_slice [ [ t + k ] ] true_o in
                       let pred_o = Mat.get_slice [ [ t + k ] ] pred_os.(t) in
                       let num = Mat.(l2norm_sqr' (true_o - pred_o)) in
                       let denom = Mat.(l2norm_sqr' (true_o - mean_o)) in
                       num, denom)
                 in
                 Float.(
                   1.
                   - (Stats.sum (Array.map ~f:fst tmp) / Stats.sum (Array.map ~f:snd tmp))))
           in
           Some r2k)
         else None)
  |> C.gatheroption
  |> fun r2ks ->
  if C.first
  then
    Mat.concatenate ~axis:0 r2ks
    |> Mat.mean ~axis:0
    |> Mat.transpose
    |> Mat.save_txt ~out:(in_dir Printf.(sprintf "%s_r2k" prefix))


let _ =
  do_for "init" prms_init;
  do_for "final" prms_final
