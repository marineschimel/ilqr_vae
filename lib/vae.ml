open Base
open Owl
include Vae_typ

(* -------------------------------------
   -- VAE
   ------------------------------------- *)

module Make (G : Generative.T) (R : Recognition.T with module G = G) = struct
  module P = Owl_parameters.Make (P.Make (G.P) (R.P))
  open Vae_typ.P
  module G = G
  module R = R

  let init g r = { generative = g; recognition = r }

  let posterior_predictive_sample ?(id = 0) ?(pre = true) ~(prms : P.p) data =
    let u_mean = R.posterior_mean ~gen_prms:prms.generative prms.recognition data in
    let u_cov_sample =
      R.posterior_cov_sample ~gen_prms:prms.generative prms.recognition
    in
    let sample = G.sample ~prms:prms.generative in
    ( u_mean
    , fun n_samples ->
        let u = AD.Maths.(u_mean + u_cov_sample ~n_samples) in
        Array.init n_samples ~f:(fun k ->
            let u = AD.Maths.(reshape (get_slice [ [ k ] ] u) [| -1; G.m |]) in
            let data = sample ~id ~pre (`some u) in
            let remove_n_beg = AD.Maths.get_slice [ [ G.n_beg - 1; -1 ] ] in
            let u = remove_n_beg (Data.u data) in
            let z = remove_n_beg (Data.z data) in
            Data.fill ~u ~z data) )


  let elbo ~prms ~n_posterior_samples =
    let h = R.entropy ~gen_prms:prms.generative prms.recognition in
    let log_joint =
      let mu = R.posterior_mean ~gen_prms:prms.generative prms.recognition in
      let sample_cov =
        R.posterior_cov_sample ~gen_prms:prms.generative prms.recognition
      in
      let log_prior = G.log_prior ~prms:prms.generative in
      let log_likelihood = G.log_likelihood ~prms:prms.generative in
      let integrate = G.integrate ~prms:prms.generative in
      let norm_const = Float.(1. / of_int n_posterior_samples) in
      fun data ->
        let samples =
          AD.Maths.(AD.expand_to_3d (mu data) + sample_cov ~n_samples:n_posterior_samples)
        in
        let z = integrate ~u:samples in
        let z = AD.Maths.get_slice [ []; [ G.n_beg - 1; -1 ]; [ 0; G.n - 1 ] ] z in
        let data = Data.fill data ~u:samples ~z in
        AD.Maths.((log_prior samples + log_likelihood data) * AD.F norm_const)
    in
    fun data -> AD.Maths.(h + log_joint data)


  (* NOTE: each node has its own local data -- this must be distributed upstream using e.g. C.scatter *)
  let train
      ?(n_posterior_samples = fun _ -> 1)
      ?mini_batch
      ?max_iter
      ?save_progress_to
      ?in_each_iteration
      ?learning_rate
      ?regularizer
      ~init_prms
      data
    =
    let n_trials = Array.length data in
    let data_ids = Array.init n_trials ~f:Fn.id in
    let module Packer = Owl_parameters.Packer () in
    let handle = P.pack (module Packer) init_prms in
    let theta, lbound, ubound = Packer.finalize () in
    let theta = AD.unpack_arr theta in
    let adam_loss k theta gradient =
      Stdlib.Gc.full_major ();
      let theta = C.broadcast theta in
      let data_batch =
        match mini_batch with
        | None -> data
        | Some size ->
          (* each node samples a local minibatch from their own set *)
          Array.permute data_ids;
          let ids = Array.sub data_ids ~pos:0 ~len:size in
          Array.map ids ~f:(Array.get data)
      in
      let n_posterior_samples = n_posterior_samples k in
      (* do a quick pass through the data to work out the problem size;
      we need this to normalise the loss independently of the regularizer *)
      let total_size =
        Array.fold data_batch ~init:0 ~f:(fun accu datai ->
            Int.(accu + G.L.numel (Data.o datai)))
      in
      let loss, g =
        Array.foldi
          data_batch
          ~init:(0., Arr.(zeros (shape theta)))
          ~f:(fun i (accu_loss, accu_g) datai ->
            (* try *)
            let open AD in
            let theta = make_reverse (Arr (Owl.Mat.copy theta)) (AD.tag ()) in
            let prms = P.unpack handle theta in
            let elbo = elbo ~prms ~n_posterior_samples datai in
            let loss = AD.Maths.(neg elbo / F Float.(of_int total_size)) in
            (* optionally add regularizer *)
            let loss =
              Option.value_map regularizer ~default:loss ~f:(fun r ->
                  AD.Maths.(loss + r ~prms))
            in
            reverse_prop (F 1.) loss;
            accu_loss +. unpack_flt loss, Owl.Mat.(accu_g + unpack_arr (adjval theta))
            (* with
            | _ ->
              Stdio.printf "Trial %i on node %i failed with some exception." i C.rank;
              accu_loss, accu_g *))
      in
      let loss = Mpi.reduce_float loss Mpi.Float_sum 0 Mpi.comm_world in
      Mpi.reduce_bigarray g gradient Mpi.Sum 0 Mpi.comm_world;
      Mat.div_scalar_ gradient Float.(of_int C.n_nodes);
      Float.(loss / of_int C.n_nodes)
    in
    let stop iter current_loss =
      (* optionally do something based on the parameters in each iteration *)
      Option.iter in_each_iteration ~f:(fun do_this ->
          let prms = P.unpack handle (AD.pack_arr theta) in
          do_this ~prms iter);
      C.root_perform (fun () ->
          Stdio.printf "\r[%05i]%!" iter;
          Option.iter save_progress_to ~f:(fun (loss_every, prms_every, prefix) ->
              let kk = Int.((iter - 1) / loss_every) in
              if Int.((iter - 1) % prms_every) = 0
              then (
                let prefix = Printf.sprintf "%s_%i" prefix kk in
                let prms = P.unpack handle (AD.pack_arr theta) in
                Misc.save_bin ~out:(prefix ^ ".params.bin") prms;
                P.save_to_files ~prefix ~prms);
              if Int.((iter - 1) % loss_every) = 0
              then (
                Stdio.printf "\r[%05i] %.4f%!" iter current_loss;
                let z = [| [| Float.of_int kk; current_loss |] |] in
                Mat.(save_txt ~append:true (of_arrays z) ~out:(prefix ^ ".loss")))));
      match max_iter with
      | Some mi -> iter > mi
      | None -> false
    in
    let _ = Adam.min ?eta:learning_rate ?lb:lbound ?ub:ubound ~stop adam_loss theta in
    theta |> AD.pack_arr |> P.unpack handle


  let save_results ~prefix ~prms data =
    let prms = C.broadcast prms in
    let file s = prefix ^ "." ^ s in
    (* save the parameters *)
    C.root_perform (fun () ->
        Misc.save_bin ~out:(file "params.bin") prms;
        P.save_to_files ~prefix ~prms);
    (* save inference results *)
    let results =
      Array.map data ~f:(fun d ->
          let id = Data.id d in
          let u_mean, f = posterior_predictive_sample ~pre:true ~prms d in
          (* draw 100 samples from the predictive distribution *)
          let pred = f 100 in
          (* estimate mean and variances of u, z, and o *)
          let mean_and_var g =
            let tmp =
              Array.map pred ~f:(fun d -> AD.expand_to_3d (g d))
              |> AD.Maths.concatenate ~axis:0
              |> AD.unpack_arr
            in
            ( AD.pack_arr (Arr.mean ~keep_dims:false ~axis:0 tmp)
            , AD.pack_arr (Arr.var ~keep_dims:false ~axis:0 tmp) )
          in
          let _, u_var = mean_and_var Data.u in
          let z_mean, z_var = mean_and_var Data.z in
          let o_mean, o_var = G.L.stats (Array.map pred ~f:Data.o) in
          let mean = Data.pack ~id o_mean |> Data.fill ~u:u_mean ~z:z_mean in
          let var = Data.pack ~id o_var |> Data.fill ~u:u_var ~z:z_var in
          mean, var)
    in
    Data.save ~prefix:(file "mean") G.L.save_output (Array.map results ~f:fst);
    Data.save ~prefix:(file "var") G.L.save_output (Array.map results ~f:snd)
end
