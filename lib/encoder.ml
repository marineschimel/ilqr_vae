open Base
open Owl
include Recognition_typ
include Encoder_typ
open Rnn

module Init = struct
  include Init_P
  module I = Init_P.Make
  module P = Owl_parameters.Make (I)

  (* n is the size of the state of the RNN, n_output of the output we want for the encoder, and n_input of the input to the encoder*)
  let init ~n ~n_input ~n_output (set : Owl_parameters.setter) =
    let size_w = 2 * n in
    { rnn = RNN.init ~n ~n_input set
    ; e_f = set (AD.Mat.gaussian ~sigma:0.001 1 n)
    ; e_i = set (AD.Mat.gaussian ~sigma:0.001 1 n)
    ; w_mean = set (AD.Mat.gaussian ~sigma:0.001 size_w n_output)
    ; w_std = set (AD.Mat.gaussian ~sigma:0.001 size_w n_output)
    }


  let encode ~prms ~input =
    let e_i = Owl_parameters.extract prms.e_i in
    let e_f = Owl_parameters.extract prms.e_f in
    let forward = RNN.run ~backward:false ~prms:prms.rnn ~h0:e_i ~input in
    let backward = RNN.run ~backward:true ~prms:prms.rnn ~h0:e_f ~input in
    let unroll =
      (*check that this is what LFADS actually does*)
      AD.Maths.concatenate
        ~axis:1
        [| AD.Maths.get_slice [ [ -1 ] ] forward
         ; AD.Maths.get_slice [ [ -1 ] ] backward
        |]
    in
    let w_mean = Owl_parameters.extract prms.w_mean in
    let w_std = Owl_parameters.extract prms.w_std in
    AD.Maths.(unroll *@ w_mean), AD.Maths.(exp (unroll *@ w_std))
end

module Controller = struct
  include Controller_P
  module C = Controller_P.Make
  module P = Owl_parameters.Make (C)

  (* n is the size of the state of the RNN, n_output of the output we want for the encoder, and n_input of the input to the encoder*)
  let init ~n ~n_input ~n_output (set : Owl_parameters.setter) =
    { rnn = RNN.init ~n ~n_input set
    ; e_f = set (AD.Mat.gaussian ~sigma:0.001 1 n)
    ; e_i = set (AD.Mat.gaussian ~sigma:0.001 1 n)
    ; controller = RNN.init ~n ~n_input:(2 * n) set
    ; w_c0 = set (AD.Mat.gaussian ~sigma:0.001 (2 * n) n)
    ; w_mean = set (AD.Mat.gaussian ~sigma:0.001 n n_output)
    ; w_std = set (AD.Mat.gaussian ~sigma:0.001 n n_output)
    }


  let encode ~prms ~input =
    let e_i = Owl_parameters.extract prms.e_i in
    let e_f = Owl_parameters.extract prms.e_f in
    let forward = RNN.run ~backward:false ~prms:prms.rnn ~h0:e_i ~input in
    let backward = RNN.run ~backward:true ~prms:prms.rnn ~h0:e_f ~input in
    let n_steps = AD.Mat.row_num backward in
    let backward_rev =
      backward
      |> fun z ->
      AD.Maths.split ~axis:0 (Array.init n_steps ~f:(fun _ -> 1)) z
      |> Array.to_list
      |> List.rev
      |> Array.of_list
      |> fun z -> AD.Maths.concatenate ~axis:0 z
    in
    let es = AD.Maths.concatenate ~axis:1 [| forward; backward_rev |] in
    let unroll =
      AD.Maths.concatenate
        ~axis:1
        [| AD.Maths.get_slice [ [ -1 ] ] forward
         ; AD.Maths.get_slice [ [ -1 ] ] backward
        |]
    in
    let wc0 = Owl_parameters.extract prms.w_c0 in
    let c0 = AD.Maths.(unroll *@ wc0) in
    let controls = RNN.run ~backward:false ~prms:prms.controller ~input:es ~h0:c0 in
    let w_mean = Owl_parameters.extract prms.w_mean in
    let w_std = Owl_parameters.extract prms.w_std in
    AD.Maths.(controls *@ w_mean), AD.Maths.(exp (controls *@ w_std))
end
