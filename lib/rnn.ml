open Owl
open Base
include Rnn_typ

module RNN = struct
  module P = Owl_parameters.Make (RNN_P)
  open RNN_P

  let sigma x = AD.Maths.sigmoid x
  let phi x = AD.Maths.relu x
  let requires_linesearch = true

  let init ~n ~n_input (set : Owl_parameters.setter) =
    let sigma = Float.(0.01 / sqrt (of_int n)) in
    { wh = set (AD.Mat.gaussian ~sigma n_input n)
    ; bh = set (AD.Mat.zeros 1 n)
    ; uh = set (AD.Mat.gaussian ~sigma n n)
    ; uf = set (AD.Mat.gaussian ~sigma:0. n n)
    }


  let dyn ~theta =
    let wh = Owl_parameters.extract theta.wh in
    let bh = Owl_parameters.extract theta.bh in
    let uh = Owl_parameters.extract theta.uh in
    let uf = Owl_parameters.extract theta.uf in
    let n = AD.Mat.col_num bh in
    fun ~k ~h ~input ->
      let x = input in
      let f = sigma AD.Maths.(h *@ uf) in
      let h_hat =
        let hf = AD.Maths.(h * f) in
        AD.Maths.(phi AD.Maths.(bh + (hf *@ uh)) + (x *@ wh))
      in
      AD.Maths.(((F 1. - f) * h) + (f * h_hat))


  let run ?(backward = false) ~prms =
    let dyn_k = dyn ~theta:prms in
    fun ~h0 ~input ->
      (* now u is T x K x M *)
      let n_steps = AD.Mat.row_num input in
      let inputs =
        AD.Maths.split ~axis:0 (Array.init n_steps ~f:(fun _ -> 1)) input |> Array.to_list
      in
      let inputs = if backward then List.rev inputs else inputs in
      let rec dyn k h hs us =
        match us with
        | [] -> List.rev hs
        | input :: input_nexts ->
          let new_h = dyn_k ~k ~h ~input in
          dyn (k + 1) new_h (new_h :: hs) input_nexts
      in
      dyn 0 h0 [] inputs |> Array.of_list |> AD.Maths.concatenate ~axis:0
end