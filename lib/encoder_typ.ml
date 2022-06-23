open Owl_parameters
open Rnn
open Rnn_typ

module type Encoder_T = sig
  module P : Owl_parameters.T
  open P

  val encode : prms:p -> input:AD.t -> AD.t * AD.t
end

module Init_P = struct
  type ('a, 'b) prm_ =
    { rnn : 'b
    ; e_f : 'a
    ; e_i : 'a
    ; w_mean : 'a
    ; w_std : 'a
    }
  [@@deriving accessors ~submodule:A]

  module Make = struct
    type 'a prm = ('a, 'a RNN_P.prm) prm_

    let map ~f x =
      { rnn = RNN.P.map ~f x.rnn
      ; e_i = f x.e_i
      ; e_f = f x.e_f
      ; w_mean = f x.w_mean
      ; w_std = f x.w_std
      }


    let fold ?prefix ~init ~f x =
      let w = with_prefix ?prefix in
      let init = RNN.P.fold ~prefix:(w "forward") ~init ~f x.rnn in
      let init = f init (x.e_i, with_prefix ?prefix "e_i") in
      let init = f init (x.e_f, with_prefix ?prefix "e_f") in
      let init = f init (x.w_mean, with_prefix ?prefix "w_mean") in
      f init (x.w_std, with_prefix ?prefix "w_cov")
  end
end

module Controller_P = struct
  type ('a, 'b) prm_ =
    { w_c0 : 'a
    ; e_i : 'a
    ; e_f : 'a
    ; rnn : 'b
    ; controller : 'b
    ; w_mean : 'a
    ; w_std : 'a
    }
  [@@deriving accessors ~submodule:A]

  module Make = struct
    type 'a prm = ('a, 'a RNN_P.prm) prm_

    let map ~f x =
      { w_c0 = f x.w_c0
      ; controller = RNN.P.map ~f x.controller
      ; rnn = RNN.P.map ~f x.rnn
      ; e_i = f x.e_i
      ; e_f = f x.e_f
      ; w_mean = f x.w_mean
      ; w_std = f x.w_std
      }


    let fold ?prefix ~init ~f x =
      let w = with_prefix ?prefix in
      let init = RNN.P.fold ~prefix:(w "rnn") ~init ~f x.rnn in
      let init = f init (x.e_i, with_prefix ?prefix "e_i") in
      let init = f init (x.e_f, with_prefix ?prefix "e_f") in
      let init = RNN.P.fold ~prefix:(w "controller") ~init ~f x.controller in
      let init = f init (x.w_c0, with_prefix ?prefix "w_c0") in
      let init = f init (x.w_mean, with_prefix ?prefix "w_mean") in
      f init (x.w_std, with_prefix ?prefix "w_std")
  end
end
