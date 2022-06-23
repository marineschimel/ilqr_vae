open Rnn
include module type of Encoder_typ
include module type of Recognition_typ

module Init : sig
  (* module I : module type of Owl_parameters.Make (Init_P.Make (R.P))
  open I *)
  include Encoder_T with type 'a P.prm = ('a, 'a RNN_P.prm) Init_P.prm_

  val init : n:int -> n_input:int -> n_output:int -> Owl_parameters.setter -> P.p
end

module Controller : sig
  include Encoder_T with type 'a P.prm = ('a, 'a RNN_P.prm) Controller_P.prm_

  val init : n:int -> n_input:int -> n_output:int -> Owl_parameters.setter -> P.p
end
