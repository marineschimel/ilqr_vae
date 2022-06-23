include module type of Rnn_typ

module RNN : sig
  include RNN_T with type 'a P.prm = 'a RNN_P.prm

  val init : n:int -> n_input:int -> Owl_parameters.setter -> Owl_parameters.t P.prm
end
