open Base
open Owl_parameters

module Pair_P = struct
  type ('a, 'b) prm_ =
    { fst : 'a
    ; snd : 'b
    }
  [@@deriving accessors ~submodule:A]

  module Make (X1 : sig
    include Owl_parameters.T

    val label : string
  end) (X2 : sig
    include Owl_parameters.T

    val label : string
  end) =
  struct
    type 'a prm = ('a X1.prm, 'a X2.prm) prm_

    let map ~f x = { fst = X1.map ~f x.fst; snd = X2.map ~f x.snd }

    let fold ?prefix ~init ~f x =
      let w = with_prefix ?prefix in
      let init = X1.fold ~prefix:(w X1.label) ~init ~f x.fst in
      X2.fold ~prefix:(w X2.label) ~init ~f x.snd
  end
end
