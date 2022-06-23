open Base
open Owl_parameters

module type RNN_T = sig
  module P : Owl_parameters.T
  open P

  val run : ?backward:bool -> prms:p -> h0:AD.t -> input:AD.t -> AD.t
end

module RNN_P = struct
  type 'a prm =
    { uf : 'a
    ; wh : 'a
    ; uh : 'a
    ; bh : 'a
    }
  [@@deriving accessors ~submodule:A]

  let map ~f x = { uf = f x.uf; bh = f x.bh; uh = f x.uh; wh = f x.wh }

  let fold ?prefix ~init ~f x =
    let init = f init (x.uh, with_prefix ?prefix "uh") in
    let init = f init (x.uf, with_prefix ?prefix "uf") in
    let init = f init (x.bh, with_prefix ?prefix "bh") in
    f init (x.wh, with_prefix ?prefix "wh")
end