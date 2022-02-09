open Base
open Owl_parameters

module type Dims_T = sig
  val n : int
  val m : int
end

module type T = sig
  module P : Owl_parameters.T
  open P
  include Dims_T

  val n_beg : int
  val requires_linesearch : bool
  val dyn : theta:p -> k:int -> x:AD.t -> u:AD.t -> AD.t
  val dyn_x : (theta:p -> k:int -> x:AD.t -> u:AD.t -> AD.t) option
  val dyn_u : (theta:p -> k:int -> x:AD.t -> u:AD.t -> AD.t) option
end

module Linear_P = struct
  type 'a prm =
    { d : 'a
    ; u : 'a
    ; q : 'a
    ; b : 'a option
    }
  [@@deriving accessors ~submodule:A]

  let map ~f x = { d = f x.d; u = f x.u; q = f x.q; b = Option.map ~f x.b }

  let fold ?prefix ~init ~f x =
    let init = f init (x.d, with_prefix ?prefix "d") in
    let init = f init (x.u, with_prefix ?prefix "u") in
    f init (x.q, with_prefix ?prefix "q")
end

module Nonlinear_P = struct
  type 'a prm =
    { a : 'a
    ; bias : 'a
    ; b : 'a option
    }
  [@@deriving accessors ~submodule:A]

  let map ~f x = { a = f x.a; bias = f x.bias; b = Option.map ~f x.b }

  let fold ?prefix ~init ~f x =
    let init = f init (x.a, with_prefix ?prefix "a") in
    let init = f init (x.bias, with_prefix ?prefix "bias") in
    match x.b with
    | None -> init
    | Some b -> f init (b, with_prefix ?prefix "b")
end

module Nonlinear_Init_P = struct
  type 'a prm =
    { a : 'a
    ; bias : 'a
    ; b : 'a option
    }
  [@@deriving accessors ~submodule:A]

  let map ~f x = { a = f x.a; bias = f x.bias; b = Option.map ~f x.b }

  let fold ?prefix ~init ~f x =
    let init = f init (x.a, with_prefix ?prefix "a") in
    let init = f init (x.bias, with_prefix ?prefix "bias") in
    match x.b with
    | None -> init
    | Some b -> f init (b, with_prefix ?prefix "b")
end

module MGU_P = struct
  type 'a prm =
    { wf : 'a
    ; uf : 'a
    ; wh : 'a
    ; uh : 'a
    ; bf : 'a
    ; bh : 'a
    }
  [@@deriving accessors ~submodule:A]

  let map ~f x =
    { wf = f x.wf; uf = f x.uf; bf = f x.bf; bh = f x.bh; uh = f x.uh; wh = f x.wh }


  let fold ?prefix ~init ~f x =
    let init = f init (x.uh, with_prefix ?prefix "uh") in
    let init = f init (x.uf, with_prefix ?prefix "uf") in
    let init = f init (x.bh, with_prefix ?prefix "bh") in
    let init = f init (x.bf, with_prefix ?prefix "bf") in
    let init = f init (x.wh, with_prefix ?prefix "wh") in
    f init (x.wf, with_prefix ?prefix "wf")
end

module MGU2_P = struct
  type 'a prm =
    { uf : 'a
    ; wh : 'a
    ; uh : 'a
    ; bf : 'a
    ; bh : 'a
    }
  [@@deriving accessors ~submodule:A]

  let map ~f x = { uf = f x.uf; bh = f x.bh; bf = f x.bf; uh = f x.uh; wh = f x.wh }

  let fold ?prefix ~init ~f x =
    let init = f init (x.uh, with_prefix ?prefix "uh") in
    let init = f init (x.uf, with_prefix ?prefix "uf") in
    let init = f init (x.bh, with_prefix ?prefix "bh") in
    let init = f init (x.bf, with_prefix ?prefix "bf") in
    f init (x.wh, with_prefix ?prefix "wh")
end

module Mini_GRU_IO_P = struct
  type 'a prm =
    { uf : 'a
    ; wh : 'a
    ; uh : 'a
    ; bh : 'a
    ; b : 'a option
    }
  [@@deriving accessors ~submodule:A]

  let map ~f x =
    { uf = f x.uf; bh = f x.bh; uh = f x.uh; wh = f x.wh; b = Option.map ~f x.b }


  let fold ?prefix ~init ~f x =
    let init = f init (x.uh, with_prefix ?prefix "uh") in
    let init = f init (x.uf, with_prefix ?prefix "uf") in
    let init = f init (x.bh, with_prefix ?prefix "bh") in
    let init = f init (x.wh, with_prefix ?prefix "wh") in
    match x.b with
    | None -> init
    | Some b -> f init (b, with_prefix ?prefix "b")
end