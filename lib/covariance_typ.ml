open Owl_parameters

module Covariance_P = struct
  type 'a prm =
    { d : 'a
    ; t : 'a
    }
  [@@deriving accessors ~submodule:A]

  let map ~f x = { d = f x.d; t = f x.t }

  let fold ?prefix ~init ~f x =
    let d = f init (x.d, with_prefix ?prefix "d") in
    f d (x.t, with_prefix ?prefix "t")
end
