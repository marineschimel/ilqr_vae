open Base
open Owl_parameters

module P = struct
  type ('g, 'r) prm_ =
    { generative : 'g
    ; recognition : 'r
    }
  [@@deriving accessors ~submodule:A]

  module Make (G : Owl_parameters.T) (R : Owl_parameters.T) = struct
    type 'a prm = ('a G.prm, 'a R.prm) prm_

    let map ~f prms =
      { generative = G.map ~f prms.generative; recognition = R.map ~f prms.recognition }


    let fold ?prefix ~init ~f prms =
      let w = with_prefix ?prefix in
      G.fold ~prefix:(w "generative") ~init ~f prms.generative
      |> fun init -> R.fold ~prefix:(w "recognition") ~init ~f prms.recognition
  end
end
