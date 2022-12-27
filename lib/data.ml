open Printf
open Base

type 'o output =
  { hash : string (* unique hash -- useful e.g. to recycle u in iLQR *)
  ; o : 'o
  ; ext_u : AD.t Option.t
  }

type (_, 'o) t =
  { id : int (* some trial ID used e.g. for saving to file *)
  ; u : AD.t Option.t
  ; z : AD.t Option.t
  ; o : 'o output
  }

let pack ?(id = 0) ?(ext_u = None) o =
  let hash = String.of_char_list (List.init 10 ~f:(fun _ -> Random.char ())) in
  { id; u = None; z = None; o = { hash; o; ext_u } }


let fill ~u ~z x = { x with u = Some u; z = Some z }
let hash x = x.o.hash
let id x = x.id
let u x = Option.value_exn x.u
let u_ext x = x.o.ext_u
let z x = Option.value_exn x.z
let o x = x.o.o
let reset_ids x = Array.mapi x ~f:(fun i xi -> { xi with id = i })

let save ?zip ?prefix save_o data =
  Array.iter data ~f:(fun d ->
      let with_prefix s =
        match prefix with
        | None -> s
        | Some p -> sprintf "%s.%s.%i" p s d.id
      in
      Option.iter d.u ~f:(fun u ->
          Misc.save_mat ?zip ~out:(with_prefix "u") (AD.unpack_arr u));
      Option.iter d.z ~f:(fun z ->
          Misc.save_mat ?zip ~out:(with_prefix "z") (AD.unpack_arr z));
      save_o ?zip ?prefix:(Some (with_prefix "o")) d.o.o)


let distribute x =
  C.scatter
    (if C.first
    then (
      let x = Lazy.force x in
      Array.init C.n_nodes ~f:(fun i ->
          Array.foldi x ~init:[] ~f:(fun j accu xj ->
              if j % C.n_nodes = i then xj :: accu else accu)
          |> Array.of_list_rev))
    else [||])


let split_and_distribute ?(reuse = false) ~prefix ~train x =
  let train_file = prefix ^ ".train.bin" in
  let test_file = prefix ^ ".test.bin" in
  let x_train, x_test =
    if reuse
    then lazy (Misc.load_bin train_file), lazy (Misc.load_bin test_file)
    else
      ( Lazy.map x ~f:(fun x ->
            let y = Array.sub x ~pos:0 ~len:train in
            let y = reset_ids y in
            Misc.save_bin ~out:train_file y;
            y)
      , Lazy.map x ~f:(fun x ->
            let y = Array.sub x ~pos:train ~len:(Array.length x - train) in
            let y = reset_ids y in
            Misc.save_bin ~out:test_file y;
            y) )
  in
  let _ =
    Stdio.printf
      "%i %i %!"
      (Array.length (Lazy.force x_train))
      (Array.length (Lazy.force x_test))
  in
  distribute x_train, distribute x_test
