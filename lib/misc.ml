open Base

let save_bin ~out:filename m =
  let output = Stdio.Out_channel.create filename in
  Caml.Marshal.to_channel output m [ Caml.Marshal.No_sharing ];
  Stdio.Out_channel.close output


let load_bin filename =
  let input = Stdio.In_channel.create filename in
  let m = Caml.Marshal.from_channel input in
  Stdio.In_channel.close input;
  m


let save_zip_mat ?(sep = "\t") ~out x =
  let open Owl_dense_matrix_generic in
  (* will be AND'ed with user's umask *)
  let _op = Owl_utils.elt_to_str (kind x) in
  let h = Gzip.open_out ~level:9 out in
  let cr = Bytes.of_string "\n" in
  Stdlib.Fun.protect
    (fun () ->
      iter_rows
        (fun y ->
          iter
            (fun z ->
              let s = Bytes.of_string (Printf.sprintf "%s%s" (_op z) sep) in
              Gzip.output h s 0 (Bytes.length s))
            y;
          Gzip.output h cr 0 (Bytes.length cr))
        x)
    ~finally:(fun () -> Gzip.close_out h)


let save_mat ?(zip = false) ?sep ~out x =
  if zip then save_zip_mat ?sep ~out x else Owl.Mat.save_txt ?sep ~out x


let time_this ~label f =
  let t0 = Unix.gettimeofday () in
  let res = f () in
  let t = Unix.gettimeofday () -. t0 in
  C.print_endline (Printf.sprintf "[%s] %f seconds\n%!" label t);
  res
