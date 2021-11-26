open Base

let save_bin ~out:filename m =
  let output = Stdio.Out_channel.create filename in
  Caml.Marshal.to_channel output m [ Caml.Marshal.No_sharing ];
  Stdio.Out_channel.close output


let read_bin filename =
  let input = Stdio.In_channel.create filename in
  let m = Caml.Marshal.from_channel input in
  Stdio.In_channel.close input;
  m


let time_this ~label f =
  let t0 = Unix.gettimeofday () in
  let res = f () in
  let t = Unix.gettimeofday () -. t0 in
  C.print_endline (Printf.sprintf "[%s] %f seconds\n%!" label t);
  res
