open Base

val save_bin : out:String.t -> 'a -> unit
val load_bin : String.t -> 'a
val save_zip_mat : ?sep:String.t -> out:String.t -> Owl.Mat.mat -> unit
val save_mat : ?zip:bool -> ?sep:String.t -> out:String.t -> Owl.Mat.mat -> unit
val time_this : label:String.t -> (unit -> 'a) -> 'a
