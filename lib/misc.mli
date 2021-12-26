open Base

val save_bin : out:String.t -> 'a -> unit
val load_bin : String.t -> 'a
val time_this : label:String.t -> (unit -> 'a) -> 'a
