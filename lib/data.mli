open Base

type 'o output
type (_, 'o) t

val pack : ?id:int -> 'o -> ([ `o ], 'o) t
val fill : u:AD.t -> z:AD.t -> ([> `o ], 'o) t -> ([ `o | `uz ], 'o) t
val hash : (_, 'o) t -> String.t
val id : (_, 'o) t -> int
val u : ([> `uz ], 'o) t -> AD.t
val z : ([> `uz ], 'o) t -> AD.t
val o : (_, 'o) t -> 'o
val reset_ids : ('d, 'o) t Array.t -> ('d, 'o) t Array.t

val save
  :  ?prefix:String.t
  -> (?prefix:String.t -> 'o -> unit)
  -> (_, 'o) t Array.t
  -> unit

(** [distribute x] chuncks the (lazy) data array [x]
    into [C.n_nodes] arrays of (approximately) equal size,
    and distributes them to each MPI node;
    only the root node evaluates the lazy [x] *)
val distribute : ('typ, 'o) t Array.t Lazy.t -> ('typ, 'o) t Array.t

(** [split_and_distribute ?reuse ~prefix ~train x] 
    splits the lazy [x] into a train set and test set,
    saves each of them in binary form to [prefix ^ ".{train/test}.bin"],
    and distributes them across MPI nodes.
    Each node thus gets a pair of local train / test sets.
    Note that only the root node evaluates the lazy [x].

    If [reuse=true], then the data is instead read from
    [prefix ^ ".{train/test}.bin"] and distributed all the same. *)
val split_and_distribute
  :  ?reuse:bool
  -> prefix:String.t
  -> train:int
  -> ('typ, 'o) t Array.t Lazy.t
  -> ('typ, 'o) t Array.t * ('typ, 'o) t Array.t
