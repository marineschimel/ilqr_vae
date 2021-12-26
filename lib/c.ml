include Comm.Mpi (struct
  let init_rng seed =
    Owl_stats_prng.init seed;
    Random.init (Owl_stats_prng.rand_int ())
end)

(* make sure all MPI nodes have a different random seed *)
let _ = self_init_rng ()
