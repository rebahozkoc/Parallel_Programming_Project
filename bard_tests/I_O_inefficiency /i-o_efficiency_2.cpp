//commit 4903a90
  num = ga_cluster_nnodes_();

        return( MIN( ga_nnodes_(), num));
}
Expand All
	@@ -258,8 +266,17 @@ Integer zero = 0;
        * if proc id beyond I/O procs number, negate it
        */

        if(me == ga_cluster_procid_(&nodeid, &zero)) me = nodeid;
        else me = -1;

/*        if (me >= dai_io_procs(d_a)) me = -me;*/
        return (me);