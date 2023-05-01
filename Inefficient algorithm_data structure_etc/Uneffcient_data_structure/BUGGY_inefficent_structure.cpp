  : m_localization_id(0)
# endif
  {
    visited_facets[0] = 
      visited_facets[1] = 
      visited_facets[2] = 
      visited_facets[3] = false;
  }

public: 
Expand All
	@@ -123,26 +120,34 @@ class Compact_mesh_cell_base_3_base<Parallel_tag>
  {
    ++this->m_erase_counter;
  }

  /// Marks \c facet as visited
  void set_facet_visited (const int facet)
  {
    CGAL_precondition(facet>=0 && facet<4);
    visited_facets[facet] = true;
  }

  /// Marks \c facet as not visited
  void reset_visited (const int facet)
  {
    CGAL_precondition(facet>=0 && facet<4);
    visited_facets[facet] = false;
  }

  /// Returns \c true if \c facet is marked as visited
  bool is_facet_visited (const int facet) const
  {
    CGAL_precondition(facet>=0 && facet<4);
    return visited_facets[facet];
  }

# ifdef CGAL_MESH_3_TASK_SCHEDULER_WITH_LOCALIZATION_IDS
Expand All
	@@ -165,8 +170,8 @@ class Compact_mesh_cell_base_3_base<Parallel_tag>
#endif

private:
  /// Stores visited facets
  int visited_facets[4]; // CJTODO: ne pas mettre bool car risque de data race. A remplacer par un tbb::atomic<char> et utiliser CAS pour le mettre à jour bit à bit
};
#endif // CGAL_LINKED_WITH_TBB