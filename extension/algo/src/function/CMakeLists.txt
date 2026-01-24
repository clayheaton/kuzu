add_subdirectory(config)

add_library(kuzu_algo_function
        OBJECT
        component_ids.cpp
        strongly_connected_components.cpp
        strongly_connected_components_kosaraju.cpp
        weakly_connected_components.cpp
        page_rank.cpp
        k_core_decomposition.cpp
        louvain.cpp
        spanning_forest.cpp
        graph_density.cpp
        clustering_coefficient.cpp
        eigenvector_centrality.cpp
        hits.cpp
        closeness_centrality.cpp
        betweenness_centrality.cpp
        # Layout algorithms
        random_layout.cpp
        circular_layout.cpp
        fruchterman_reingold_layout.cpp
        radial_layout.cpp
        hierarchical_layout.cpp
        yifan_hu_layout.cpp
        forceatlas2_layout.cpp
        )

set(ALGO_EXTENSION_OBJECT_FILES
        ${ALGO_EXTENSION_OBJECT_FILES} $<TARGET_OBJECTS:kuzu_algo_function>
        PARENT_SCOPE)
