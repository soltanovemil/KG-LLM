import networkx as nx
from typing import List, Dict, Optional
import pandas as pd
import logging

class KnowledgeGraphBuilder:
    """Class for building and managing knowledge graphs."""

    def __init__(self):
        """Initialize KnowledgeGraphBuilder."""
        self.graph = nx.Graph()
        self.data = None

    def build_from_dataframe(self, df: pd.DataFrame) -> None:
        """
        Build knowledge graph from DataFrame.

        Args:
            df: DataFrame containing subject, object, and relation columns
        """
        self.data = df

        # Add nodes and edges
        for _, row in df.iterrows():
            source = row['subject']
            target = row['object']
            relationship = row['relation']

            # Add nodes if they don't exist
            if source not in self.graph.nodes:
                self.graph.add_node(source)
            if target not in self.graph.nodes:
                self.graph.add_node(target)

            # Add edge with relationship
            self.graph.add_edge(source, target, relationship_type=relationship)

    def get_graph_stats(self) -> Dict:
        """
        Get statistics about the knowledge graph.

        Returns:
            Dictionary containing graph statistics
        """
        return {
            'num_nodes': len(self.graph.nodes),
            'num_edges': len(self.graph.edges),
            'num_components': nx.number_connected_components(self.graph),
            'average_degree': sum(dict(self.graph.degree()).values()) / len(self.graph),
            'density': nx.density(self.graph),
            'is_connected': nx.is_connected(self.graph)
        }

    def get_relationship_stats(self) -> Dict:
        """
        Get statistics about relationships in the graph.

        Returns:
            Dictionary containing relationship statistics
        """
        relationship_counts = {}
        for _, _, data in self.graph.edges(data=True):
            rel_type = data['relationship_type']
            if rel_type not in relationship_counts:
                relationship_counts[rel_type] = 0
            relationship_counts[rel_type] += 1

        return relationship_counts

    def get_central_nodes(self, top_k: int = 10) -> Dict:
        """
        Get most central nodes using different centrality measures.

        Args:
            top_k: Number of top nodes to return

        Returns:
            Dictionary containing central nodes by different measures
        """
        centrality_measures = {
            'degree': nx.degree_centrality(self.graph),
            'betweenness': nx.betweenness_centrality(self.graph),
            'closeness': nx.closeness_centrality(self.graph)
        }

        top_nodes = {}
        for measure, scores in centrality_measures.items():
            sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_nodes[measure] = sorted_nodes[:top_k]

        return top_nodes

    def filter_by_relation(self, relation_type: str) -> nx.Graph:
        """
        Create subgraph containing only specified relation type.

        Args:
            relation_type: Type of relation to filter by

        Returns:
            Filtered graph
        """
        filtered_graph = nx.Graph()

        for source, target, data in self.graph.edges(data=True):
            if data['relationship_type'] == relation_type:
                filtered_graph.add_node(source)
                filtered_graph.add_node(target)
                filtered_graph.add_edge(
                    source, target,
                    relationship_type=relation_type
                )

        return filtered_graph

    def get_node_neighborhood(
        self,
        node: str,
        depth: int = 1
    ) -> nx.Graph:
        """
        Get subgraph of node's neighborhood.

        Args:
            node: Central node
            depth: Neighborhood depth

        Returns:
            Neighborhood subgraph
        """
        nodes = {node}
        for _ in range(depth):
            for n in list(nodes):
                nodes.update(self.graph.neighbors(n))

        return self.graph.subgraph(nodes)

    def save_graph(self, filepath: str) -> None:
        """
        Save graph to file.

        Args:
            filepath: Path to save file
        """
        nx.write_gpickle(self.graph, filepath)

    @classmethod
    def load_graph(cls, filepath: str) -> 'KnowledgeGraphBuilder':
        """
        Load graph from file.

        Args:
            filepath: Path to graph file

        Returns:
            KnowledgeGraphBuilder instance
        """
        builder = cls()
        builder.graph = nx.read_gpickle(filepath)
        return builder
