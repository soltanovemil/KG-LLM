import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Optional, Dict
import seaborn as sns

class KnowledgeGraphVisualizer:
    """Class for visualizing knowledge graphs."""

    def __init__(self, graph: nx.Graph):
        """
        Initialize visualizer.

        Args:
            graph: NetworkX graph to visualize
        """
        self.graph = graph

    def plot_static(
        self,
        figsize: tuple = (15, 10),
        node_size: int = 2500,
        node_color: str = 'lightblue',
        edge_color: str = 'gray',
        font_size: int = 10,
        title: str = "Knowledge Graph",
        save_path: Optional[str] = None
    ) -> None:
        """
        Create static visualization of the graph.

        Args:
            figsize: Figure size
            node_size: Size of nodes
            node_color: Color of nodes
            edge_color: Color of edges
            font_size: Size of font
            title: Plot title
            save_path: Path to save the plot
        """
        plt.figure(figsize=figsize)

        # Generate layout
        layout = nx.spring_layout(self.graph, seed=42)

        # Draw nodes and edges
        nx.draw(
            self.graph,
            layout,
            with_labels=True,
            node_size=node_size,
            node_color=node_color,
            edge_color=edge_color,
            font_size=font_size,
            font_weight='bold'
        )

        # Add relationship labels
        edge_labels = nx.get_edge_attributes(self.graph, 'relationship_type')
        nx.draw_networkx_edge_labels(
            self.graph,
            layout,
            edge_labels=edge_labels,
            font_color='red',
            font_size=8
        )

        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    def plot_interactive(
        self,
        title: str = "Interactive Knowledge Graph",
        save_path: Optional[str] = None
    ) -> None:
        """
        Create interactive visualization of the graph.

        Args:
            title: Plot title
            save_path: Path to save the plot HTML
        """
        # Generate layout
        layout = nx.spring_layout(self.graph, seed=42)

        # Prepare edge trace
        edge_x = []
        edge_y = []
        edge_text = []

        for edge in self.graph.edges(data=True):
            x0, y0 = layout[edge[0]]
            x1, y1 = layout[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_text.append(edge[2]['relationship_type'])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1, color='gray'),
            hoverinfo='text',
            text=edge_text,
            mode='lines'
        )

        # Prepare node trace
        node_x = []
        node_y = []
        node_text = []

        for node in self.graph.nodes():
            x, y = layout[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="top center",
            marker=dict(
                size=20,
                color='lightblue',
                line_width=2
            )
        )

        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=title,
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )

        if save_path:
            fig.write_html(save_path)
        fig.show()

    def plot_relation_distribution(
        self,
        figsize: tuple = (10, 6),
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot distribution of relationship types.

        Args:
            figsize: Figure size
            save_path: Path to save the plot
        """
        # Count relationships
        relationship_counts = {}
        for _, _, data in self.graph.edges(data=True):
            rel_type = data['relationship_type']
            if rel_type not in relationship_counts:
                relationship_counts[rel_type] = 0
            relationship_counts[rel_type] += 1

        # Create plot
        plt.figure(figsize=figsize)
        sns.barplot(
            x=list(relationship_counts.keys()),
            y=list(relationship_counts.values()),
            palette='viridis'
        )

        plt.title('Distribution of Relationship Types', fontsize=14)
        plt.xlabel('Relationship Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    def plot_subgraph(
        self,
        subgraph: nx.Graph,
        title: str = "Knowledge Graph Subgraph",
        interactive: bool = True,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot a subgraph.

        Args:
            subgraph: Subgraph to plot
            title: Plot title
            interactive: Whether to create interactive plot
            save_path: Path to save the plot
        """
        temp_visualizer = KnowledgeGraphVisualizer(subgraph)
        if interactive:
            temp_visualizer.plot_interactive(title=title, save_path=save_path)
        else:
            temp_visualizer.plot_static(title=title, save_path=save_path)
