import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import json
import ast

class NetworkAnalyzer:
    """Analyzes urban drainage network connectivity for clustering.
    
    This class processes manhole and link data to identify connected components
    in the drainage network, supporting the clustering approach described in the paper.
    The analysis is based on manhole connectivity patterns and link characteristics.
    """
    
    def __init__(self):
        """Initialize network analyzer."""
        self.excluded_manholes = []
        self.to_deal_with_neighbor = []
        self.network_dict = {}
        
    def load_data(self, link_file: str, manhole_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load network data from Excel files.
        
        Args:
            link_file: Path to Excel file containing link data
            manhole_file: Path to Excel file containing manhole data
            
        Returns:
            Tuple of (manhole_matrix, link_matrix)
        """
        # Maximum number of rows to display when printing DataFrames
        # Set to a large value to ensure all network connections are visible
        # This is particularly important for large metropolitan networks like Houston
        # which has over 66,000 manholes and their connections
        pd.options.display.max_rows = 200000
        manhole_data = pd.read_excel(manhole_file)
        link_data = pd.read_excel(link_file)
        
        return np.array(manhole_data), np.array(link_data)
        
    def find_connected_links(self, current_man: int, link_mat: np.ndarray) -> List:
        """Find all links connected to a given manhole.
        
        Args:
            current_man: Manhole ID to analyze
            link_mat: Matrix of link data
            
        Returns:
            List of connected links with their properties
        """
        existing_links = []
        for row in range(len(link_mat)):
            if link_mat[row][1] == current_man:
                existing_links.append([link_mat[row][0], link_mat[row][1], link_mat[row][2]])
                self.to_deal_with_neighbor.append(link_mat[row][2])
            elif link_mat[row][2] == current_man:
                existing_links.append([link_mat[row][0], link_mat[row][1], link_mat[row][2]])
                self.to_deal_with_neighbor.append(link_mat[row][1])
        return existing_links
    
    # Minimum number of manholes required to form a valid cluster
    # Based on empirical analysis showing optimal performance
     
    def analyze_network(self, min_links: int = 20) -> Dict:
        """Analyze network connectivity and form clusters.
        
        Args:
            min_links: Minimum number of components for valid cluster
            
        Returns:
            Dictionary of network clusters
        """
        # Input data file paths
        # M3.xlsx contains manhole information
        # L3.xlsx contains link (pipe) information   
        manhole_mat, link_mat = self.load_data('M3.xlsx', 'L3.xlsx')
        
        for man_index in range(len(manhole_mat)):
            current_man = manhole_mat[man_index][0]
            
            if current_man not in self.excluded_manholes:
                network = self._process_manhole(current_man, link_mat)
                if network and len(network) > min_links:
                    self.network_dict[str(man_index)] = network
                    
        return self.network_dict
    
    def _process_manhole(self, current_man: int, link_mat: np.ndarray) -> List:
        """Process a single manhole and its connections.
        
        Args:
            current_man: Current manhole ID
            link_mat: Matrix of link data
            
        Returns:
            List of connected network elements
        """
        existing_links = self.find_connected_links(current_man, link_mat)
        if not existing_links:
            return None
            
        network = existing_links.copy()
        self.excluded_manholes.append(current_man)
        
        # Process neighbors
        item = 0
        while len(self.to_deal_with_neighbor) > 0 and item < len(self.to_deal_with_neighbor):
            neighbor = self.to_deal_with_neighbor[item]
            if neighbor not in self.excluded_manholes:
                new_links = self.find_connected_links(neighbor, link_mat)
                network.extend(new_links)
                self.excluded_manholes.append(neighbor)
            item += 1
            
        return network
        
    def save_results(self, filename: str = 'network_clusters.txt'):
        """Save network clustering results to file.
        
        Args:
            filename: Output filename
        """
        with open(filename, "wt") as f:
            st = json.loads(str(self.network_dict).replace("'", '"'))
            f.write(str(st))
