# In src/simulation/swmm_parallel_runner.py

import os
from pyswmm import Simulation
import sys
import csv
import time
from typing import List, Dict, Tuple
from datetime import datetime

class SWMMParallelRunner:
    """Parallel SWMM simulation runner for HPC environments.
    
    Assumptions:
    1. SWMM input files are named 'h{index}.inp' where index ranges from 0 to n
    2. PySWMM is properly installed on the HPC system
    3. All required SWMM input files are present in the specified directory
    4. HPC environment has sufficient resources to run parallel simulations
    5. User has write permissions in the output directory
    """
    
    def __init__(self, base_path: str = None):
        """Initialize SWMM runner with paths."""
        self.base_path = base_path or os.getcwd()
        self.input_dir = os.path.join(self.base_path, 'input', 'swmm_files')
        self.output_dir = os.path.join(self.base_path, 'output')
        self.results_dir = os.path.join(self.output_dir, 'results')
        self.runtime_dir = os.path.join(self.output_dir, 'runtimes')
        
        # Create output directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.runtime_dir, exist_ok=True)
        
    def run_simulation(self, simulation_index: int) -> Tuple[List, float]:
        """Run a single SWMM simulation and track runtime.
        
        Args:
            simulation_index: Index of the SWMM input file to run
            
        Returns:
            Tuple of (simulation results, runtime in seconds)
        """
        inp_file_path = os.path.join(self.input_dir, f'h{simulation_index}.inp')
        print(f"Starting simulation for {inp_file_path}")
        
        try:
            # Initialize storage and timer
            node_data = []
            start_time = time.time()
            
            # Run simulation
            with Simulation(inp_file_path) as sim:
                for step in sim:
                    node_info = sim.nodes['NodeName']
                    node_data.append([
                        sim.current_time,
                        node_info.depth,
                        node_info.total_inflow
                    ])
            
            # Calculate runtime
            runtime = time.time() - start_time
            
            # Save results and runtime
            self._save_results(simulation_index, node_data)
            self._save_runtime(simulation_index, inp_file_path, runtime)
            
            return node_data, runtime
            
        except Exception as e:
            error_msg = f"Error in simulation {simulation_index}: {str(e)}"
            print(error_msg)
            self._save_runtime(simulation_index, inp_file_path, -1, error=error_msg)
            return [], -1
            
    def _save_results(self, simulation_index: int, node_data: List) -> None:
        """Save simulation results to CSV."""
        csv_file_path = os.path.join(self.results_dir, f'h{simulation_index}_report.csv')
        
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Time', 'Node Depth (m)', 'Total Inflow (cms)'])
            writer.writerows(node_data)
            
        print(f"Results saved to {csv_file_path}")
        
    def _save_runtime(self, simulation_index: int, inp_file: str, 
                     runtime: float, error: str = None) -> None:
        """Save runtime information to log file.
        
        Args:
            simulation_index: Index of the simulation
            inp_file: Path to input file
            runtime: Execution time in seconds
            error: Error message if simulation failed
        """
        log_file = os.path.join(self.runtime_dir, f'runtime_log_{simulation_index}.txt')
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(log_file, 'w') as file:
            if error:
                file.write(f"[{timestamp}] Simulation failed for {inp_file}\n")
                file.write(f"Error: {error}\n")
            else:
                file.write(f"[{timestamp}] Simulation for {inp_file}\n")
                file.write(f"Completed in {runtime:.2f} seconds\n")
                file.write(f"Average processing speed: {1/runtime:.2f} iterations/second\n")

def main():
    """Main entry point for HPC execution."""
    if len(sys.argv) != 2:
        print("Usage: python swmm_parallel_runner.py <simulation_index>")
        sys.exit(1)
        
    simulation_index = int(sys.argv[1])
    runner = SWMMParallelRunner('/path/to/your/project')
    
    # Run simulation and get results with runtime
    results, runtime = runner.run_simulation(simulation_index)
    
    if runtime > 0:
        print(f"Simulation {simulation_index} completed successfully in {runtime:.2f} seconds")
    else:
        print(f"Simulation {simulation_index} failed")

if __name__ == "__main__":
    main()