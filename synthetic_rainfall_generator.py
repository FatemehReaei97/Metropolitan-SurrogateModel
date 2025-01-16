import numpy as np
import datetime
import pandas as pd
from typing import Dict, List, Tuple

class RainfallGenerator:
    """A class to generate synthetic rainfall events for urban flood modeling.
    
    This generator creates rainfall events with specified characteristics:
    - Peak intensity
    - Total depth
    - Time to peak
    - Duration
    
    The generated events are used as inputs to SWMM model for training surrogate models
    for flood prediction in the Houston metropolitan area. The events cover a wide range
    of possible rainfall patterns to ensure robust model training.
    
    Attributes:
        threshold: Minimum rainfall intensity to consider valid (default: 0.01)
    """
    
    def __init__(self, threshold: float = 0.01):
        """Initialize rainfall generator with parameters.
        
        Args:
            threshold: Minimum rainfall intensity to consider (mm/hr)
        """
        self.threshold = threshold
        
    def generate_random_values(self, n: int, total: float, peak: float) -> List[float]:
        """Generate random rainfall intensities following physical constraints.
        
        Uses a sorting-based approach to generate rainfall values that:
        1. Sum to the specified total depth
        2. Don't exceed the peak intensity
        3. Stay above the minimum threshold
        
        Args:
            n: Number of time steps
            total: Total rainfall depth to distribute
            peak: Maximum allowable rainfall intensity
            
        Returns:
            List of rainfall intensities or None if constraints cannot be met
        """
        for _ in range(n):
            values = [1, total] + list(np.random.uniform(low=1, high=total, size=n-1))
            if all(v > self.threshold and v < peak for v in values):
                values.sort()
                return [values[i+1] - values[i] for i in range(n)]
        return None

    def generate_events(self, 
                       peak_range: Tuple[int, int, int] = (10, 30, 5),
                       timestep_range: Tuple[int, int, int] = (20, 420, 40),
                       depth_range: Tuple[int, int, int] = (50, 1500, 50),
                       min_time_to_peak: int = 10) -> Dict:
        """Generate multiple rainfall events with varying characteristics.
        
        Creates a set of synthetic events by varying:
        - Peak intensity
        - Number of time steps (duration)
        - Total depth
        - Time to peak
        
        Args:
            peak_range: (start, stop, step) for peak intensity values
            timestep_range: (start, stop, step) for number of time steps
            depth_range: (start, stop, step) for total rainfall depth
            min_time_to_peak: Minimum time steps before peak can occur
            
        Returns:
            Tuple of (events dictionary, list of peak values)
            events dictionary: {event_id: list of rainfall intensities}
        """
        events = {}
        count = 0
        peaks = []

        for peak in range(*peak_range):
            for tstep in range(*timestep_range):
                for depth in range(*depth_range):
                    for tpeak in range(min_time_to_peak, tstep-2, 50):
                        rainfall = self.generate_random_values(tstep-1, depth - peak, peak)
                        if rainfall:
                            rainfall.insert(tpeak-1, peak)
                            events[count] = rainfall
                            peaks.append(peak)
                            count += 1
        
        return events, peaks

    def filter_events(self, events: Dict, peaks: List[float]) -> Dict:
        """Filter events to ensure they meet peak intensity constraints.
        
        Removes any events where any intensity value exceeds its designated peak.
        This ensures physical consistency of the generated events.
        
        Args:
            events: Dictionary of generated rainfall events
            peaks: List of peak values corresponding to each event
            
        Returns:
            Dictionary of filtered events that meet all constraints
        """
        filtered_events = {}
        for idx, (key, values) in enumerate(events.items()):
            if all(val <= peaks[idx] for val in values):
                filtered_events[idx] = values
        return filtered_events

    def save_to_excel(self, events: Dict, peaks: List[float], filename: str = 'parameters.xlsx'):
        """Save event parameters to Excel file for analysis and documentation.
        
        Creates a dataframe with columns:
        - peak: Maximum intensity
        - tstep: Number of time steps
        - depth: Total rainfall depth
        - tpeak: Time to peak
        
        Args:
            events: Dictionary of rainfall events
            peaks: List of peak intensities
            filename: Output Excel file name
        """
        data = []
        for idx, key in enumerate(events):
            data.append({
                'peak': peaks[idx],
                'tstep': len(events[key]),
                'depth': sum(events[key]),
                'tpeak': key + 1
            })
        df = pd.DataFrame(data)
        df.to_excel(filename, index=False)

    def write_to_swmm(self, events: Dict, output_dir: str = 'harvey'):
        """Write rainfall events to SWMM input format.
        
        Creates SWMM input files for each event, using a template structure
        with rainfall data inserted at appropriate locations.
        
        Args:
            events: Dictionary of rainfall events
            output_dir: Directory to save SWMM input files
        """
        # Read template parts
        with open("part1.txt", "rt") as f:
            part1 = f.read()
        with open("part3.txt", "rt") as f:
            part3 = f.read()

        # Generate files for each event
        for event_id, rainfall in events.items():
            part2 = ""
            dt = datetime.datetime(2017, 8, 22)
            
            # Create rainfall time series
            for val in rainfall:
                part2 += (f"harvey\t{dt.month}/{dt.day}/{dt.year}\t"
                         f"{dt.hour}:{dt.minute}\t{val}\n")
                dt += datetime.timedelta(minutes=30)
            
            # Write complete SWMM input file
            with open(f"{output_dir}/h{event_id}.inp", "wt") as f:
                f.write(part1 + part2 + "\n" + part3)

