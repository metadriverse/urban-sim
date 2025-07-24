# Copyright (c) 2025, Simulation Toolkit Project
# Safety Metrics Collection and Analysis

import torch
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict
import time


@dataclass
class SafetyMetrics:
    """Container for safety metrics."""
    total_collisions: int = 0
    total_near_misses: int = 0
    total_zone_violations: int = 0
    total_emergency_stops: int = 0
    
    min_human_distance: float = float('inf')
    avg_human_distance: float = 0.0
    
    max_velocity_violation: float = 0.0
    max_acceleration_violation: float = 0.0
    
    reaction_times: List[float] = field(default_factory=list)
    stopping_distances: List[float] = field(default_factory=list)
    
    time_in_zones: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    violation_history: List[Dict] = field(default_factory=list)


class SafetyMetricsCollector:
    """Collects and analyzes safety metrics during simulation."""
    
    def __init__(self, history_size: int = 1000):
        """Initialize metrics collector.
        
        Args:
            history_size: Maximum number of historical events to store
        """
        self.metrics = SafetyMetrics()
        self.history_size = history_size
        self.start_time = time.time()
        self.last_update_time = self.start_time
        
        # For calculating rates
        self._window_size = 100
        self._collision_window = []
        self._near_miss_window = []
        
    def update(self, violations: List, dt: float):
        """Update metrics based on current violations.
        
        Args:
            violations: List of SafetyViolation objects
            dt: Time step
        """
        current_time = time.time()
        
        for violation in violations:
            # Count by type
            if violation.violation_type == "collision":
                self.metrics.total_collisions += 1
                self._collision_window.append(current_time)
            elif violation.violation_type == "near_miss":
                self.metrics.total_near_misses += 1
                self._near_miss_window.append(current_time)
            elif violation.violation_type == "zone_violation":
                self.metrics.total_zone_violations += 1
                zone = violation.details.get("zone", "unknown")
                self.metrics.time_in_zones[zone] += dt
            elif violation.violation_type == "hardware_limit":
                self._update_hardware_violations(violation)
            
            # Track minimum distances
            if violation.object_id is not None and violation.details.get("object_type") == "human":
                self.metrics.min_human_distance = min(
                    self.metrics.min_human_distance,
                    violation.distance
                )
            
            # Store in history
            if len(self.metrics.violation_history) < self.history_size:
                self.metrics.violation_history.append({
                    "type": violation.violation_type,
                    "severity": violation.severity,
                    "distance": violation.distance,
                    "timestamp": violation.timestamp,
                    "details": violation.details
                })
        
        # Clean old entries from rate windows
        self._clean_rate_windows(current_time)
        self.last_update_time = current_time
    
    def record_emergency_stop(self, timestamp: float):
        """Record an emergency stop event.
        
        Args:
            timestamp: When the emergency stop occurred
        """
        self.metrics.total_emergency_stops += 1
        # In a real system, would calculate actual reaction time
        reaction_time = 0.05  # Placeholder
        self.metrics.reaction_times.append(reaction_time)
    
    def record_stopping_distance(self, distance: float):
        """Record measured stopping distance.
        
        Args:
            distance: Stopping distance in meters
        """
        self.metrics.stopping_distances.append(distance)
    
    def get_current_metrics(self) -> Dict:
        """Get current metrics snapshot."""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        return {
            "total_collisions": self.metrics.total_collisions,
            "total_near_misses": self.metrics.total_near_misses,
            "total_zone_violations": self.metrics.total_zone_violations,
            "total_emergency_stops": self.metrics.total_emergency_stops,
            "collision_rate": self._calculate_rate(self._collision_window, current_time),
            "near_miss_rate": self._calculate_rate(self._near_miss_window, current_time),
            "min_human_distance": self.metrics.min_human_distance,
            "avg_reaction_time": np.mean(self.metrics.reaction_times) if self.metrics.reaction_times else 0.0,
            "avg_stopping_distance": np.mean(self.metrics.stopping_distances) if self.metrics.stopping_distances else 0.0,
            "elapsed_time": elapsed_time,
            "time_in_zones": dict(self.metrics.time_in_zones)
        }
    
    def get_summary(self) -> Dict:
        """Get comprehensive metrics summary."""
        metrics = self.get_current_metrics()
        
        # Add safety score (0-100, higher is safer)
        safety_score = self._calculate_safety_score(metrics)
        metrics["safety_score"] = safety_score
        
        # Add performance metrics
        if metrics["elapsed_time"] > 0:
            metrics["mtbf"] = metrics["elapsed_time"] / max(1, metrics["total_collisions"])  # Mean time between failures
        
        return metrics
    
    def export_metrics(self, filepath: str):
        """Export metrics to file for analysis.
        
        Args:
            filepath: Where to save metrics
        """
        import json
        
        export_data = {
            "summary": self.get_summary(),
            "metrics": self.metrics.__dict__,
            "history": self.metrics.violation_history[-1000:]  # Last 1000 events
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
    
    def _update_hardware_violations(self, violation):
        """Update hardware-related violation metrics."""
        details = violation.details
        if details.get("limit_type") == "velocity":
            self.metrics.max_velocity_violation = max(
                self.metrics.max_velocity_violation,
                details.get("current", 0) - details.get("limit", 0)
            )
        elif details.get("limit_type") == "acceleration":
            self.metrics.max_acceleration_violation = max(
                self.metrics.max_acceleration_violation,
                details.get("current", 0) - details.get("limit", 0)
            )
    
    def _calculate_rate(self, window: List[float], current_time: float) -> float:
        """Calculate rate of events per minute."""
        if not window:
            return 0.0
        
        # Count events in last minute
        one_minute_ago = current_time - 60.0
        recent_events = [t for t in window if t > one_minute_ago]
        
        return len(recent_events)  # Events per minute
    
    def _clean_rate_windows(self, current_time: float):
        """Remove old entries from rate calculation windows."""
        one_minute_ago = current_time - 60.0
        
        self._collision_window = [t for t in self._collision_window if t > one_minute_ago]
        self._near_miss_window = [t for t in self._near_miss_window if t > one_minute_ago]
    
    def _calculate_safety_score(self, metrics: Dict) -> float:
        """Calculate overall safety score (0-100)."""
        score = 100.0
        
        # Deduct points for violations
        score -= metrics["total_collisions"] * 20  # Heavy penalty for collisions
        score -= metrics["total_near_misses"] * 5
        score -= metrics["total_zone_violations"] * 0.5
        score -= metrics["total_emergency_stops"] * 10
        
        # Consider rates
        score -= metrics["collision_rate"] * 10
        score -= metrics["near_miss_rate"] * 2
        
        # Bonus for good minimum distance
        if metrics["min_human_distance"] > 2.0:
            score += 10
        
        return max(0.0, min(100.0, score))