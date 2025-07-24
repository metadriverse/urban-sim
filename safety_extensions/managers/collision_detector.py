# Copyright (c) 2025, Simulation Toolkit Project
# Collision Detection System

import torch
from typing import List, Tuple, Optional
import numpy as np


class CollisionDetector:
    """Detects collisions and near-misses between robot and environment."""
    
    def __init__(self, collision_threshold: float = 0.0, near_miss_threshold: float = 0.5):
        """Initialize collision detector.
        
        Args:
            collision_threshold: Distance threshold for collision (meters)
            near_miss_threshold: Distance threshold for near-miss (meters)
        """
        self.collision_threshold = collision_threshold
        self.near_miss_threshold = near_miss_threshold
        
    def check_collision(
        self,
        robot_meshes: List[torch.Tensor],
        obstacle_meshes: List[torch.Tensor],
        safety_padding: float = 0.1
    ) -> Tuple[bool, float]:
        """Check for collision between robot and obstacles.
        
        Args:
            robot_meshes: List of robot collision meshes
            obstacle_meshes: List of obstacle meshes
            safety_padding: Additional safety margin (meters)
            
        Returns:
            Tuple of (is_collision, minimum_distance)
        """
        min_distance = float('inf')
        
        # Simplified distance check - in production use proper mesh collision
        for robot_mesh in robot_meshes:
            for obstacle_mesh in obstacle_meshes:
                distance = self._mesh_distance(robot_mesh, obstacle_mesh)
                min_distance = min(min_distance, distance)
                
                if distance <= (self.collision_threshold + safety_padding):
                    return True, distance
        
        return False, min_distance
    
    def _mesh_distance(self, mesh1: torch.Tensor, mesh2: torch.Tensor) -> float:
        """Calculate minimum distance between two meshes.
        
        Simplified implementation - use GJK or similar in production.
        """
        # For now, use centroid distance as approximation
        centroid1 = torch.mean(mesh1, dim=0)
        centroid2 = torch.mean(mesh2, dim=0)
        return torch.norm(centroid1 - centroid2).item()