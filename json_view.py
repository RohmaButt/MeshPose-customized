import json
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

class MeshPoseAnalyzer:
    def __init__(self, json_file_path: str):
        """
        Analyze MeshPose JSON output and implement texture mapping
        
        Args:
            json_file_path: Path to the JSON file containing mesh predictions
        """
        self.json_file_path = json_file_path
        self.predictions = None
        self.analysis_results = {}
        
    def analyze_json_structure(self):
        """Analyze the JSON file structure and contents"""
        print(f"Analyzing JSON file: {self.json_file_path}")
        print("=" * 60)
        
        try:
            with open(self.json_file_path, 'r') as f:
                self.predictions = json.load(f)
        except FileNotFoundError:
            print(f"Error: File {self.json_file_path} not found!")
            return
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {self.json_file_path}")
            return
            
        # Basic structure analysis
        print(f"JSON Type: {type(self.predictions)}")
        print(f"Total frames: {len(self.predictions)}")
        print()
        
        # Analyze first few frames
        for frame_idx in range(min(3, len(self.predictions))):
            print(f"FRAME {frame_idx}:")
            frame_data = self.predictions[frame_idx]
            print(f"  Type: {type(frame_data)}")
            print(f"  Number of detected people: {len(frame_data)}")
            
            # Analyze each person in this frame
            for person_idx, person_data in enumerate(frame_data):
                print(f"  PERSON {person_idx}:")
                print(f"    Type: {type(person_data)}")
                print(f"    Keys: {list(person_data.keys())}")
                
                # Analyze each key in detail
                for key, value in person_data.items():
                    if isinstance(value, list):
                        if len(value) > 0 and isinstance(value[0], list):
                            # 2D array
                            print(f"    {key}: shape {len(value)}x{len(value[0])} (2D array)")
                            if len(value) < 10:  # Show small arrays
                                print(f"      Sample: {value[:2]}...")
                        else:
                            # 1D array
                            print(f"    {key}: length {len(value)} (1D array)")
                            if len(value) < 20:  # Show small arrays
                                print(f"      Values: {value}")
                    else:
                        print(f"    {key}: {type(value)} = {value}")
                print()
            print("-" * 40)
            
        # Summary statistics
        self.generate_summary_stats()
        
    def generate_summary_stats(self):
        """Generate summary statistics about the data"""
        print("\nSUMMARY STATISTICS:")
        print("=" * 60)
        
        # Count keys across all frames and people
        key_counts = defaultdict(int)
        key_shapes = defaultdict(list)
        
        for frame_data in self.predictions:
            for person_data in frame_data:
                for key, value in person_data.items():
                    key_counts[key] += 1
                    if isinstance(value, list):
                        if len(value) > 0 and isinstance(value[0], list):
                            shape = f"{len(value)}x{len(value[0])}"
                        else:
                            shape = f"{len(value)}"
                        key_shapes[key].append(shape)
        
        print("Key frequency and shapes:")
        for key, count in key_counts.items():
            shapes = list(set(key_shapes[key]))
            print(f"  {key}: appears {count} times, shapes: {shapes}")
        
        # Check for 3D coordinates
        potential_3d_keys = []
        for key, shapes in key_shapes.items():
            for shape in shapes:
                if 'x3' in shape or (shape.count('x') == 0 and shape.isdigit() and int(shape) % 3 == 0):
                    potential_3d_keys.append(key)
                    break
        
        if potential_3d_keys:
            print(f"\nPotential 3D coordinate keys: {potential_3d_keys}")
        
    def extract_mesh_data(self, frame_idx: int = 0, person_idx: int = 0):
        """Extract mesh data for a specific frame and person"""
        if not self.predictions:
            print("No predictions loaded. Run analyze_json_structure() first.")
            return None
            
        try:
            person_data = self.predictions[frame_idx][person_idx]
            print(f"\nExtracting mesh data for Frame {frame_idx}, Person {person_idx}:")
            print("Available keys:", list(person_data.keys()))
            return person_data
        except (IndexError, KeyError) as e:
            print(f"Error extracting data: {e}")
            return None

class MeshTextureMapper:
    def __init__(self, mesh_data: Dict):
        """
        Initialize texture mapper with actual mesh data
        
        Args:
            mesh_data: Dictionary containing mesh predictions for one person
        """
        self.mesh_data = mesh_data
        self.vertices = None
        self.faces = None
        self.extract_vertices()
        
    def extract_vertices(self):
        """Extract 3D vertices from mesh data"""
        # Try to find 3D coordinates
        possible_vertex_keys = ['xyz_hp', 'vertices', 'xyz', 'coordinates', 'points']
        
        for key in possible_vertex_keys:
            if key in self.mesh_data:
                vertices = np.array(self.mesh_data[key])
                print(f"Found vertices in key '{key}', shape: {vertices.shape}")
                
                # Reshape if needed
                if vertices.ndim == 1 and len(vertices) % 3 == 0:
                    vertices = vertices.reshape(-1, 3)
                elif vertices.ndim == 2 and vertices.shape[1] == 3:
                    pass  # Already correct shape
                else:
                    print(f"Warning: Unexpected vertex shape: {vertices.shape}")
                    continue
                    
                self.vertices = vertices
                print(f"Extracted {len(vertices)} vertices")
                break
        
        if self.vertices is None:
            print("Could not find 3D vertices in the data!")
            print("Available keys:", list(self.mesh_data.keys()))
    
    def create_default_topology(self, num_vertices: int):
        """Create a default mesh topology for visualization"""
        # Create a simple topology - this is just for demonstration
        # In practice, you need the actual face indices from MeshPose
        
        if num_vertices < 3:
            return np.array([])
            
        # Create triangles connecting nearby vertices (very basic approach)
        faces = []
        for i in range(0, num_vertices - 2, 3):
            if i + 2 < num_vertices:
                faces.append([i, i + 1, i + 2])
        
        return np.array(faces) if faces else np.array([])
    
    def visualize_mesh(self, save_path: Optional[str] = None):
        """Visualize the 3D mesh"""
        if self.vertices is None:
            print("No vertices available for visualization")
            return
        
        # Create Open3D mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(self.vertices)
        
        # Try to create faces if we don't have them
        if self.faces is None:
            self.faces = self.create_default_topology(len(self.vertices))
        
        if len(self.faces) > 0:
            mesh.triangles = o3d.utility.Vector3iVector(self.faces)
            mesh.compute_vertex_normals()
        
        # Visualize
        print(f"Visualizing mesh with {len(self.vertices)} vertices and {len(self.faces)} faces")
        o3d.visualization.draw_geometries([mesh])
        
        if save_path:
            o3d.io.write_triangle_mesh(save_path, mesh)
            print(f"Saved mesh to {save_path}")
    
    def apply_texture_from_image(self, texture_image_path: str):
        """Apply texture from an image (placeholder implementation)"""
        if self.vertices is None:
            print("No vertices available for texturing")
            return
        
        # Load texture image
        try:
            texture = cv2.imread(texture_image_path)
            texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
            print(f"Loaded texture image: {texture.shape}")
        except:
            print(f"Could not load texture image: {texture_image_path}")
            return
        
        # This is where you would implement UV mapping
        # For now, just showing the structure
        print("Texture mapping requires UV coordinates and proper mesh topology")
        print("This would map 2D texture coordinates to 3D mesh vertices")

def main():
    # Usage example
    json_file = "/content/MeshPose/output_videos/demo/demo.json"
    
    if not Path(json_file).exists():
        print(f"File {json_file} does not exist!")
        return
    
    # Analyze the JSON structure
    analyzer = MeshPoseAnalyzer(json_file)
    analyzer.analyze_json_structure()
    
    # Extract mesh data for first person in first frame
    mesh_data = analyzer.extract_mesh_data(frame_idx=0, person_idx=0)
    
    if mesh_data:
        # Create texture mapper
        mapper = MeshTextureMapper(mesh_data)
        
        # Visualize mesh if vertices were found
        if mapper.vertices is not None:
            print("\nDo you want to visualize the 3D mesh? (y/n): ", end="")
            if input().lower().startswith('y'):
                mapper.visualize_mesh()
        
        # Option to apply texture
        print("\nDo you want to apply texture? (y/n): ", end="")
        if input().lower().startswith('y'):
            texture_path = input("Enter path to texture image: ").strip()
            mapper.apply_texture_from_image(texture_path)

if __name__ == "__main__":
    main()