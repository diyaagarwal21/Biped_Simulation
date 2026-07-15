import os
import trimesh

def reduce_stl_faces(file_path, target_max_faces=15000):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Processing mesh: {os.path.basename(file_path)}...")
    
    # Load the asset
    loaded = trimesh.load(file_path)
    
    # FIX: If it's a Scene (multi-body STL), merge it into a single mesh object
    if isinstance(loaded, trimesh.Scene):
        if len(loaded.geometry) == 0:
            print(f" -> Warning: {os.path.basename(file_path)} contains no geometry data.")
            return
        mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
    else:
        mesh = loaded

    current_faces = len(mesh.faces)
    print(f" -> Current face count: {current_faces}")
    
    if current_faces > target_max_faces:
        # Calculate fractional target drop
        reduction_fraction = 1.0 - (target_max_faces / current_faces)
        reduction_fraction = max(0.01, min(0.99, reduction_fraction))
        
        # Simplify geometry faces
        mesh = mesh.simplify_quadric_decimation(reduction_fraction)
        print(f" -> Reduced to: {len(mesh.faces)} faces.")
    
    # Save optimized copy strictly as standard BINARY STL format
    mesh.export(file_path, file_type='stl')
    print(f" -> Successfully saved binary optimized copy!\n")

# Target all required meshes explicitly 
script_dir = os.path.dirname(os.path.abspath(__file__))
mesh_folder = os.path.join(script_dir, 'STLs', 'meshes')

# Scan the entire meshes folder automatically to capture everything
if os.path.exists(mesh_folder):
    boom_files = [os.path.join(mesh_folder, f) for f in os.listdir(mesh_folder) if f.lower().endswith('.stl')]
else:
    # Fallback to the default STLs folder if meshes doesn't exist
    stl_folder = os.path.join(script_dir, 'STLs')
    boom_files = [os.path.join(stl_folder, f) for f in os.listdir(stl_folder) if f.lower().endswith('.stl')]

for file_path in boom_files:
    try:
        reduce_stl_faces(file_path)
    except Exception as e:
        print(f"Failed processing {os.path.basename(file_path)}: {e}")

print("All boom assemblies are fully optimized for MuJoCo!")
