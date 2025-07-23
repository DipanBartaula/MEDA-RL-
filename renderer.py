import trimesh
import numpy as np
from PIL import Image

def run_cad_code(code_str, tmp_stl_path="tmp.stl"):
    """
    Executes `code_str`, expecting a trimesh.Trimesh in `mesh`.
    """
    local_vars = {}
    exec(code_str, {'trimesh': trimesh, 'np': np}, local_vars)
    mesh = local_vars.get('mesh', None)
    assert isinstance(mesh, trimesh.Trimesh), "Code must set `mesh`"
    mesh.export(tmp_stl_path)
    return mesh

def render_orthographic(mesh, image_size=256):
    scene = mesh.scene()
    scene.set_camera(
        angles=[0, 0, 0],
        distance=mesh.scale * 3,
        center=mesh.centroid,
        fov=0
    )
    data = scene.save_image(resolution=(image_size, image_size), visible=True)
    img = Image.open(trimesh.util.wrap_as_stream(data)).convert('L')
    return np.array(img) / 255.0
