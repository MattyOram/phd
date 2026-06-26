import pyvista as pv
import zipfile

def mesh_files_to_multi_part_3mf(mesh_files, output_3mf_path):
    """Convert multiple STL files to a single 3MF with separate objects
    
    Args:
        mesh_files: List of tuples [(mesh_path, object_name), ...]
        output_3mf_path: Output 3MF file path
    """
    model_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
  <resources>
'''
    
    build_items = []
    
    for obj_id, (mesh_path, mesh_name) in enumerate(mesh_files, 1):
        stl_mesh = pv.read(mesh_path)
        
        model_xml += f'    <object id="{obj_id}" type="model" name="{mesh_name}">\n'
        model_xml += '      <mesh>\n        <vertices>\n'
        
        vertex_count = 0
        triangles = []
        
        # Add vertices and build triangles
        for triangle in stl_mesh.points[stl_mesh.faces.reshape(-1, 4)[:, 1:]]:
            triangle_indices = []
            for vertex in triangle:
                model_xml += f'          <vertex x="{vertex[0]:.6f}" y="{vertex[1]:.6f}" z="{vertex[2]:.6f}"/>\n'
                triangle_indices.append(vertex_count)
                vertex_count += 1
            triangles.append(triangle_indices)
        
        model_xml += '        </vertices>\n        <triangles>\n'
        
        for triangle in triangles:
            model_xml += f'          <triangle v1="{triangle[0]}" v2="{triangle[1]}" v3="{triangle[2]}"/>\n'
        
        model_xml += '        </triangles>\n      </mesh>\n    </object>\n'
        build_items.append(str(obj_id))
    
    # Add build section
    model_xml += '  </resources>\n  <build>\n'
    for obj_id in build_items:
        model_xml += f'    <item objectid="{obj_id}"/>\n'
    model_xml += '  </build>\n</model>'
    
    # Create the 3MF file (ZIP archive)
    with zipfile.ZipFile(output_3mf_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add the main model file
        zip_file.writestr('3D/3dmodel.model', model_xml)
        
        # Add required content types
        content_types = '''<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>'''
        zip_file.writestr('[Content_Types].xml', content_types)
        
        # Add relationships
        relationships = '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>'''
        zip_file.writestr('_rels/.rels', relationships)
