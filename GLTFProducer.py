'''
Created on 2023年8月9日

@author: Administrator
'''
'''
Created on 2023年8月7日

@author: Administrator
'''

import pathlib
import pygltflib
import numpy as np
from pygltflib import *

class GLTFProducer(object):
    
    def __init__(self):
        return
    
    def gltf_from_array(self, points, triangles, colors, line_points, line_indices, line_colors, output_path):
        
        triangles_binary_blob = triangles.flatten().tobytes()
        points_binary_blob = points.tobytes()
        colors_binary_blob = colors.flatten().tobytes()
        
        
        line_indices_binary_blob = line_indices.flatten().tobytes()
        line_points_binary_blob = line_points.tobytes()
        line_colors_binary_blob = line_colors.flatten().tobytes()

        # gltf 2
        gltf = pygltflib.GLTF2(
    
        # scences
        scene=0,
        scenes=[pygltflib.Scene(nodes=[0, 1])],
        nodes=[pygltflib.Node(mesh=0)],
        meshes=[
            pygltflib.Mesh(
                primitives=[
                pygltflib.Primitive(
                    attributes=pygltflib.Attributes(
                        POSITION=1, 
                        COLOR_0=2), 
                    indices=0,
                    material=0
                )
                ]   
            )
         ],
        # accessors
        accessors=[
        pygltflib.Accessor(
            bufferView=0,
            componentType=pygltflib.UNSIGNED_INT,
            count=triangles.size,
            type=pygltflib.SCALAR,
            max=[int(triangles.max())],
            min=[int(triangles.min())],
            ),
        pygltflib.Accessor(
            bufferView=1,
            componentType=pygltflib.FLOAT,
            count=len(points),
            type=pygltflib.VEC3,
            max=points.max(axis=0).tolist(),
            min=points.min(axis=0).tolist(),
           ),
        pygltflib.Accessor(
            bufferView=2,
            componentType=pygltflib.FLOAT,
            count=len(colors),
            type=pygltflib.VEC4,
            max=colors.max(axis=0).tolist(),
            min=colors.min(axis=0).tolist(),
            normalized = False,
           ),
         ],
       # bufferviews
        bufferViews=[
        pygltflib.BufferView(
            buffer=0,
            byteLength=len(triangles_binary_blob),
            target=pygltflib.ELEMENT_ARRAY_BUFFER,
          ),
        pygltflib.BufferView(
            buffer=0,
            byteOffset=len(triangles_binary_blob),
            byteLength=len(points_binary_blob),
            target=pygltflib.ARRAY_BUFFER,
          ),
        pygltflib.BufferView(
            buffer=0,
            byteOffset=len(triangles_binary_blob)+len(points_binary_blob),
            byteLength=len(colors_binary_blob),
            target=pygltflib.ARRAY_BUFFER,
          ),
        ],
        
        # buffers
        buffers=[
            pygltflib.Buffer(
                byteLength=len(triangles_binary_blob) + len(points_binary_blob) + len(colors_binary_blob)
                + len(line_indices_binary_blob) + len(line_points_binary_blob) + len(line_colors_binary_blob)
            )
        ],
        )
        gltf.set_binary_blob(triangles_binary_blob + points_binary_blob + colors_binary_blob 
                             + line_indices_binary_blob + line_points_binary_blob + line_colors_binary_blob)
        
        
        # materials
        material = Material()
        pbr = PbrMetallicRoughness() # Use PbrMetallicRoughness
        #pbr.baseColorFactor = [1.0, 1.0, 0.0, 0.1] # solid red
        material.pbrMetallicRoughness = pbr
        material.doubleSided = True # make material double sided
        material.alphaMode = BLEND   # to get around 'MATERIAL_ALPHA_CUTOFF_INVALID_MODE' warning
        material.alphaCutoff=None
        gltf.materials.append(material)
        
        
        gltf.nodes.append(pygltflib.Node(mesh=1)),
        gltf.meshes.append(pygltflib.Mesh(
                primitives=[
                pygltflib.Primitive(
                    attributes=pygltflib.Attributes(
                        POSITION=4, 
                        COLOR_0=5), 
                    indices=3,
                    mode = 1
                )
                ]   
            ))
    
        gltf.accessors.extend([
        pygltflib.Accessor(
            bufferView=3,
            componentType=pygltflib.UNSIGNED_INT,
            count=line_indices.size,
            type=pygltflib.SCALAR,
            max=[int(line_indices.max())],
            min=[int(line_indices.min())],
            ),
        pygltflib.Accessor(
            bufferView=4,
            componentType=pygltflib.FLOAT,
            count=len(line_points),
            type=pygltflib.VEC3,
            max=line_points.max(axis=0).tolist(),
            min=line_points.min(axis=0).tolist(),
           ),
        pygltflib.Accessor(
            bufferView=5,
            componentType=pygltflib.FLOAT,
            count=len(line_colors),
            type=pygltflib.VEC4,
            max=line_colors.max(axis=0).tolist(),
            min=line_colors.min(axis=0).tolist(),
            normalized = False,
           )])
            
        gltf.bufferViews.extend([
            pygltflib.BufferView(
            buffer=0,
            byteOffset=len(triangles_binary_blob) + len(points_binary_blob) + len(colors_binary_blob),
            byteLength=len(line_indices_binary_blob),
            target=pygltflib.ELEMENT_ARRAY_BUFFER,
          ),
            pygltflib.BufferView(
            buffer=0,
            byteOffset=len(triangles_binary_blob) + len(points_binary_blob) + len(colors_binary_blob) + len(line_indices_binary_blob),
            byteLength=len(line_points_binary_blob),
            target=pygltflib.ARRAY_BUFFER,
          ),
            pygltflib.BufferView(
            buffer=0,
            byteOffset=len(triangles_binary_blob) + len(points_binary_blob) + len(colors_binary_blob) + len(line_indices_binary_blob) + len(line_points_binary_blob),
            byteLength=len(line_colors_binary_blob),
            target=pygltflib.ARRAY_BUFFER,
          )])
        
        gltf.save(output_path)
        self.gltf = gltf
        return


    def decode_gltf(self):
        gltf = self.gltf
        binary_blob = gltf.binary_blob()
        
        triangles_accessor = gltf.accessors[gltf.meshes[0].primitives[0].indices]
        triangles_buffer_view = gltf.bufferViews[triangles_accessor.bufferView]
        triangles = np.frombuffer(
            binary_blob[
                triangles_buffer_view.byteOffset
                + triangles_accessor.byteOffset : triangles_buffer_view.byteOffset
                + triangles_buffer_view.byteLength
            ],
        dtype="uint32",
        count=triangles_accessor.count,
        ).reshape((-1, 3))
    
        points_accessor = gltf.accessors[gltf.meshes[0].primitives[0].attributes.POSITION]
        points_buffer_view = gltf.bufferViews[points_accessor.bufferView]
        
        points = np.frombuffer(
            binary_blob[
            points_buffer_view.byteOffset
        +   points_accessor.byteOffset : points_buffer_view.byteOffset
        + points_buffer_view.byteLength
        ],
        dtype="float32",
        count=points_accessor.count * 3,
        ).reshape((-1, 3))
                  
        colors_accessor = gltf.accessors[gltf.meshes[0].primitives[0].attributes.COLOR_0]
        colors_buffer_view = gltf.bufferViews[colors_accessor.bufferView]
        colors = np.frombuffer(
            binary_blob[
            colors_buffer_view.byteOffset
        +   colors_accessor.byteOffset : colors_buffer_view.byteOffset
        + colors_buffer_view.byteLength
        ],
        dtype="float32",
        count=colors_accessor.count * 4,
        ).reshape((-1, 4))
    
        return (points, triangles, colors)


if __name__ == "__main__":
    
    points = np.array(
    [
        [-0.5666666666666, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [-0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.5, -0.5, -0.5],
        [-0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [1.0, 0.5, 0.5],
        [0.5, -0.5, 0.5],
    ],
    dtype="float64",
    )
    
    colors = np.array(
    [
        [0.5, 0.5, 0.5, 0.5],
        [1, 0, 0, 0.5],
        [1, 0, 0, 0.5],
        [1, 1, 0, 0.5],
        [1, 0, 1, 0.5],
        [1, 0, 1, 0.5],
        [1, 0, 1, 0.5],
        [1, 1, 0, 0.5],
        [1, 1, 0, 0.5],
        [1, 1, 0, 0.5],
    ],
    dtype="float32",
    )
    
    triangles = np.array(
    [
        [0, 1, 2],
        [3, 2, 1],
        [1, 0, 4],
        [5, 4, 0],
        [3, 1, 6],
        [4, 6, 1],
        [2, 3, 7],
        [6, 7, 3],
        [0, 2, 5],
        [7, 5, 2],
        [5, 7, 4],
        [6, 4, 7],
        [8, 3, 9],
        
    ],
    dtype="uint32",
    )
    
    
    lines = np.array(
    [
        [0, 0, 0],
        [9, 9, 9],
        [10, 10, 10],
        [15, 15, 15],
        
    ],
    dtype="float32",
    )
    
    
    line_index = np.array(
    [
        [0, 1],
        [2, 3]
    ],
    dtype="uint32",
    )
    
    line_colors = np.array(
    [
        [1, 0, 0, 0.5],
        [1, 0, 0, 0.5],
        [1, 0, 0, 0.5],
        [1, 0, 0, 0.5]
    ],
    dtype="float32",
    ) 
    
    
    gp = GLTFProducer()
    print("input array =",points, triangles)
    print("input colors =", colors.shape, colors)
    gp.gltf_from_array(points, triangles, colors, lines, line_index, line_colors, "../output/test_color3.glb")
    (points_output, triangles_output, colors_output) = gp.decode_gltf()
    print("output array =", points_output, triangles_output, colors_output)
    
    
    
    