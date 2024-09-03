import geopandas as gpd
import numpy as np
import geopandas as gpd
import numpy as np
import pygltflib

class GLTFProducer:
    def gltf_from_array(self, vertices, indices, colors, output_path):
        if vertices.size == 0:
            print("Error: Vertex array is empty. Skipping GLB generation.")
            return

        vertices_binary_blob = vertices.tobytes()
        colors_binary_blob = colors.tobytes()
        indices_binary_blob = indices.flatten().tobytes()

        gltf = pygltflib.GLTF2(
            scene=0,
            scenes=[pygltflib.Scene(nodes=[0])],
            nodes=[pygltflib.Node(mesh=0)],
            meshes=[
                pygltflib.Mesh(
                    primitives=[
                        pygltflib.Primitive(
                            attributes=pygltflib.Attributes(
                                POSITION=0,
                                COLOR_0=1
                            ),
                            indices=2,
                            mode=4  # GL_TRIANGLES
                        )
                    ]
                )
            ],
            accessors=[
                pygltflib.Accessor(
                    bufferView=0,
                    componentType=pygltflib.FLOAT,
                    count=vertices.shape[0],  # 顶点数量
                    type=pygltflib.VEC3,
                    max=vertices.max(axis=0).tolist(),
                    min=vertices.min(axis=0).tolist(),
                ),
                pygltflib.Accessor(
                    bufferView=1,
                    componentType=pygltflib.FLOAT,
                    count=colors.shape[0],  # 颜色数量应与顶点数量一致
                    type=pygltflib.VEC4,
                    max=colors.max(axis=0).tolist(),
                    min=colors.min(axis=0).tolist(),
                    normalized=False,
                ),
                pygltflib.Accessor(
                    bufferView=2,
                    componentType=pygltflib.UNSIGNED_INT,
                    count=len(indices.flatten()),  # 索引数量
                    type=pygltflib.SCALAR
                )
            ],
            bufferViews=[
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=0,
                    byteLength=len(vertices_binary_blob),
                    target=pygltflib.ARRAY_BUFFER,
                ),
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=len(vertices_binary_blob),
                    byteLength=len(colors_binary_blob),
                    target=pygltflib.ARRAY_BUFFER,
                ),
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=len(vertices_binary_blob) + len(colors_binary_blob),
                    byteLength=len(indices_binary_blob),
                    target=pygltflib.ELEMENT_ARRAY_BUFFER,
                )
            ],
            buffers=[
                pygltflib.Buffer(
                    byteLength=len(vertices_binary_blob) + len(colors_binary_blob) + len(indices_binary_blob),
                )
            ],
        )

        # 将顶点数据、颜色数据和索引数据打包成二进制 blob
        binary_blob = vertices_binary_blob + colors_binary_blob + indices_binary_blob
        gltf.set_binary_blob(binary_blob)

        # 保存 GLB 文件
        gltf.save(output_path)

    def gltf_from_shapefile(self, shapefile_path, output_path):
        gdf = gpd.read_file(shapefile_path)

        vertices = []
        indices = []
        colors = []
        fixed_color = [0.3, 0.3, 0.5, 1]  # 固定颜色，红色 (RGBA)

        for geom, attr_value in zip(gdf.geometry, gdf["HEIGHT"]):  # 假设 "HEIGHT" 是您要使用的属性名
            if geom.geom_type == "Polygon" or geom.geom_type == "MultiPolygon":
                for poly in [geom] if geom.geom_type == "Polygon" else geom.geoms:
                    exterior_coords = np.array(poly.exterior.coords)
                    if len(exterior_coords) > 0:
                        base_index = len(vertices)
                        height = attr_value  # 使用属性值作为 z 坐标

                        # 底面顶点
                        for coord in exterior_coords:
                            vertices.append([coord[0], coord[1], 0.0])  # z = 0
                            colors.append(fixed_color)

                        # 顶面顶点
                        for coord in exterior_coords:
                            vertices.append([coord[0], coord[1], height])  # z = height
                            colors.append(fixed_color)

                        num_points = len(exterior_coords)
                        if num_points > 3:
                            for i in range(num_points - 1):
                                indices.append([base_index + i, base_index + i + 1, base_index + num_points + i + 1])
                                indices.append(
                                    [base_index + i, base_index + num_points + i + 1, base_index + num_points + i])
                            indices.append([base_index + num_points - 1, base_index, base_index + num_points])
                            indices.append(
                                [base_index + num_points - 1, base_index + num_points, base_index + 2 * num_points - 1])

                        # 处理柱体的上下盖
                        for i in range(1, num_points - 1):
                            indices.append([base_index, base_index + i, base_index + i + 1])  # 底盖
                            indices.append([base_index + num_points, base_index + num_points + i + 1,
                                            base_index + num_points + i])  # 顶盖

        # 转换为numpy数组
        vertices = np.array(vertices, dtype="float32")
        indices = np.array(indices, dtype="uint32")
        colors = np.array(colors, dtype="float32")
        print(vertices.shape, indices.shape, colors.shape)
        # 调用gltf_from_array生成GLB文件
        self.gltf_from_array(vertices, indices, colors, output_path)

    def decode_gltf(self):
        gltf = self.gltf
        binary_blob = gltf.binary_blob()

        # 解码点数据
        points_accessor = gltf.accessors[gltf.meshes[0].primitives[0].attributes.POSITION]
        points_buffer_view = gltf.bufferViews[points_accessor.bufferView]

        # 打印调试信息
        print(f"Points Buffer View ByteOffset: {points_buffer_view.byteOffset}")
        print(f"Points Buffer View ByteLength: {points_buffer_view.byteLength}")
        print(f"Points Accessor ByteOffset: {points_accessor.byteOffset}")
        print(f"Points Accessor Count: {points_accessor.count}")

        try:
            points = np.frombuffer(
                binary_blob[
                points_buffer_view.byteOffset
                + points_accessor.byteOffset: points_buffer_view.byteOffset
                                              + points_buffer_view.byteLength
                ],
                dtype="float32",
                count=points_accessor.count * 3,  # 3D坐标 (x, y, z)
            ).reshape((-1, 3))
        except ValueError as e:
            print(f"Error decoding points: {e}")
            return None, None, None

        # 解码颜色数据
        colors_accessor = gltf.accessors[gltf.meshes[0].primitives[0].attributes.COLOR_0]
        colors_buffer_view = gltf.bufferViews[colors_accessor.bufferView]

        # 打印调试信息
        print(f"Colors Buffer View ByteOffset: {colors_buffer_view.byteOffset}")
        print(f"Colors Buffer View ByteLength: {colors_buffer_view.byteLength}")
        print(f"Colors Accessor ByteOffset: {colors_accessor.byteOffset}")
        print(f"Colors Accessor Count: {colors_accessor.count}")

        try:
            colors = np.frombuffer(
                binary_blob[
                colors_buffer_view.byteOffset
                + colors_accessor.byteOffset: colors_buffer_view.byteOffset
                                              + colors_buffer_view.byteLength
                ],
                dtype="float32",
                count=colors_accessor.count * 4,  # RGBA颜色
            ).reshape((-1, 4))
        except ValueError as e:
            print(f"Error decoding colors: {e}")
            return None, None, None

        # 如果存在三角形索引，则解码三角形索引
        triangles = None
        if 'indices' in dir(gltf.meshes[0].primitives[0]):
            triangles_accessor = gltf.accessors[gltf.meshes[0].primitives[0].indices]
            triangles_buffer_view = gltf.bufferViews[triangles_accessor.bufferView]

            # 打印调试信息
            print(f"Triangles Buffer View ByteOffset: {triangles_buffer_view.byteOffset}")
            print(f"Triangles Buffer View ByteLength: {triangles_buffer_view.byteLength}")
            print(f"Triangles Accessor ByteOffset: {triangles_accessor.byteOffset}")
            print(f"Triangles Accessor Count: {triangles_accessor.count}")

            try:
                triangles = np.frombuffer(
                    binary_blob[
                    triangles_buffer_view.byteOffset
                    + triangles_accessor.byteOffset: triangles_buffer_view.byteOffset
                                                     + triangles_buffer_view.byteLength
                    ],
                    dtype="uint32",
                    count=triangles_accessor.count,
                ).reshape((-1, 3))
            except  ValueError as e:
                print(f"Error decoding triangles: {e}")
                return points, None, colors

        return points, triangles, colors

if __name__ == "__main__":
    gp = GLTFProducer()
    gp.gltf_from_shapefile("C:/Users/Administrator/Desktop/noisemap/Baoding_Building13.shp",
                                           "C:/Users/Administrator/Desktop/testfile/threedimension_building_height.glb")
    #gp.gltf_from_array(vertices1, indices1, colors1,
                        #"C:/Users/Administrator/Desktop/testfile/test_cube.glb")
    # 解码 GLTF 数据
    #points, triangles, colors = gp.decode_gltf()
# print("Points:\n", points)
# print("Triangles:\n", triangles)
 # print("Colors:\n", colors)