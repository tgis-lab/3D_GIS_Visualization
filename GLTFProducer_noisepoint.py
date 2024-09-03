import geopandas as gpd
import numpy as np
import shapefile
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pygltflib

class GLTFProducer(object):

    def __init__(self):
        return

    def gltf_from_array(self, points, triangles, colors, line_points, line_indices, line_colors, output_path):
        if points.size == 0:
            print("Error: Points array is empty. Skipping GLB generation.")
            return

        # 如果三角形数组为空，则不处理三角形
        if triangles.size == 0:
            print("No triangles provided. Generating GLB with points and colors only.")
            triangles_binary_blob = b''
        else:
            triangles_binary_blob = triangles.flatten().tobytes()

        points_binary_blob = points.tobytes()
        colors_binary_blob = colors.flatten().tobytes()

        line_indices_binary_blob = line_indices.flatten().tobytes()
        line_points_binary_blob = line_points.tobytes()
        line_colors_binary_blob = line_colors.flatten().tobytes()
        gltf = pygltflib.GLTF2(
            # scenes
            scene=0,
            scenes=[pygltflib.Scene(nodes=[0])],
            nodes=[pygltflib.Node(mesh=0)],
            meshes=[
                pygltflib.Mesh(
                    primitives=[
                        pygltflib.Primitive(
                            attributes=pygltflib.Attributes(
                                POSITION=0,  # 指向位置的 accessor
                                COLOR_0=1  # 指向颜色的 accessor
                            ),
                            mode=0  # GL_LINES: 只渲染线段而不是三角形
                        )
                    ]
                )
            ],
            # accessors
            accessors=[
                pygltflib.Accessor(
                    bufferView=0,
                    componentType=pygltflib.FLOAT,
                    count=len(points),
                    type=pygltflib.VEC3,
                    max=points.max(axis=0).tolist(),
                    min=points.min(axis=0).tolist(),
                ),
                pygltflib.Accessor(
                    bufferView=1,
                    componentType=pygltflib.FLOAT,
                    count=len(colors),
                    type=pygltflib.VEC4,
                    max=colors.max(axis=0).tolist(),
                    min=colors.min(axis=0).tolist(),
                    normalized=False,
                ),
            ],
            # bufferViews
            bufferViews=[
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=0,
                    byteLength=len(points_binary_blob),
                    target=pygltflib.ARRAY_BUFFER,
                ),
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=len(points_binary_blob),
                    byteLength=len(colors_binary_blob),
                    target=pygltflib.ARRAY_BUFFER,
                ),
            ],
            # buffers
            buffers=[
                pygltflib.Buffer(
                    byteLength=len(points_binary_blob) + len(colors_binary_blob),
                )
            ],
        )

        # 将顶点数据和颜色数据打包成二进制 blob
        points_binary_blob = points.tobytes()
        colors_binary_blob = colors.tobytes()

        # 设置二进制数据到 GLTF 文件
        gltf.set_binary_blob(points_binary_blob + colors_binary_blob)

        # 保存 GLTF 文件
        gltf.save(output_path)

    def decode_gltf(self):
        gltf = self.gltf
        binary_blob = gltf.binary_blob()

        # 解码点数据
        points_accessor = gltf.accessors[gltf.meshes[0].primitives[0].attributes.POSITION]
        points_buffer_view = gltf.bufferViews[points_accessor.bufferView]
        points = np.frombuffer(
            binary_blob[
            points_buffer_view.byteOffset
            + points_accessor.byteOffset: points_buffer_view.byteOffset
                                          + points_buffer_view.byteLength
            ],
            dtype="float32",
            count=points_accessor.count * 3,
        ).reshape((-1, 3))

        # 解码颜色数据
        colors_accessor = gltf.accessors[gltf.meshes[0].primitives[0].attributes.COLOR_0]
        colors_buffer_view = gltf.bufferViews[colors_accessor.bufferView]
        colors = np.frombuffer(
            binary_blob[
            colors_buffer_view.byteOffset
            + colors_accessor.byteOffset: colors_buffer_view.byteOffset
                                          + colors_buffer_view.byteLength
            ],
            dtype="float32",
            count=colors_accessor.count * 4,
        ).reshape((-1, 4))

        # 如果存在三角形索引，则解码三角形索引
        triangles = None
        if 'indices' in dir(gltf.meshes[0].primitives[0]):
            triangles_accessor = gltf.accessors[gltf.meshes[0].primitives[0].indices]
            triangles_buffer_view = gltf.bufferViews[triangles_accessor.bufferView]
            triangles = np.frombuffer(
                binary_blob[
                triangles_buffer_view.byteOffset
                + triangles_accessor.byteOffset: triangles_buffer_view.byteOffset
                                                 + triangles_buffer_view.byteLength
                ],
                dtype="uint32",
                count=triangles_accessor.count,
            ).reshape((-1, 3))

        return (points, triangles, colors)

    def get_heatmap_color(self, value, min_value, max_value):
        norm = mcolors.Normalize(vmin=min_value, vmax=max_value)
        cmap = plt.get_cmap('hot')  # 选择热力图色图
        return cmap(norm(value))[:3]  # 获取 RGB 颜色，不包括 Alpha

    def gltf_from_shapefile(self, shapefile_path, attribute_name, output_path):
        # 使用 geopandas 读取 SHP 文件
        gdf = gpd.read_file(shapefile_path)
        #print(gdf)

        # 提取几何数据和属性值
        points = []
        triangles = []
        colors = []

        min_attr = gdf[attribute_name].min()
        max_attr = gdf[attribute_name].max()

        for i, (geom, attr_value) in enumerate(zip(gdf.geometry, gdf[attribute_name])):
            if geom.geom_type == "Point":
                # 提取点坐标
                point_coords = np.array(geom.coords)
                points.append(point_coords[0])
                # 将属性值映射为颜色
                color = self.get_heatmap_color(attr_value, min_attr, max_attr) + (1,)
                colors.append(color)
                #print(f"Point {i}: {point_coords[0]}")
                #print(f"Color for this point: {color}")

            elif geom.geom_type == "Polygon" or geom.geom_type == "MultiPolygon":
                for poly in [geom] if geom.geom_type == "Polygon" else geom.geoms:
                    exterior_coords = np.array(poly.exterior.coords)
                    #print(f"Exterior coords: {exterior_coords}")
                    if len(exterior_coords) > 0:
                        start_idx = len(points)
                        points.extend(exterior_coords)
                        # 生成三角形索引
                        num_points = len(exterior_coords)
                        if num_points > 3:
                            for i in range(1, num_points - 1):
                                triangles.append([start_idx, start_idx + i, start_idx + i + 1])

                        #print(f"Triangles: {triangles[-1] if triangles else 'None'}")

                        # 将属性值映射为颜色
                        color = self.get_heatmap_color(attr_value, min_attr, max_attr) + (1,)
                        colors.extend([color] * (num_points - 2))
                        #print(f"Color for this polygon: {color}")

        # 转换为 numpy 数组
        points = np.array(points, dtype="float32")
        triangles = np.array(triangles, dtype="uint32") if triangles else np.empty((0, 3), dtype="uint32")
        colors = np.array(colors, dtype="float32")

        # 检查输出是否正确
       # print(f"Total points: {len(points)}, Total colors: {len(colors)}")

        # 处理线条数据（可选）
        line_points = np.empty((0, 3), dtype="float32")
        line_indices = np.empty((0, 2), dtype="uint32")
        line_colors = np.empty((0, 4), dtype="float32")

        # 调用 gltf_from_array 生成 GLB 文件
        self.gltf_from_array(points, triangles, colors, line_points, line_indices, line_colors, output_path)





if __name__ == "__main__":
    gp = GLTFProducer()
    gp.gltf_from_shapefile("your_file_patch.shp", "attribute_name", "your_output_patch.glb")
