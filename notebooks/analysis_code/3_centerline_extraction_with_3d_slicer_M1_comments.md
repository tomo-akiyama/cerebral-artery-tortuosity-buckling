# Comments from Jupyter Notebook: 3 centerline extraction with 3d slicer M1.ipynb

## Cell 1

# Centerline extraction

## Cell 11

# https://github.com/bhavika-g/vmtk/blob/main/endpoint.py

## Cell 19

pl = pv.Plotter()
pl.add_mesh(centerlines, color='lime', opacity=1, line_width=5)
pl.add_mesh(vessel_mesh, color='lightblue', opacity=0.5)
# pl.add_points(sources_on_mesh, color="magenta", opacity=0.8, point_size=15, render_points_as_spheres=True,ambient=1.0)
pl.add_points(targets_on_mesh, color="yellow", opacity=0.8, point_size=15, render_points_as_spheres=True,ambient=1.0)
pl.add_mesh(sphere1, color='red', opacity=0.5)
# pl.add_points(sources_on_mesh[0], color="red", opacity=0.8, point_size=15, render_points_as_spheres=True,ambient=1.0)
pl.add_points(m1p1, color="red", opacity=0.8, point_size=20, render_points_as_spheres=True,ambient=1.0)
pl.add_title("vmtk CL")
pl.camera_position = 'zx'
pl.camera.zoom(1)
pl.show()

## Cell 21

# Red = 右?
# Blue = 左?

# 上 が右、下が左
