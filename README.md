### From author:


**If you have any questions - just write an issue!**

Unfortuantely I had no time to finish this project and make nice CLI, but it works! You can edit `main.py` and put your model's name there.  

# VoxExporter

__VoxExporter__ is a library that has a lot of functionality that helps to export voxel models 
from .vox format into .obj format. 

The key feature of **VoxExporter** is that it creates custom uv-unwrapping of 
model texture, and do not use default 256px palette-texture. With custom uv-s it is possible to easily bake lightmaps
in Unity3D, Blender etc.

## Example of unwrapped texture
![texture](/images/screen3.png?raw=true)

## Example of work
![shaded](/images/screen1.png?raw=true)
![wireframe](/images/screen2.png?raw=true)
