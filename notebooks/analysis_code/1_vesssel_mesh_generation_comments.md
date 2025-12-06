# Comments from Jupyter Notebook: 1 vesssel mesh generation.ipynb

## Cell 1

# Segmentation and mesh generation

## Cell 11

>> face数とdensity (elements/mm^3)を計算  
2022Sci Rep Kashyap.pdfではradius adaptive meshing methodを用いて  
The average number of elements was 1.85 × 106 ± 0.25 × 106 (mean ± std. dev.) with an average density of 5790 ± 4373 elements/mm3 (std. dev. 4373)

本当はA radius adaptive meshing method was used, that generates denser meshes in narrower regions, hence long or narrow vessels would have the highest number of elements.が理想。
