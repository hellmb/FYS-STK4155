import matplotlib.colors

def DefineColormap(n):
    """
    function for deciding costum colormaps
    param n: name of colormap, string
    """

    if n == 'afternoon':
        colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#8C0004","#C8000A","#E8A735","#E2C499"])
    elif n == 'timeless':
        colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#16253D","#002C54","#EFB509","#CD7213"])
    elif n == 'arctic':
        colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#006C84","#6EB5C0","#E2E8E4","#FFCCBB"])
    elif n == 'sunkissed':
        colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#D24136","#EB8A3E","#EBB582","#785A46"])
    elif n == 'berry':
        colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#D0E1F9","#4D648D","#283655","#1E1F26"])
    elif n == 'sunset':
        colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#363237","#2D4262","#73605B","#D09683"])
    elif n == 'watery':
        colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#021C1E","#004445","#2C7873","#6FB98F"])
    elif n == 'bright':
        colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#061283","#FD3C3C","#FFB74C","#138D90"])
    elif n == 'school':
        colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#81715E","#FAAE3D","#E38533","#E4535E"])
    elif n == 'golden':
        colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#323030","#CDBEA7","#C29545","#882426"])
    elif n == 'misty':
        colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#04202C","#2C493F","#5B7065","#C9D1C8"])
    elif n == 'coolblues':
        colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#003B46","#07575B","#66A5AD","#C4DFE6"])
    elif n == 'candy':
        colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#AD1457","#D81B60","#FFA000","#FDD835","#FFEE58"])
    else:
        colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#8C0004","#C8000A","#E8A735","#E2C499"])

    return colormap
