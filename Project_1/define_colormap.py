import matplotlib.colors

def DefineColormap(n):
    """
    function for deciding costum colormaps
    param n: name of colormap, string
    """

    if n == 'afternoon':
        colormap = cm_afternoon = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#8C0004","#C8000A","#E8A735","#E2C499"])
    elif n == 'timeless':
        colormap = cm_timeless  = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#16253D","#002C54","#EFB509","#CD7213"])
    elif n == 'arctic':
        colormap = cm_arctic    = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#006C84","#6EB5C0","#E2E8E4","#FFCCBB"])
    elif n == 'sunkissed':
        colormap = cm_sunkissed = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#D24136","#EB8A3E","#EBB582","#785A46"])
    elif n == 'berry':
        colormap = cm_berry     = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#D0E1F9","#4D648D","#283655","#1E1F26"])
    elif n == 'sunset':
        colormap = cm_sunset    = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#363237","#2D4262","#73605B","#D09683"])
    elif n == 'watery':
        colormap = cm_watery    = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#021C1E","#004445","#2C7873","#6FB98F"])
    else:
        colormap = cm_afternoon

    return colormap
