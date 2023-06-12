import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

# Define sqr() function to draw a square
def sqr(ax,x,y,size=1,linestyle="-",color="gray",linewidth=1):
    ax.plot([x,x+size],[y,y],linestyle=linestyle,
        color=color,linewidth=linewidth)
    ax.plot([x,x],[y+size,y],linestyle=linestyle,
        color=color,linewidth=linewidth)
    ax.plot([x+size,x+size],[y,y+size],linestyle=linestyle,
        color=color,linewidth=linewidth)
    ax.plot([x,x+size],[y+size,y+size],linestyle=linestyle,
        color=color,linewidth=linewidth)


def stride_steps(h,v,image,output):
    fig = plt.figure(figsize=(12,8), dpi=200)
    ax = fig.add_subplot(111)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-4, 4)
    plt.axis("off")
    # Draw the 6 by 6 image 
    for i in range(6):
        for j in range(6):
            sqr(ax,-6+i,-2+j,linewidth=2)
            ax.text(i-5.6, 3.4-j, image[j,i],size=16, color="gray")
            
    # add text to explain 
    ax.annotate(
        'Apply a 2 by 2 filter on a 6 by 6 image\nwithout zero-padding\nstride=2', 
        xy=(0,0),
        xytext = (0.02,0.05), 
        textcoords = 'axes fraction', 
        bbox = dict(boxstyle="round", fc="0.9"), 
        fontsize = 20)

    # Draw a filter on the side
    ax.text(2.8,2.8, "filter=", size=20, color="b")
    for i in range(2):
        for j in range(2):
            sqr(ax,4+i,2+j, color="b", linewidth=3)
            if i==j:
                ax.text(4.4+j, 2.4+i, "0", size=16, color="b")
            else:
                ax.text(4.4+j, 2.4+i, "1", size=16, color="b")
    # Apply filter        
    for i in range(2):
        for j in range(2):
            sqr(ax,-6+i+2*h, 2+j-2*v, color="b", linewidth=3)      
    # Draw the output matrix
    for i in range(3):
        for j in range(3):
            sqr(ax,2+i,j-3, linewidth=2, color="k")
            ax.text(2.4+i, -0.6-j, f"{output[j,i]}", size=16, color="gray")  
    plt.savefig(f"files/ch08/stride{h}{v}step1.png")
    # add text to explain 
    ax.annotate('', xy = (2.3+h,-0.3-v), xytext = (-8+i+2*h, j-2*v), 
        arrowprops=dict(arrowstyle = '->', color = 'g', linewidth = 1))        
    ax.annotate('', xy = (2.3+h,-0.3-v), xytext = (-6+i+2*h, 2+j-2*v), 
        arrowprops=dict(arrowstyle = '->', color = 'g', linewidth = 1)) 
    plt.savefig(f"files/ch08/stride{h}{v}step2.png")
    
    ax.text(2.4+h, -0.6-v, f"{output[v,h]}", size=16, color="b")
    plt.savefig(f"files/ch08/stride{h}{v}step3.png")
    plt.close(fig)


