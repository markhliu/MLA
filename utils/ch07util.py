from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import pickle
import os


def horse_pic(p_horse,ws,bs,epochs,pic):   	
    # Generate 26 pictures
    for i in range(26):
        fig = plt.figure(figsize=(14,10), dpi=100)
        ax = fig.add_subplot(111)  
        # Draw the four input neurons
        c=plt.Circle((-3,4),radius=0.8,color='white',ec="m")
        ax.add_artist(c)
        c=plt.Circle((-3,-4),radius=0.8,color='white',ec="m")
        ax.add_artist(c)
        c=plt.Circle((-3,2),radius=0.8,color='white',ec="m")
        ax.add_artist(c)
        c=plt.Circle((-3,-2),radius=0.8,color='white',ec="m")
        ax.add_artist(c)
        # Draw the output neuron
        c=plt.Circle((1,0),radius=0.8,color='white',ec="m")
        ax.add_artist(c)
        # Draw connections between neurons
        xys=[[(0.2,0), (-2.2,2)],
               [(0.2,0), (-2.2,-2)],
               [(-3.8,-2),(-5.5,-2)],
               [(-3.8,2),(-5.5,2)],
               [(-3.8,-4),(-5.5,-4)],
               [(-3.8,4),(-5.5,4)],
               [(0.2,0), (-2.2,4)],
               [(0.2,0), (-2.2,-4)],
               [(4.5,0),(4,0)]]
        for xy in xys:
            ax.annotate("",xy=xy[0],xytext=xy[1],
            arrowprops=dict(arrowstyle='->',color='g',linewidth=2))       
        # Put explanation texts on the graph
        zs=[[-5.25,4.1,"bias",20,"k",0],
            [-5.25,-3.9,"pixel",20,"k",0],
            [-3.2,3.8,"1",30,"k",0],
            [-3.6,-4.2,r"$X_{1024}$",30,"k",0],
            [-3.3,0.2,r"$\vdots$",30,"k",0],
            [-3.3,-0.4,r"$\vdots$",30,"k",0],
            [-5.25,2.1,"pixel",20,"k",0],
            [-5.25,-1.9,"pixel",20,"k",0],
            [-3.4,1.8,r"$X_{1}$",30,"k",0],
            [-3.6,-2.2,r"$X_{1023}$",30,"k",0],
            [0.5,-0.1,"wX+b",20,"k",0],
            [4.6,-0.1,f"Y={p_horse[i]:.4f}",20,"r",0],
            [2.05,-0.1,r"$\frac{1}{1+e^{-(wX+b)}}$",25,"k",0],
            [-1.9,-2,f"w={ws[i][-2]:.3f}",20,"r",39],
            [-1.7,2.0,f"b={bs[i]:.3f}",20,"r",-57],
            [-1.7,-3.5,f"w={ws[i][-1]:.3f}",20,"r",57],
            [-2.1,0.9,f"w={ws[i][0]:.3f}",20,"r",-39]]
        for z in zs:
            plt.text(z[0],z[1],z[2],fontsize=z[3],
                     color=z[4],rotation=z[5])     
        # put epoch number and losses up
        plt.text(1,4,f"epoch {epochs[i]}",fontsize=40,color="g")   
        # add text to explain 
        txt=ax.annotate('Sigmoid\nActivation',xy=(3,-1), 
            xytext=(0.55,0.2),textcoords='axes fraction', 
            bbox=dict(boxstyle="round",fc="0.9"), 
            arrowprops=dict(arrowstyle='->',color='g',linewidth=1), 
            fontsize=20)
        txt=ax.annotate('probability of\n being a horse',
            xy=(4.8,0.2),xytext=(0.7,0.65),
            textcoords='axes fraction', 
            bbox=dict(boxstyle="round",fc="0.9"), 
            arrowprops=dict(arrowstyle='->',color='g',linewidth=1), 
            fontsize=20)
        # Add a rectangle to plot
        ax.add_patch(Rectangle((2,-1),2,2,edgecolor='k',alpha=0.1))  
        ax.set_xlim(-7,7)
        ax.set_ylim(-5,5)
        plt.axis("off")
        # Add horse picture
        newax=fig.add_axes([0.1,0.4,0.2,0.2])
        newax.imshow(pic)
        newax.axis('off')
        plt.savefig(f"files/ch07/p_horse{i}.png")
        plt.close(fig)
    
def deer_pic(p_deer,ws,bs,epochs,pic):
    # Generate 26 pictures
    for i in range(26):
        fig = plt.figure(figsize=(14,10), dpi=100)
        ax = fig.add_subplot(111)  
        # Draw the four input neurons
        c=plt.Circle((-3,4),radius=0.8,color='white',ec="m")
        ax.add_artist(c)
        c=plt.Circle((-3,-4),radius=0.8,color='white',ec="m")
        ax.add_artist(c)
        c=plt.Circle((-3,2),radius=0.8,color='white',ec="m")
        ax.add_artist(c)
        c=plt.Circle((-3,-2),radius=0.8,color='white',ec="m")
        ax.add_artist(c)
        # Draw the output neuron
        c=plt.Circle((1,0),radius=0.8,color='white',ec="m")
        ax.add_artist(c)
        # Draw connections between neurons
        xys=[[(0.2,0), (-2.2,2)],
               [(0.2,0), (-2.2,-2)],
               [(-3.8,-2),(-5.5,-2)],
               [(-3.8,2),(-5.5,2)],
               [(-3.8,-4),(-5.5,-4)],
               [(-3.8,4),(-5.5,4)],
               [(0.2,0), (-2.2,4)],
               [(0.2,0), (-2.2,-4)],
               [(4.5,0),(4,0)]]
        for xy in xys:
            ax.annotate("",xy=xy[0],xytext=xy[1],
            arrowprops=dict(arrowstyle='->',color='g',linewidth=2))       
        # Put explanation texts on the graph
        zs=[[-5.25,4.1,"bias",20,"k",0],
            [-5.25,-3.9,"pixel",20,"k",0],
            [-3.2,3.8,"1",30,"k",0],
            [-3.6,-4.2,r"$X_{1024}$",30,"k",0],
            [-3.3,0.2,r"$\vdots$",30,"k",0],
            [-3.3,-0.4,r"$\vdots$",30,"k",0],
            [-5.25,2.1,"pixel",20,"k",0],
            [-5.25,-1.9,"pixel",20,"k",0],
            [-3.4,1.8,r"$X_{1}$",30,"k",0],
            [-3.6,-2.2,r"$X_{1023}$",30,"k",0],
            [0.5,-0.1,"wX+b",20,"k",0],
            [4.6,-0.1,f"Y={p_deer[i]:.4f}",20,"r",0],
            [4.6,-1,f"1-Y={1-p_deer[i]:.4f}",20,"r",0], 
            [2.05,-0.1,r"$\frac{1}{1+e^{-(wX+b)}}$",25,"k",0],
            [-1.9,-2,f"w={ws[i][-2]:.3f}",20,"r",39],
            [-1.7,2.0,f"b={bs[i]:.3f}",20,"r",-57],
            [-1.7,-3.5,f"w={ws[i][-1]:.3f}",20,"r",57],
            [-2.1,0.9,f"w={ws[i][0]:.3f}",20,"r",-39]]
        for z in zs:
            plt.text(z[0],z[1],z[2],fontsize=z[3],
                     color=z[4],rotation=z[5])     
        # put epoch number and losses up
        plt.text(1,4,f"epoch {epochs[i]}",fontsize=40,color="g")   
        # add text to explain 
        txt=ax.annotate('Sigmoid\nActivation',xy=(3,1), 
            xytext=(0.7,0.7),textcoords='axes fraction', 
            bbox=dict(boxstyle="round",fc="0.9"), 
            arrowprops=dict(arrowstyle='->',color='g',linewidth=1), 
            fontsize=20) 
        txt=ax.annotate('probability of\n being a deer', 
            xy=(5.5,-1),xytext=(0.7,0.15), 
            textcoords='axes fraction', 
            bbox=dict(boxstyle="round",fc="0.9"), 
            arrowprops=dict(arrowstyle='->',color='g',linewidth=1), 
            fontsize=20)
        # add rectangle to plot
        ax.add_patch(Rectangle((2,-1),2,2,edgecolor='k',alpha=0.1))  
        ax.set_xlim(-7,7)
        ax.set_ylim(-5,5)
        plt.axis("off")
        # Add deer picture
        newax=fig.add_axes([0.1,0.4,0.2,0.2])
        newax.imshow(pic)
        newax.axis('off')
        plt.savefig(f"files/ch07/p_deer{i}.png")
        plt.close(fig)