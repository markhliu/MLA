


import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

inputs=["cart\nposition","cart\nvelocity",
          "pole\nangle","pole\nvelocity"]
hs=[3.1,1.3,-1.4,-3.2]

def save_graph(step,frames,states,actions,predictions):
    fig = plt.figure(figsize=(24,10), dpi=100)
    ax = fig.add_subplot(111) 
    ax.set_xlim(-10, 10)
    ax.set_ylim(-5,5)
    #plt.grid()
    plt.axis("off")
    ax.text(-4.5,4.25,
    f"The Cart Pole Game, Time Step {step+1}",fontsize=40)
    # Add deep-Q network
    ax.add_patch(Rectangle((3, -1), 2.5, 2,
                 facecolor = 'b',alpha=0.5)) 
    ax.text(3.2,-0.75,"Trained \nDeep Q\nNetwork",fontsize=30)
    # Add output boxes
    ax.add_patch(Rectangle((6.5, 1.5), 3,2.5,
                 facecolor = 'g',alpha=0.25))
    ax.add_patch(Rectangle((6.5, -4), 3,2.5,
                 facecolor = 'g',alpha=0.25)) 
    ax.text(6.7,2,
    f"Move Left \nQ-Value =\n{predictions[step][0]:.5f}",
    fontsize=30)
    ax.text(6.7,-3.5,
    f"Move Right \nQ-Value =\n{predictions[step][1]:.5f}",
    fontsize=30)
    ax.annotate("",xy=(6.5,1.5),xytext=(5.5,0),
        arrowprops=dict(arrowstyle='->',color='g',linewidth=2))
    ax.annotate("",xy=(6.5,-1.5),xytext=(5.5,0),
        arrowprops=dict(arrowstyle='->',color='g',linewidth=2))
    # highlight the best action
    ax.add_patch(Rectangle((6.5, 1.5-5.5*actions[step]),
                           3,2.5,facecolor='r'))   
    # add rectangle to plot
    for i in range(4):
        ax.add_patch(Rectangle((0,-0.6+hs[i]), 2, 1.3,
                     facecolor = 'b',alpha=0.1)) 
        plt.text(0.2,hs[i]-0.5,
         f"{inputs[i]}\n{states[step][i]:.4f}",fontsize=20)   
        ax.annotate("",xy=(3,0),xytext=(2,hs[i]),
        arrowprops=dict(arrowstyle='->',color='g',linewidth=2))   
    # Add new picture
    newax = fig.add_axes([0.02, 0.2, 0.6, 0.6])
    newax.imshow(frames[step])
    newax.axis('off') 
    plt.savefig(f"files/ch17/cartpole_DeepQ{step+1}.png")




def memory_graphs(frames,states,actions,predictions):
    graphs = []
    for step in range(0,200):
        fig = plt.figure(figsize=(24,10), dpi=40)
        ax = fig.add_subplot(111) 
        ax.set_xlim(-10, 10)
        ax.set_ylim(-5,5)
        #plt.grid()
        plt.axis("off")
        ax.text(-4.5,4.25,f"The Cart Pole Game, Time Step {step+1}",fontsize=40)
    
        # Add deep Q network
        ax.add_patch(Rectangle((3, -1), 2.5, 2,
                     facecolor = 'b',alpha=0.5)) 
        ax.text(3.2,-0.75,"Trained \nDeep Q\nNetwork",fontsize=30)
    
        # Add output boxes
        ax.add_patch(Rectangle((6.5, 1.5), 3,2.5,
                     facecolor = 'g',alpha=0.25))
        ax.add_patch(Rectangle((6.5, -4), 3,2.5,
                     facecolor = 'g',alpha=0.25)) 
        ax.text(6.7,2,f"Move Left \nQ-Value =\n{predictions[step][0]:.5f}",fontsize=30)
        ax.text(6.7,-3.5,f"Move Right \nQ-Value =\n{predictions[step][1]:.5f}",fontsize=30)
        ax.annotate("",xy=(6.5,1.5),xytext=(5.5,0),
            arrowprops=dict(arrowstyle = '->', color = 'g', linewidth = 2))
        ax.annotate("",xy=(6.5,-1.5),xytext=(5.5,0),
            arrowprops=dict(arrowstyle = '->', color = 'g', linewidth = 2))
    
        # highlight the best action
        ax.add_patch(Rectangle((6.5, 1.5-5.5*actions[step]), 3,2.5,facecolor='r'))   
    
        # add rectangle to plot
        for i in range(4):
            ax.add_patch(Rectangle((0,-0.6+hs[i]), 2, 1.3,
                         facecolor = 'b',alpha=0.1)) 
            plt.text(0.2,hs[i]-0.5,f"{inputs[i]}\n{states[step][i]:.4f}",fontsize=20)   
            ax.annotate("",xy=(3,0),xytext=(2,hs[i]),
            arrowprops=dict(arrowstyle = '->', color = 'g', linewidth = 2))   
    
        # Add new picture
        newax = fig.add_axes([0.02, 0.2, 0.6, 0.6])
        newax.imshow(frames[step])
        newax.axis('off') 
    
        fig.canvas.draw()
        # Now we can save it to a numpy array.
        graph = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        graph = graph.reshape(fig.canvas.get_width_height()[::-1]+ (3,))
        graphs.append(graph)
        plt.close(fig)
    
    return graphs    






