from matplotlib.patches import Rectangle
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt

# Names of the labels
names = ['plane', 'car', 'bird', 'cat', 
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']    

def p_truck(truck):
    # Generate 26 pictures
    for stage in range(26):
        # reload the deep neural network
        reload = tf.keras.models.load_model(f"files/ch09/multi_epoch{stage*5}.h5")
        # the predictions from the DNN
        ps = reload.predict(truck.reshape(1,32,32,3), verbose=0)[0]
        fig = plt.figure(figsize=(14,10), dpi=100)
        ax = fig.add_subplot(111)  
        # Draw the four input neurons
        circle = plt.Circle((-3,4),radius=0.8, color='white',ec="m")
        ax.add_artist(circle)
        circle = plt.Circle((-3,-4),radius=0.8, color='white',ec="m")
        ax.add_artist(circle)
        circle = plt.Circle((-3,2),radius=0.8, color='white',ec="m")
        ax.add_artist(circle)
        circle = plt.Circle((-3,-2),radius=0.8, color='white',ec="m")
        ax.add_artist(circle)
    
        # Draw connections between neurons
        xys = [[(0.,0), (-2.2,2)],
               [(0.,0), (-2.2,-2)],
               [(-3.8,-2),(-5.5,-2)],
               [(-3.8,2),(-5.5,2)],
               [(-3.8,-4),(-5.5,-4)],
               [(-3.8,4),(-5.5,4)],
               [(0.,0), (-2.2,4)],
               [(0.,0), (-2.2,-4)],
               [(4,-4.1),(2,0)],
               [(4,-3.2),(2,0)],           
               [(4,-2.3),(2,0)],           
               [(4,-1.4),(2,0)],
               [(4,-0.5),(2,0)],
               [(4,0.4),(2,0)],
               [(4,1.3),(2,0)],
               [(4,2.2),(2,0)],
               [(4,3.1),(2,0)],
               [(4,4),(2,0)]]
        for xy in xys:
            ax.annotate("",xy=xy[0],xytext=xy[1],
            arrowprops=dict(arrowstyle = '->', color = 'g', linewidth = 2))  
        # Put explanation texts on the graph
        zs = [[-5.25, 4.1, "bias", 20, "k", 0],
                [-5.25, -3.9, "pixel", 20, "k", 0],
                [-3.2, 3.8, "1", 30, "k", 0],
                [-3.6, -4.2, r"$X_{3072}$", 30, "k", 0],
                [-3.3, 0.2, r"$\vdots$", 30, "k", 0],
                [-3.3, -0.4, r"$\vdots$", 30, "k", 0],
                [-5.25, 2.1, "pixel", 20, "k", 0],
                [-5.25, -1.9, "pixel", 20, "k", 0],        
                [-3.4, 1.8, r"$X_{1}$", 30, "k", 0],
                [-3.6, -2.2, r"$X_{3071}$", 30, "k", 0],
                [0.2, -0.5, "Deep\nNeural\nNetwork", 20, "k", 0]]
        for z in zs:
            plt.text(z[0], z[1], z[2], fontsize=z[3], color=z[4], rotation=z[5])  
        for i in range(10):
            plt.text(4.1, 4-0.9*i, f"p({names[i]})={ps[i]:.4f}", fontsize=15, color="r")  
    
        # put epoch number up
        plt.text(-1, 4.3, f"epoch {stage*5}", fontsize=40, color="g")   
        # add text to explain 
        txt = ax.annotate('Softmax\nActivation', 
            xy = (1,-1), 
            xytext = (0.5,0.2), 
            textcoords = 'axes fraction', 
            bbox = dict(boxstyle="round", fc="0.9"), 
            arrowprops=dict(arrowstyle = '->', color = 'g', linewidth = 1), 
            fontsize = 20)
        txt = ax.annotate('convolutional\nlayers', 
            xy = (1,1), 
            xytext = (0.5,0.75), 
            textcoords = 'axes fraction', 
            bbox = dict(boxstyle="round", fc="0.9"), 
            arrowprops=dict(arrowstyle = '->', color = 'g', linewidth = 1), 
            fontsize = 20)
        # add rectangle to plot
        ax.add_patch(Rectangle((0,-1), 2, 2,
                     edgecolor = 'k',alpha=0.1)) 
        ax.add_patch(Rectangle((4,-4.3), 2.6, 0.6,
                     facecolor = 'g',alpha=0.25))  
        ax.set_xlim(-7, 7)
        ax.set_ylim(-5, 5)
        plt.axis('off')
        # Add truck picture
        newax = fig.add_axes([0.1, 0.39, 0.23, 0.23])
        newax.imshow(truck)
        newax.axis('off')
        plt.savefig(f"files/ch09/p_truck{stage}.png")
        plt.close(fig)
        
        
        
def p_frog(frog):
    # Generate 26 pictures
    for stage in range(26):
        # reload the deep neural network
        reload = tf.keras.models.load_model(f"files/ch09/multi_epoch{stage*5}.h5")
        # the predictions from the DNN
        ps = reload.predict(frog.reshape(1,32,32,3))[0]
        fig = plt.figure(figsize=(14,10), dpi=100)
        ax = fig.add_subplot(111)  
        # Draw the four input neurons
        circle = plt.Circle((-3,4),radius=0.8, color='white',ec="m")
        ax.add_artist(circle)
        circle = plt.Circle((-3,-4),radius=0.8, color='white',ec="m")
        ax.add_artist(circle)
        circle = plt.Circle((-3,2),radius=0.8, color='white',ec="m")
        ax.add_artist(circle)
        circle = plt.Circle((-3,-2),radius=0.8, color='white',ec="m")
        ax.add_artist(circle)
    
        # Draw connections between neurons
        xys = [[(0.,0), (-2.2,2)],
               [(0.,0), (-2.2,-2)],
               [(-3.8,-2),(-5.5,-2)],
               [(-3.8,2),(-5.5,2)],
               [(-3.8,-4),(-5.5,-4)],
               [(-3.8,4),(-5.5,4)],
               [(0.,0), (-2.2,4)],
               [(0.,0), (-2.2,-4)],
               [(4,-4.1),(2,0)],
               [(4,-3.2),(2,0)],           
               [(4,-2.3),(2,0)],           
               [(4,-1.4),(2,0)],
               [(4,-0.5),(2,0)],
               [(4,0.4),(2,0)],
               [(4,1.3),(2,0)],
               [(4,2.2),(2,0)],
               [(4,3.1),(2,0)],
               [(4,4),(2,0)]]
        for xy in xys:
            ax.annotate("",xy=xy[0],xytext=xy[1],
            arrowprops=dict(arrowstyle = '->', color = 'g', linewidth = 2))  
        # Put explanation texts on the graph
        zs = [[-5.25, 4.1, "bias", 20, "k", 0],
                [-5.25, -3.9, "pixel", 20, "k", 0],
                [-3.2, 3.8, "1", 30, "k", 0],
                [-3.6, -4.2, r"$X_{3072}$", 30, "k", 0],
                [-3.3, 0.2, r"$\vdots$", 30, "k", 0],
                [-3.3, -0.4, r"$\vdots$", 30, "k", 0],
                [-5.25, 2.1, "pixel", 20, "k", 0],
                [-5.25, -1.9, "pixel", 20, "k", 0],        
                [-3.4, 1.8, r"$X_{1}$", 30, "k", 0],
                [-3.6, -2.2, r"$X_{3071}$", 30, "k", 0],
                [0.2, -0.5, "Deep\nNeural\nNetwork", 20, "k", 0]]
        for z in zs:
            plt.text(z[0], z[1], z[2], fontsize=z[3], color=z[4], rotation=z[5])  
        for i in range(10):
            plt.text(4.1, 4-0.9*i, f"p({names[i]})={ps[i]:.4f}", fontsize=15, color="r")  
    
        # put epoch number up
        plt.text(-1, 4.3, f"epoch {stage*5}", fontsize=40, color="g")   
        # add text to explain 
        txt = ax.annotate('Softmax\nActivation', 
            xy = (1,-1), 
            xytext = (0.5,0.2), 
            textcoords = 'axes fraction', 
            bbox = dict(boxstyle="round", fc="0.9"), 
            arrowprops=dict(arrowstyle = '->', color = 'g', linewidth = 1), 
            fontsize = 20)
        txt = ax.annotate('convolutional\nlayers', 
            xy = (1,1), 
            xytext = (0.5,0.75), 
            textcoords = 'axes fraction', 
            bbox = dict(boxstyle="round", fc="0.9"), 
            arrowprops=dict(arrowstyle = '->', color = 'g', linewidth = 1), 
            fontsize = 20)
        # add rectangle to plot
        ax.add_patch(Rectangle((0,-1), 2, 2,
                     edgecolor = 'k',alpha=0.1)) 
        ax.add_patch(Rectangle((4,-1.6), 2.6, 0.6,
                     facecolor = 'g',alpha=0.25))  
        ax.set_xlim(-7, 7)
        ax.set_ylim(-5, 5)
        plt.axis('off')
        # Add frog picture
        newax = fig.add_axes([0.1, 0.39, 0.23, 0.23])
        newax.imshow(frog)
        newax.axis('off')
        plt.savefig(f"files/ch09/p_frog{stage}.png")
        plt.close(fig)
 
        
 
    
 
        
        
        
        
        
        
        
        
        
        
        
        
        
        