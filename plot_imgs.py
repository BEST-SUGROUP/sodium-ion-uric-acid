import itertools
import numpy as np

import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure()
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize='xx-large',fontweight='heavy')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90,fontsize=14,fontweight='bold',fontproperties='Arial')
    plt.yticks(tick_marks, classes,fontsize=14,fontweight='bold',fontproperties='Arial')
    plt.tick_params(labelsize=14)

    plt.axis("equal")




    ax = plt.gca()
    left, right = plt.xlim()
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(j, i, num,
             verticalalignment='center',
             horizontalalignment="center",
             color="white" if num > thresh else "black")

    plt.ylabel('Ground Truth',fontsize=16,fontweight='bold',fontproperties='Arial')
    plt.xlabel('Predict Results',fontsize=16,fontweight='bold',fontproperties='Arial')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('method_2.png', transparent=True, dpi=800)

    plt.show()

trans_mat = np.array([[56, 0, 0], [0, 76, 3], [0, 0, 46]], dtype=int)

"""method 2"""
if True:
    #label = ["Patt {}".format(i) for i in range(1, trans_mat.shape[0] + 1)]
    label = ['Low','Normal','High']
    plot_confusion_matrix(trans_mat, label)
