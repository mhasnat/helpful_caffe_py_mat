import matplotlib.pyplot as plt

tImg 

n = 32

for i in range(n):
    plt.subplot(6,6,i+1)
    plt.imshow(tImg[i])
    plt.xticks([])
    plt.yticks([])
    plt.title('my title')
    
plt.subplots_adjust(wspace=0.0, hspace=0.0)
plt.savefig('images.png', dpi = 300)
plt.show()