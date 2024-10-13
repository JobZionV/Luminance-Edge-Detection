from detector import Luminance_ed, trial
import sys
import matplotlib.pyplot as plt

if __name__ == "__main__":
    if len(sys.argv) == 1:
        trial()
    
    else:
        ed_img = Luminance_ed(sys.argv[1])
        plt.imshow(ed_img, 'gray')
        plt.show()